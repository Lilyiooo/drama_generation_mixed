"""Resume the full gate study after lifecycle extraction output truncation."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import jinja2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from drama_local.runtime import atomic_json
from service.drama_by_creativity.local_llm import LLM
from service.drama_by_creativity.state_lifecycle_memory import StateLifecycleMemory, EXTRACTION_RULES, digest
from tools import full_scene_gate_study as study
from tools.recover_full_scene_gate import FeedbackGate

VERSION = 'state_extraction_tokens_v2'
TRUNCATED = 'Local LLM output truncated'


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dependencies():
    return {str(path): file_hash(path) for path in (Path(__file__), REPO / 'recover_full_scene_gate_state.sh')}


def previous_truncation(memory, episode_id, script):
    prefix = f'E{episode_id + 1:02d}_'
    records = []
    for path in sorted((memory.root / 'failures').glob(prefix + '*.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        if record.get('script_sha256') == digest(script) and TRUNCATED in str(record.get('error', '')):
            records.append({'path': str(path), 'sha256': file_hash(path)})
    return records


async def extraction_with_extended_limit(self, ctx, episode_id, script):
    prior = previous_truncation(self, episode_id, script)
    limits = (16384, 24576) if len(prior) >= 3 else (4096, 8192, 16384)
    client = LLM(model='Qwen3.6-27B', system_prompt='只抽取剧本中实际发生的状态变化。',
                 temperature=0.0, top_p=1.0, top_k=50, max_length=131072,
                 max_new_tokens=limits[0], template=jinja2.Template('{{ Prompt }}'))
    client.thinking = False
    prompt = (EXTRACTION_RULES + f'\n当前集：第{episode_id + 1}集\n上一完整状态：\n'
              + json.dumps(self.state, ensure_ascii=False) + '\n本集最终剧本：\n' + script)
    if prior:
        prompt += ('\n恢复说明：此前对同一剧本的完整JSON输出因长度上限被截断。严格遵守原有数量限制，'
                   '只写本集真正新增或改变的短状态，不复制上一状态，不扩写解释；保持事实判断不变。')
    attempt_root = self.root / 'state_extraction_recovery' / f'E{episode_id + 1:02d}' / uuid4().hex
    atomic_json(attempt_root / 'binding.json', {
        'version': VERSION, 'episode_id': episode_id, 'script_sha256': digest(script),
        'previous_truncated_attempts': prior, 'limits': list(limits), 'previous_state_sha256': digest(self.state),
    })
    last_error = ''
    for attempt, limit in enumerate(limits, 1):
        client.max_tokens = limit
        suffix = (f'\n上次校验错误：{last_error}。请重新输出完整且精简的JSON；不得复制上一完整状态。'
                  if last_error else '')
        try:
            response = await client.request(ctx, {'Prompt': prompt + suffix,
                                                  'EpisodeNumber': str(episode_id + 1)},
                                            f'state_recovery_E{episode_id + 1:02d}_{attempt}')
            result = self.engine.extract_json_value(response.response)
            self.validate_extraction(result)
            atomic_json(attempt_root / f'attempt_{attempt}.json', {
                'max_tokens': limit, 'accepted': True, 'response': response.response, 'result': result})
            return result
        except (ValueError, TypeError, KeyError, RuntimeError) as error:
            last_error = str(error)
            atomic_json(attempt_root / f'attempt_{attempt}.json', {
                'max_tokens': limit, 'error': last_error,
                'response': getattr(client, 'last_output', None), 'script_sha256': digest(script)})
    raise RuntimeError(f'State extraction recovery failed for episode {episode_id + 1}: {last_error}')


def profile_dir(root, run):
    return root / 'recovery' / VERSION / run


def prepare(root, run):
    directory = profile_dir(root, run)
    if (directory / 'manifest.json').exists():
        load(root, run)
        return
    manifest, bundle = study.load(root)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Recovery directory is nonempty without a manifest')
    with (root / '.tasks.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = study.report(root, (run,))
        protected = [root / 'manifest.json', root / 'frozen_inputs.json']
        snapshots = []
        for arm in study.ARMS:
            trajectory = study.verify_directory(root, arm, run, manifest, bundle)
            for folder in ('05_drama', 'state_lifecycle', 'scene_state_gate'):
                protected.extend(path for path in (trajectory / folder).rglob('*.json')
                                 if 'failures' not in path.parts and 'state_extraction_recovery' not in path.parts)
            for path in (trajectory / 'narrative_memory').rglob('*.json'):
                target = directory / 'before' / arm / path.relative_to(trajectory)
                atomic_json(target, json.loads(path.read_text(encoding='utf-8')))
                snapshots.append({'source': str(path.relative_to(root)), 'snapshot': str(target.relative_to(root)),
                                  'sha256': file_hash(path)})
            protected.extend([trajectory / 'arm_binding.json', trajectory / 'generation_assets.json'])
        atomic_json(directory / 'manifest.json', {
            'version': VERSION, 'run': run, 'created_at': datetime.now(timezone.utc).isoformat(),
            'implementation': dependencies(), 'original_manifest_sha256': file_hash(root / 'manifest.json'),
            'retained_files': {str(path.relative_to(root)): file_hash(path) for path in sorted(set(protected))},
            'mutable_memory_snapshots': snapshots, 'starting_progress': rows,
            'scope': 'Reuse saved scripts. Existing repeated truncation starts at 16384 then 24576 tokens; '
                     'new extraction keeps 4096/8192 before 16384. Validation, state merge, gate and evaluator stay unchanged.',
        })


def load(root, run):
    study.load(root)
    profile = json.loads((profile_dir(root, run) / 'manifest.json').read_text(encoding='utf-8'))
    if profile['version'] != VERSION or profile['run'] != run or profile['implementation'] != dependencies():
        raise ValueError('State recovery code or run changed')
    for name, checksum in profile['retained_files'].items():
        if file_hash(root / name) != checksum:
            raise ValueError(f'Retained artifact changed before recovery: {name}')
    return profile


def worker(root, arm, run, limit):
    load(root, run)
    manifest, bundle = study.load(root)
    directory = study.verify_directory(root, arm, run, manifest, bundle)
    with (directory / '.trajectory.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ['DRAMA_SCENE_STATE_GATE'] = 'off'
        original_extract = StateLifecycleMemory.extract
        StateLifecycleMemory.extract = extraction_with_extended_limit
        try:
            factory = FeedbackGate.factory if arm == 'candidate_gate' else None
            asyncio.run(study.study.generate_directory(bundle['inputs'], directory, 'hybrid',
                        study.identifier(arm, run), limit, scene_gate_factory=factory))
        finally:
            StateLifecycleMemory.extract = original_extract
    load(root, run)


def generate(root, run, limit, workers):
    with (root / '.tasks.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        load(root, run)
        rows = study.report(root, (run,))
        pending = [row for row in rows if row['complete_episodes'] < limit]
        print(f'pending={len(pending)} recovery={VERSION}', flush=True)

        def launch(row):
            path = profile_dir(root, run) / f'generate_{row["arm"]}.log'
            command = [sys.executable, str(Path(__file__)), 'generate', '--root', str(root), '--run', run,
                       '--arm', row['arm'], '--worker', '--limit', str(limit), '--execute-api']
            with path.open('a') as stream:
                result = subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
            print(json.dumps({'arm': row['arm'], 'run': run, 'returncode': result.returncode,
                              'log': str(path)}, ensure_ascii=False), flush=True)
            return result.returncode

        with ThreadPoolExecutor(max_workers=workers) as executor:
            failures = [future.result() for future in as_completed([executor.submit(launch, row) for row in pending])]
        load(root, run)
        rows = study.report(root, (run,))
        if any(failures) or any(row['complete_episodes'] < limit for row in rows):
            raise SystemExit('State recovery incomplete; inspect the v2 recovery log. Successful outputs are retained.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'status', 'generate', 'report'))
    parser.add_argument('--root', type=Path, default=study.ROOT)
    parser.add_argument('--run', choices=study.RUNS, default='R01')
    parser.add_argument('--limit', type=int, default=60)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--execute-api', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--arm', choices=study.ARMS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.limit <= 60 or args.workers < 1:
        parser.error('Require limit 1..60 and positive workers')
    root = args.root.resolve()
    if args.action == 'prepare':
        prepare(root, args.run)
    elif args.worker:
        if args.action != 'generate' or not args.execute_api or args.arm is None:
            parser.error('Worker requires generation, arm and API permission')
        worker(root, args.arm, args.run, args.limit)
        return
    else:
        profile = load(root, args.run)
        if args.action == 'generate' and args.execute_api:
            generate(root, args.run, args.limit, args.workers)
            return
        if args.action == 'report':
            print(json.dumps({'recovery_version': VERSION, 'starting_progress': profile['starting_progress'],
                              'scope': profile['scope']}, ensure_ascii=False, indent=2))
    study.report(root, (args.run,), write=args.action == 'report')
    print(f'recovery={VERSION}; retained scripts verified; no model calls', flush=True)


if __name__ == '__main__':
    main()
