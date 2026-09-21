"""Versioned recovery with actionable schema feedback; frozen classifier stays unchanged."""

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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from drama_local.runtime import atomic_json
from service.drama_by_creativity import scene_state_gate as original
from service.drama_by_creativity.conservative_scene_gate import ConservativeSceneGate, decision
from tools.continue_scene_gate_decision import compatible_validate
from tools.repair_scene_gate_calibration import evidence_catalog
from tools import full_scene_gate_study as study

VERSION = 'check_feedback_v1'
FEEDBACK_RULES = '''以下是上一份无效响应的校验诊断，不是新的历史事实或预设结论。重新阅读原始M/S证据，输出完整JSON。
memory_conflicts只能表示历史M记录之间对同一对象、同一时点的互斥，不是前一集与新场次之间缺少转场、人物未出场或缺乏铺垫。
state_basis=conflicting时，checks.state_ids必须包含某个真实相关的已声明memory_conflicts组的全部M编号。
不要为了通过校验把不相关编号一股脑加进checks；先核实是否真的存在M与M互斥，再让分组和引用一致。
如果证据确实支持记忆冲突就保留，不能为了通过校验改判无冲突。如果只是不知道某项必要事实，依据原始规则判断insufficient。
M与候选S之间的互斥应在checks判断；缺少铺垫不自动等于硬冲突。先前response中的推测不是证据。
不要伪造第二条M、自动补齐引用、移除真实冲突或按预期分数修改判断。判定组合、证据要求与初始任务完全相同。'''


def feedback(raw, error):
    diagnostic = {'validation_error': str(error)}
    try:
        value = json.loads(raw.strip().removeprefix('```json').removesuffix('```').strip())
    except (ValueError, TypeError, AttributeError):
        value = None
    if isinstance(value, dict):
        groups = value.get('memory_conflicts')
        checks = value.get('checks')
        if isinstance(groups, list) and isinstance(checks, list):
            declared = [group.get('state_ids') for group in groups if isinstance(group, dict)]
            diagnostic['declared_memory_groups'] = declared
            diagnostic['conflicting_checks'] = [
                {'index': index, 'scene_id': check.get('scene_id'), 'state_ids': check.get('state_ids')}
                for index, check in enumerate(checks) if isinstance(check, dict) and check.get('state_basis') == 'conflicting']
    diagnostic['previous_invalid_response_excerpt'] = str(raw)[:16000]
    return FEEDBACK_RULES + '\n' + json.dumps(diagnostic, ensure_ascii=False)


def validate_with_feedback(raw, scenes, catalog):
    try:
        return compatible_validate(raw, scenes, catalog)
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError(feedback(raw, error)) from error


class FeedbackGate(ConservativeSceneGate):
    async def check(self, ctx, scenes, slot='check_before'):
        if (self.directory / f'{slot}.json').exists():
            return await super().check(ctx, scenes, slot)
        packet = {**self.packet(scenes), 'scene_evidence_catalog': evidence_catalog(scenes)}
        user = json.dumps(packet, ensure_ascii=False)
        recovered_slot = f'{slot}_{VERSION}'
        input_path = self.directory / f'{recovered_slot}_input.json'
        expected = {'version': VERSION, 'binding_sha256': original.digest(self.binding),
                    'packet_sha256': original.digest(packet)}
        if input_path.exists():
            frozen = original.read(input_path)
            if any(frozen.get(key) != value for key, value in expected.items()):
                raise ValueError('Recovery request input changed')
        else:
            frozen = {**expected, 'seed_attempt': None, 'user': user}
            attempts = sorted((self.directory / 'attempts' / slot).glob('*/*.json'),
                              key=lambda path: (path.stat().st_mtime_ns, str(path)), reverse=True)
            for path in attempts:
                record = original.read(path)
                request = record.get('request', {})
                if (request.get('binding') != self.binding or request.get('system') != decision.RULES
                        or request.get('user') != user or request.get('model') != 'Qwen3.6-27B'):
                    raise ValueError('Original failed attempt does not match this frozen check')
                raw = record.get('raw')
                if not isinstance(raw, str) or not record.get('error'):
                    continue
                try:
                    compatible_validate(raw, scenes, self.catalog)
                except (ValueError, TypeError, KeyError) as error:
                    frozen['seed_attempt'] = {'path': str(path), 'sha256': file_hash(path)}
                    frozen['user'] += '\n' + feedback(raw, error)
                    break
            atomic_json(input_path, frozen)
        if frozen['seed_attempt'] is not None:
            source = frozen['seed_attempt']
            if file_hash(Path(source['path'])) != source['sha256']:
                raise ValueError('Original failed response changed')
        return await self.request(ctx, recovered_slot, decision.RULES, frozen['user'],
                                  lambda raw: validate_with_feedback(raw, scenes, self.catalog))

    async def apply(self, ctx, scenes):
        selected = await super().apply(ctx, scenes)
        path = self.directory / 'outcome.json'
        outcome = original.read(path)
        outcome['recovery_protocol'] = VERSION
        outcome['recovery_implementation'] = dependencies()
        atomic_json(path, outcome)
        return selected


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dependencies():
    return {str(path): file_hash(path) for path in (Path(__file__), REPO / 'recover_full_scene_gate.sh')}


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
        for arm in study.ARMS:
            trajectory = study.verify_directory(root, arm, run, manifest, bundle)
            for folder in ('05_drama', 'state_lifecycle', 'scene_state_gate'):
                protected.extend(path for path in (trajectory / folder).rglob('*.json') if 'failures' not in path.parts)
            native = trajectory / 'narrative_memory'
            for path in native.rglob('*.json'):
                atomic_json(directory / 'before' / arm / path.relative_to(trajectory), original.read(path))
            protected.extend([trajectory / 'arm_binding.json', trajectory / 'generation_assets.json'])
        atomic_json(directory / 'manifest.json', {
            'version': VERSION, 'run': run, 'created_at': datetime.now(timezone.utc).isoformat(),
            'implementation': dependencies(), 'original_manifest_sha256': file_hash(root / 'manifest.json'),
            'retained_files': {str(path.relative_to(root)): file_hash(path) for path in sorted(set(protected))},
            'starting_progress': rows,
            'scope': 'Completed outputs retained. New schema retries include invalid response and actionable diagnostics. '
                     'Original checker rules, validator, classifier, repair and evaluator are unchanged; no automatic evidence edits.',
        })


def load(root, run):
    study.load(root)
    profile = original.read(profile_dir(root, run) / 'manifest.json')
    if profile['version'] != VERSION or profile['run'] != run or profile['implementation'] != dependencies():
        raise ValueError('Recovery code or run changed')
    for name, checksum in profile['retained_files'].items():
        if file_hash(root / name) != checksum:
            raise ValueError(f'Retained original artifact changed: {name}')
    return profile


def worker(root, arm, run, limit):
    load(root, run)
    manifest, bundle = study.load(root)
    directory = study.verify_directory(root, arm, run, manifest, bundle)
    with (directory / '.trajectory.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ['DRAMA_SCENE_STATE_GATE'] = 'off'
        factory = FeedbackGate.factory if arm == 'candidate_gate' else None
        asyncio.run(study.study.generate_directory(bundle['inputs'], directory, 'hybrid', study.identifier(arm, run),
                                                   limit, scene_gate_factory=factory))
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
            print(json.dumps({'arm': row['arm'], 'run': run, 'returncode': result.returncode, 'log': str(path)}), flush=True)
            return result.returncode

        with ThreadPoolExecutor(max_workers=workers) as executor:
            failures = [future.result() for future in as_completed([executor.submit(launch, row) for row in pending])]
        load(root, run)
        rows = study.report(root, (run,))
        if any(failures) or any(row['complete_episodes'] < limit for row in rows):
            raise SystemExit('Recovery incomplete; inspect recovery logs. Successful outputs are retained.')


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
    print(f'recovery={VERSION}; original code and retained outputs verified; no model calls', flush=True)


if __name__ == '__main__':
    main()
