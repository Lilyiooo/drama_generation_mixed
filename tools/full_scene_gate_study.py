"""Paired full-story A/B study; default pilot R01, explicit expansion to three runs."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from drama_local.runtime import atomic_json
from tools import state_hybrid_study as study
from tools.resume_state_hybrid import saved_inputs
from service.drama_by_creativity.conservative_scene_gate import ConservativeSceneGate, POLICY

ROOT = REPO / 'output/full_scene_gate_ab_v1'
SOURCE = REPO / 'output/full_pipeline_state_hybrid_v2'
ARMS = ('control_hybrid', 'candidate_gate')
RUNS = ('R01', 'R02', 'R03')
DIMENSIONS = ('剧本逻辑总分', '剧本质量最终总分')


def identifier(arm, run):
    return f'full-gate-{arm}-{run}'


def code_hashes():
    paths = [Path(__file__), REPO / 'full_scene_gate_study.sh']
    paths += [path for folder in ('service/drama_by_creativity', 'drama_local')
              for path in (REPO / folder).rglob('*')
              if path.suffix in {'.py', '.yaml', '.jinja2'} and 'tests' not in path.parts]
    paths += [REPO / 'tools' / name for name in (
        'state_hybrid_study.py', 'resume_state_hybrid.py', 'continue_scene_gate_decision.py',
        'recover_scene_decision_metadata.py', 'repair_scene_gate_calibration.py',
        'scene_gate_decision_study.py', 'scene_gate_calibration.py', 'evaluate_with_drama_evaluator.py')]
    paths += [study.TENCENT_ROOT / 'local_pipeline' / name for name in ('gated_engine.py', 'state_retrieval.py')]
    paths += list((REPO / 'evaluation/drama_evaluator_logic_quality').rglob('*.py'))
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(set(paths))}


def prepare(source, root, source_id):
    if (root / 'manifest.json').exists():
        manifest, _ = load(root)
        if manifest['source'] != str(source) or manifest['source_story_id'] != source_id:
            raise ValueError('Prepared experiment belongs to another source')
        return
    if root == source or (root.exists() and any(root.iterdir())):
        raise ValueError('Prepare requires a fresh directory; existing outputs are never overwritten')
    inputs = saved_inputs(source, source_id)
    memory = study.StateLifecycleMemory(source_id, source)
    initial = study.read_json(memory.root / 'initial_state.json')
    assets = study.assets_from_request(study.make_request(inputs, 'hybrid', 0, source_id))
    memory.validate_initial_result({'initial_state': initial['initial_state'], 'evidence': initial['evidence']}, assets)
    bundle = {'inputs': inputs, 'assets': assets, 'initial': initial}
    manifest = {
        'version': 1, 'source': str(source), 'source_story_id': source_id,
        'arms': list(ARMS), 'runs': list(RUNS), 'default_runs': ['R01'], 'episodes': 60,
        'bundle_sha256': study.digest(bundle), 'implementation': code_hashes(),
        'policy': POLICY, 'generation_model': 'Qwen3.6-27B', 'evaluation_model': 'Qwen3.8-27B',
        'scope': 'Paired assets, not paired random streams; independent native and lifecycle memory per trajectory.',
        'initial_state_origin': 'Only validated opening state copied; no source scripts or dynamic memory copied.',
        'control': 'Native narrative memory + state lifecycle + hybrid; no scene gate.',
        'candidate': 'Control plus frozen factorized checker; one hard-conflict repair and recheck; semantic fallback to original.',
        'failure_policy': 'API/schema errors fail and resume; semantic uncertainty is recorded, not called pass.',
        'evaluation': {'temperature': '0', 'enable_thinking': 'false', 'review_workers': 2,
                       'audit_workers': 2, 'arbitration_workers': 2,
                       'protocol': 'logic_quality_v2.4', 'adapter': 'evaluate_with_drama_evaluator.py'},
    }
    atomic_json(root / 'frozen_inputs.json', bundle)
    for arm in ARMS:
        for run in RUNS:
            directory = root / arm / run
            atomic_json(directory / '04_future_map/future_map.json', inputs['future_map'])
            atomic_json(directory / '04_future_map/episode_contributions.json', inputs['episode_contributions'])
            atomic_json(directory / 'generation_assets.json', assets)
            state_root = directory / 'state_lifecycle' / identifier(arm, run)
            atomic_json(state_root / 'story_assets.json', assets)
            atomic_json(state_root / 'initial_state.json', initial)
            atomic_json(directory / 'arm_binding.json', {'arm': arm, 'run': run, 'manifest_sha256': study.digest(manifest)})
    atomic_json(root / 'manifest.json', manifest)


def load(root):
    manifest = study.read_json(root / 'manifest.json')
    bundle = study.read_json(root / 'frozen_inputs.json')
    if manifest['version'] != 1 or manifest['implementation'] != code_hashes():
        raise ValueError('Frozen experiment implementation changed; do not mix versions')
    if manifest['bundle_sha256'] != study.digest(bundle):
        raise ValueError('Frozen experiment assets changed')
    return manifest, bundle


def verify_directory(root, arm, run, manifest, bundle):
    directory = root / arm / run
    expected = {'arm': arm, 'run': run, 'manifest_sha256': study.digest(manifest)}
    if study.read_json(directory / 'arm_binding.json') != expected:
        raise ValueError('Arm/run binding changed')
    study.verify_static_inputs(directory, bundle['inputs'])
    state_root = directory / 'state_lifecycle' / identifier(arm, run)
    for path, expected in ((state_root / 'initial_state.json', bundle['initial']),
                           (state_root / 'story_assets.json', bundle['assets']),
                           (directory / 'generation_assets.json', bundle['assets'])):
        if study.read_json(path) != expected:
            raise ValueError(f'Opening state/assets changed: {path}')
    return directory


def scores(directory):
    destination = directory / 'drama_evaluations_logic_quality_qwen38_v24/full_60_episodes'
    if not (destination / 'scores.json').exists():
        return None
    episodes = study.scripts(directory)
    if len(episodes) != 60:
        raise ValueError('Scores without a complete 60-episode script')
    script = '\n\n'.join(f'第{index + 1}集\n\n{item.strip()}'
                         for index, item in enumerate(episodes)) + '\n'
    protocol = study.read_json(destination / 'local_protocol.json')
    payload = study.read_json(destination / 'scores.json')
    if (protocol['script_sha256'] != hashlib.sha256(script.encode()).hexdigest()
            or protocol['model'] != 'Qwen3.8-27B' or payload['model'] != 'Qwen3.8-27B'
            or protocol['temperature'] != '0' or protocol['enable_thinking'] != 'false'):
        raise ValueError('Evaluation no longer matches this script/protocol')
    result = {name: payload['scores'][name]['final_score'] for name in DIMENSIONS}
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100
           for value in result.values()):
        raise ValueError('Invalid final scores')
    return result


def report(root, runs, write=False):
    manifest, bundle = load(root)
    rows = []
    for arm in ARMS:
        for run in runs:
            directory = verify_directory(root, arm, run, manifest, bundle)
            status = study.directory_progress(directory, 'hybrid', identifier(arm, run))
            outcomes = [study.read_json(path) for path in (directory / 'scene_state_gate').glob('*/E*/outcome.json')]
            row = {'arm': arm, 'run': run, 'complete_episodes': status['complete'], 'scores': scores(directory),
                   'gate_checked': len(outcomes)}
            for key in ('repair_attempted', 'repair_accepted', 'fallback_to_original', 'unresolved', 'text_changed'):
                row[key] = sum(bool(item[key]) for item in outcomes)
            if arm == 'candidate_gate' and len(outcomes) < status['complete']:
                raise ValueError('Candidate has completed episodes without gate outcomes')
            if arm == 'control_hybrid' and outcomes:
                raise ValueError('Control must not contain gate outcomes')
            rows.append(row)
            print(f'{arm}/{run}: episodes={status["complete"]}/60 scored={row["scores"] is not None} '
                  f'checked={len(outcomes)} repaired={row["repair_accepted"]} unresolved={row["unresolved"]}', flush=True)
    pairs = []
    for run in runs:
        control, candidate = [next(row for row in rows if row['run'] == run and row['arm'] == arm) for arm in ARMS]
        if control['scores'] is not None and candidate['scores'] is not None:
            pairs.append({'run': run, 'candidate_minus_control': {
                name: round(candidate['scores'][name] - control['scores'][name], 4) for name in DIMENSIONS}})
    result = {'rows': rows, 'paired_differences': pairs, 'scope': manifest['scope']}
    if write:
        atomic_json(root / 'reports' / ('_'.join(runs) + '.json'), result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return rows


def run_worker(root, arm, run, action, limit):
    manifest, bundle = load(root)
    directory = verify_directory(root, arm, run, manifest, bundle)
    with (directory / '.trajectory.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if action == 'generate':
            os.environ['DRAMA_SCENE_STATE_GATE'] = 'off'
            factory = ConservativeSceneGate.factory if arm == 'candidate_gate' else None
            asyncio.run(study.generate_directory(bundle['inputs'], directory, 'hybrid', identifier(arm, run),
                                                limit, scene_gate_factory=factory))
        else:
            if study.directory_progress(directory, 'hybrid', identifier(arm, run))['complete'] != 60:
                raise ValueError('Finish all scripts and both memories before evaluation')
            if scores(directory) is not None:
                return
            environment = dict(os.environ, DRAMA_EVAL_TEMPERATURE='0', DRAMA_EVAL_ENABLE_THINKING='false')
            subprocess.run([sys.executable, str(REPO / 'tools/evaluate_with_drama_evaluator.py'),
                            str(directory), '--execute-api'], cwd=REPO, env=environment, check=True)
            if scores(directory) is None:
                raise RuntimeError('Evaluator returned without valid scores')


def dispatch(root, runs, action, limit, workers):
    rows = report(root, runs)
    pending = [row for row in rows if (row['complete_episodes'] < limit if action == 'generate' else row['scores'] is None)]
    if action == 'evaluate' and any(row['complete_episodes'] != 60 for row in rows):
        raise ValueError('Finish generation in all selected trajectories before evaluation')
    print(f'pending={len(pending)} stage={action}', flush=True)

    def launch(row):
        arm, run = row['arm'], row['run']
        path = root / 'logs' / f'{action}_{arm}_{run}.log'
        path.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__)), action, '--root', str(root), '--run', run,
                   '--arm', arm, '--worker', '--limit', str(limit), '--execute-api']
        with path.open('a') as stream:
            result = subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
        row = {'arm': arm, 'run': run, 'returncode': result.returncode, 'log': str(path)}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        return result.returncode

    with (root / '.tasks.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            failures = [future.result() for future in as_completed([executor.submit(launch, row) for row in pending])]
    report(root, runs)
    if any(failures):
        raise SystemExit('Some tasks failed; successful scripts and checkpoints retained; rerun the same command')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'status', 'generate', 'evaluate', 'report'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--source-story-id', default='APID-test-001')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--run', choices=RUNS, default='R01')
    group.add_argument('--all-runs', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--limit', type=int, default=60)
    parser.add_argument('--execute-api', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--arm', choices=ARMS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.workers < 1 or not 1 <= args.limit <= 60:
        parser.error('Positive workers and limit between 1 and 60 required')
    root = args.root.resolve()
    runs = RUNS if args.all_runs else (args.run,)
    if args.action == 'prepare':
        prepare(args.source.resolve(), root, args.source_story_id)
        report(root, runs)
    elif args.worker:
        if not args.execute_api or args.arm is None or args.all_runs or args.action not in {'generate', 'evaluate'}:
            parser.error('Worker requires one arm/run and explicit API permission')
        run_worker(root, args.arm, args.run, args.action, args.limit)
    elif args.action in {'status', 'report'} or not args.execute_api:
        report(root, runs, write=args.action == 'report')
        if args.action in {'generate', 'evaluate'}:
            print('Read-only; add --execute-api to execute pending tasks')
    else:
        dispatch(root, runs, args.action, args.limit, args.workers)


if __name__ == '__main__':
    main()
