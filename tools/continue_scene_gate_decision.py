"""Continue R02/R03 with metadata compatibility while retaining frozen R01 results."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
import fcntl
import io
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from drama_local.runtime import atomic_json
from tools import recover_scene_decision_metadata as metadata
from tools import scene_gate_decision_study as study
from service.drama_by_creativity import scene_state_gate as gate

ROOT = REPO / 'output/scene_gate_decision_v2_continued'
CONTINUATION_RUNS = ('R02', 'R03')


def dependencies():
    return {**metadata.dependencies(), 'continuation': study.indexed.file_hash(Path(__file__))}


def r01_summary(root):
    with redirect_stdout(io.StringIO()):
        return metadata.report(root)


def prepare(source, r01_root, root):
    if (root / 'manifest.json').exists():
        load(root)
        return
    if root in {source, r01_root} or (root.exists() and any(root.iterdir())):
        raise ValueError('Use a fresh continuation directory')
    _, cases = study.load(source)
    summary = r01_summary(r01_root)
    recovery_manifest = gate.read(r01_root / 'manifest.json')
    if Path(recovery_manifest['source']).resolve() != source or recovery_manifest['run'] != 'R01':
        raise ValueError('R01 recovery belongs to a different source or run')
    expected = {(case['id'], arm, 'R01') for case in cases for arm in study.ARMS}
    rows = summary['results']
    if (len(rows) != len(expected) or {(row['case'], row['arm'], row['run']) for row in rows} != expected or
            not all(row['complete'] for row in rows)):
        raise ValueError('R01 must be complete in both arms before continuation')
    for case in cases:
        for run in CONTINUATION_RUNS:
            for arm in study.ARMS:
                directory = study.DecisionProbe(source, case, run, arm).directory
                if (directory / 'decision.json').exists() or any((directory / 'attempts').glob('**/*.json')):
                    raise ValueError('Original R02/R03 already contain attempts; inspect before creating a duplicate run')
    atomic_json(root / 'cases.json', cases)
    atomic_json(root / 'r01_results.json', rows)
    atomic_json(root / 'manifest.json', {'source': str(source), 'r01_root': str(r01_root),
        'cases_sha256': gate.digest(cases), 'r01_sha256': gate.digest(rows), 'implementation': dependencies(),
        'scope': 'Same frozen cases and initial prompts; R01 retained, R02/R03 use online metadata compatibility. Repeats are not independent stories.'})


def load(root):
    manifest = gate.read(root / 'manifest.json')
    cases, rows = gate.read(root / 'cases.json'), gate.read(root / 'r01_results.json')
    if (manifest['implementation'] != dependencies() or manifest['cases_sha256'] != gate.digest(cases) or
            manifest['r01_sha256'] != gate.digest(rows)):
        raise ValueError('Frozen continuation code or inputs changed')
    _, original_cases = study.load(Path(manifest['source']))
    if original_cases != cases or r01_summary(Path(manifest['r01_root']))['results'] != rows:
        raise ValueError('Original cases or retained R01 results changed')
    return manifest, cases, rows


def compatible_validate(raw, scenes, states):
    try:
        result, changes = metadata.compatible_result(raw, scenes, states)
    except ValueError as error:
        message = str(error)
        if 'memory_conflicts.state_ids' in message:
            message += ('。memory_conflicts只允许历史M记录之间的矛盾，至少两条不同M编号，不得引用S编号或编造第二条M。'
                        '候选场次与历史记录的冲突应放在checks中，state_basis描述历史本身是否一致，不是场次是否冲突。')
        raise ValueError(message) from error
    return {**result, 'metadata_compatibility': changes}


class ContinuedProbe(study.DecisionProbe):
    def __init__(self, root, case, run, arm):
        if arm != 'judge' and run not in CONTINUATION_RUNS:
            raise ValueError('R01 generation is frozen and must not be resampled')
        super().__init__(root, case, run, arm)
        self.binding['continuation_implementation'] = dependencies()

    def protocol(self):
        if self.arm == 'control_indexed':
            return super().protocol()
        return study.decision.RULES, lambda raw: compatible_validate(raw, self.scenes, self.catalog)


def report(root):
    manifest, cases, retained = load(root)
    rows = [{**row, 'provenance': 'retained_r01'} for row in retained]
    for case in cases:
        for run in CONTINUATION_RUNS:
            for arm in study.ARMS:
                probe = ContinuedProbe(root, case, run, arm)
                row = {'case': case['id'], 'run': run, 'arm': arm, 'complete': False,
                       'expected_status': case['expected_status'], 'expected_kind': case['expected_kind'],
                       'provenance': 'continuation_metadata_compatible'}
                if (probe.directory / 'decision.json').exists():
                    result = probe.read_result()
                    row.update(complete=True, status=result['status'],
                               classifications=sorted({check['classification'] for check in result.get('checks', [])}),
                               metadata_compatible=bool(result.get('metadata_compatibility')))
                rows.append(row)
    lookup = {case['id']: case for case in cases}
    for row in rows:
        judge = ContinuedProbe(root, lookup[row['case']], row['run'], 'judge')
        if (judge.directory / 'decision.json').exists():
            row['judge_status'] = judge.read_result()['status']
    for arm in study.ARMS:
        group = [row for row in rows if row['arm'] == arm]
        print(f"{arm}: complete={sum(row['complete'] for row in group)}/{len(group)}")
        for run in study.RUNS:
            selected = [row for row in group if row['run'] == run]
            print(f"  {run}={sum(row['complete'] for row in selected)}/{len(selected)}")
        for expected in ('fail', 'pass', 'uncertain'):
            selected = [row for row in group if row['expected_status'] == expected]
            print(json.dumps({'arm': arm, 'expected_status': expected, 'total': len(selected),
                              'complete': sum(row['complete'] for row in selected),
                              'matched': sum(row.get('status') == expected for row in selected)}, ensure_ascii=False))
    print(f"judge_complete={sum('judge_status' in row for row in rows if row['arm'] == study.ARMS[0])}/{len(cases) * 3}; auto_repairs=0")
    return {'scope': manifest['scope'], 'results': rows}


async def run_one(root, case, run, arm):
    probe = ContinuedProbe(root, case, run, arm)
    probe.directory.mkdir(parents=True, exist_ok=True)
    with (probe.directory / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        await probe.run()
    return f'{arm}/{case["id"]}/{run}'


def pending_tasks(cases, summary, evaluate):
    key = 'judge_status' if evaluate else None
    pending = {(row['case'], row['run'], 'judge' if evaluate else row['arm'])
               for row in summary['results'] if (key not in row if evaluate else not row['complete'])}
    return [(case, run, arm) for case in cases for run in (study.RUNS if evaluate else CONTINUATION_RUNS)
            for arm in (('judge',) if evaluate else study.ARMS) if (case['id'], run, arm) in pending]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'status', 'report', 'evaluate'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--source-dir', type=Path, default=study.ROOT)
    parser.add_argument('--r01-root', type=Path, default=metadata.ROOT)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--execute-api', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error('--workers must be between 1 and 4')
    root = args.root.resolve()
    if args.action == 'prepare':
        prepare(args.source_dir.resolve(), args.r01_root.resolve(), root)
        print(f'Prepared {root}; no API calls')
        return
    _, cases, _ = load(root)
    summary = report(root)
    if args.action in {'status', 'report'}:
        if args.action == 'report':
            atomic_json(root / 'summary.json', summary)
        return
    evaluate = args.action == 'evaluate'
    if evaluate and any(not row['complete'] for row in summary['results']):
        raise ValueError('Finish both Qwen3.6 arms for all three runs before independent review')
    tasks = pending_tasks(cases, summary, evaluate)
    print(f'pending={len(tasks)} stage={args.action}; retained_R01=True')
    if not args.execute_api or not tasks:
        print('No API calls; add --execute-api if pending')
        return
    model, url = ('Qwen3.8-27B', os.environ.get('DRAMA_EVAL_BASE_URL', 'http://127.0.0.1:8001/v1')) if evaluate else ('Qwen3.6-27B', os.environ.get('DRAMA_LLM_BASE_URL', 'http://127.0.0.1:8000/v1'))
    study.original.base.verify_server(url, model)
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(asyncio.run, run_one(root, case, run, arm)): (arm, case['id'], run)
                   for case, run, arm in tasks}
        for future in as_completed(futures):
            try:
                print('completed=' + future.result(), flush=True)
            except Exception as error:
                failures.append(futures[future])
                print(json.dumps({'task': futures[future], 'error': str(error)}, ensure_ascii=False), flush=True)
    summary = report(root)
    atomic_json(root / 'summary.json', summary)
    if failures:
        raise SystemExit('Some checks failed; successful results retained; rerun this continuation command')


if __name__ == '__main__':
    main()
