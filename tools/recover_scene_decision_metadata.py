"""Offline recovery of non-decisional domain metadata; never change decision evidence."""

import argparse
import copy
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from drama_local.runtime import atomic_json
from service.drama_by_creativity import scene_state_gate as gate
from tools import scene_gate_decision_study as study

ROOT = REPO / 'output/scene_gate_decision_v2_metadata_recovery'


def dependencies():
    return {**study.dependencies(), 'metadata_recovery': study.indexed.file_hash(Path(__file__))}


def compatible_result(raw, scenes, states):
    value = json.loads(raw.strip().removeprefix('```json').removesuffix('```').strip())
    if not isinstance(value, dict):
        raise ValueError('Require a JSON object')
    value = copy.deepcopy(value)
    changes = []
    checks = value.get('checks')
    if isinstance(checks, list):
        for index, check in enumerate(checks):
            if isinstance(check, dict) and check.get('domain') in gate.FIELDS:
                changes.append({'check_index': index, 'reported_domain': check['domain'], 'validation_domain': 'other'})
                check['domain'] = 'other'
    result = study.decision.validate_decision(json.dumps(value, ensure_ascii=False), scenes, states)
    for change in changes:
        check = result['checks'][change['check_index']]
        check['domain'] = change['reported_domain']
        check['domain_namespace'] = 'state_field'
    return result, changes


def validate_attempt(record, probe):
    request = record.get('request', {})
    system, _ = probe.protocol()
    user = json.dumps(probe.packet(probe.scenes), ensure_ascii=False)
    if (request.get('model') != 'Qwen3.6-27B' or request.get('binding') != probe.binding or
            request.get('system') != system or request.get('user') != user or
            request.get('temperature') != 0 or request.get('max_tokens') != 4096):
        raise ValueError('Attempt does not match the frozen original request')
    if not re.fullmatch(r'checks\[\d+\] has invalid scene or domain', str(record.get('error', ''))):
        raise ValueError('Not a domain-validation failure; cannot recover truncated/API/other failures')
    result, changes = compatible_result(record['raw'], probe.scenes, probe.catalog)
    if not changes:
        raise ValueError('No recognized state-field metadata to recover')
    return result, changes


def first_recoverable(probe, snapshots):
    rejected = []
    for snapshot in snapshots:
        path = Path(snapshot['path'])
        if study.indexed.file_hash(path) != snapshot['sha256']:
            raise ValueError(f'Original attempt changed: {path}')
        record = gate.read(path)
        try:
            result, changes = validate_attempt(record, probe)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            rejected.append({'path': str(path), 'error': str(error)})
        else:
            return {'source_attempt': str(path), 'raw': record['raw'], 'result': result,
                    'metadata_compatibility': changes, 'rejected_earlier_attempts': rejected}
    return None


def recover(source, root, run):
    if (root / 'manifest.json').exists():
        report(root)
        return
    if root == source or (root.exists() and any(root.iterdir())):
        raise ValueError('Use a fresh recovery directory, not the original study')
    _, cases = study.load(source)
    files = [source / 'manifest.json', source / 'cases.json']
    rows, records = [], {}
    for case in cases:
        for arm in study.ARMS:
            probe = study.DecisionProbe(source, case, run, arm)
            path = probe.directory / 'decision.json'
            row = {'case': case['id'], 'arm': arm, 'run': run, 'original_complete': path.exists(), 'attempts': []}
            if path.exists():
                probe.read_result()
                files.append(path)
            elif arm == 'candidate_factorized':
                attempts = sorted((probe.directory / 'attempts/decision').glob('*/*.json'),
                                  key=lambda item: (item.stat().st_mtime_ns, str(item)))
                row['attempts'] = [{'path': str(item), 'sha256': study.indexed.file_hash(item),
                                    'mtime_ns': item.stat().st_mtime_ns} for item in attempts]
                record = first_recoverable(probe, row['attempts'])
                if record is not None:
                    records[(arm, case['id'])] = record
            rows.append(row)
    manifest = {'source': str(source), 'run': run, 'implementation': dependencies(), 'tasks': rows,
                'source_files': {str(path): study.indexed.file_hash(path) for path in files},
                'scope': 'Offline metadata-compatible recovery; original strict-format failures remain recorded. No new model calls.'}
    for (arm, name), record in records.items():
        atomic_json(root / arm / name / run / 'recovered.json', record)
    atomic_json(root / 'manifest.json', manifest)
    report(root)


def report(root):
    manifest = gate.read(root / 'manifest.json')
    if manifest['implementation'] != dependencies():
        raise ValueError('Recovery implementation changed')
    for name, expected in manifest['source_files'].items():
        if study.indexed.file_hash(Path(name)) != expected:
            raise ValueError(f'Frozen source file changed: {name}')
    source = Path(manifest['source'])
    _, cases = study.load(source)
    lookup = {case['id']: case for case in cases}
    rows = []
    for task in manifest['tasks']:
        case = lookup[task['case']]
        probe = study.DecisionProbe(source, case, task['run'], task['arm'])
        result = None
        recovered = False
        if task['original_complete']:
            result = probe.read_result()
        else:
            expected = first_recoverable(probe, task['attempts'])
            path = root / task['arm'] / task['case'] / task['run'] / 'recovered.json'
            if expected is not None:
                if not path.exists() or gate.read(path) != expected:
                    raise ValueError(f'Recovery result changed or missing: {path}')
                result, recovered = expected['result'], True
        rows.append({'case': task['case'], 'run': task['run'], 'arm': task['arm'],
                     'original_complete': task['original_complete'], 'recovered': recovered,
                     'complete': result is not None, 'status': result['status'] if result else None,
                     'expected_status': case['expected_status'], 'expected_kind': case['expected_kind'],
                     'classifications': sorted({check['classification'] for check in result.get('checks', [])}) if result else []})
    for arm in study.ARMS:
        group = [row for row in rows if row['arm'] == arm]
        print(f"{arm}/{manifest['run']}: original={sum(row['original_complete'] for row in group)} "
              f"recovered={sum(row['recovered'] for row in group)} complete={sum(row['complete'] for row in group)}/{len(group)}")
    print('offline=True model_calls=0; original strict-format failures preserved')
    return {'scope': manifest['scope'], 'results': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('recover', 'status', 'report'))
    parser.add_argument('--source-dir', type=Path, default=study.ROOT)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--run', choices=study.RUNS, default='R01')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'recover':
        recover(args.source_dir.resolve(), root, args.run)
    else:
        summary = report(root)
        if args.action == 'report':
            atomic_json(root / 'summary.json', summary)


if __name__ == '__main__':
    main()
