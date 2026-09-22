"""Validate and summarize the paired R01 logic/quality evaluations."""

import argparse
import hashlib
import json
import os
from pathlib import Path

SCRIPT_ROOT = Path(os.environ.get(
    'SCRIPT_ROOT', '/inspire/hdd/global_user/wangqiqi-CZXS25210124/ScriptPipeline'
))
EXPERIMENT = SCRIPT_ROOT / 'output/full_scene_gate_ab_v1'
ARMS = ('control_hybrid', 'candidate_gate')
LABELS = ('剧本逻辑总分', '剧本质量最终总分')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def inspect(arm):
    directory = EXPERIMENT / arm / 'R01'
    script = directory / 'drama_evaluator_logic_quality_input/full_60_episodes.txt'
    source = script.with_suffix('.source.json')
    output = directory / 'drama_evaluations_logic_quality_qwen38/full_60_episodes'
    row = {'arm': arm, 'complete': False, 'input': str(script), 'output': str(output)}
    if not script.exists() or not source.exists():
        row['status'] = 'input_missing'
        return row
    provenance = read(source)
    checksum = hashlib.sha256(script.read_bytes()).hexdigest()
    if provenance.get('script_sha256') != checksum or len(provenance.get('episodes', [])) != 60:
        raise ValueError(f'{arm}: exported script provenance mismatch')
    if [item.get('episode') for item in provenance['episodes']] != list(range(1, 61)):
        raise ValueError(f'{arm}: exported episode order changed')
    row.update(script_sha256=checksum, characters=provenance.get('characters'))
    manifest_path = output / 'multi_agent/manifest.json'
    scores_path = output / 'scores.json'
    row['artifacts'] = {
        name: len(list((output / 'multi_agent' / name).rglob('*.json')))
        for name in ('reviews', 'audits', 'arbitrations', 'holistic_scores')
    }
    if not manifest_path.exists():
        row['status'] = 'not_started'
        return row
    manifest = read(manifest_path)
    expected = {'script_sha256': checksum, 'model': 'Qwen3.8-27B', 'context_mode': 'direct',
                'direct_char_limit': 300000, 'chunk_chars': 100000}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(f'{arm}: evaluator manifest differs from frozen R01 protocol')
    row['prompt_version'] = manifest.get('version')
    if not scores_path.exists():
        row['status'] = 'running_or_incomplete'
        return row
    scores = read(scores_path)
    if scores.get('model') != 'Qwen3.8-27B' or set(scores.get('scores', {})) != set(LABELS):
        raise ValueError(f'{arm}: invalid score dimensions or model')
    values = {label: scores['scores'][label].get('final_score') for label in LABELS}
    if any(not isinstance(value, (int, float)) or not 0 <= value <= 100 for value in values.values()):
        raise ValueError(f'{arm}: invalid final scores')
    details = {
        label: {
            'final_score': scores['scores'][label]['final_score'],
            'ledger_subdimension_mean': scores['scores'][label]['ledger_subdimension_mean'],
            'holistic_subdimension_mean': scores['scores'][label]['holistic_subdimension_mean'],
            'subdimensions': scores['scores'][label]['subdimensions'],
        } for label in LABELS
    }
    row.update(complete=True, status='complete', scores=values, score_details=details,
               evaluation_mode=scores.get('evaluation_mode'))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    rows = [inspect(arm) for arm in ARMS]
    for row in rows:
        scores = row.get('scores', {})
        suffix = ' '.join(f'{label}={scores[label]}' for label in LABELS if label in scores)
        print(f"{row['arm']}: status={row['status']} artifacts={row.get('artifacts', {})} {suffix}".rstrip())
    paired = None
    if all(row['complete'] for row in rows):
        control, candidate = rows
        paired = {
            'final_scores': {
                label: round(candidate['scores'][label] - control['scores'][label], 4) for label in LABELS
            },
            'components': {
                label: {
                    'ledger_subdimension_mean': round(
                        candidate['score_details'][label]['ledger_subdimension_mean']
                        - control['score_details'][label]['ledger_subdimension_mean'], 4),
                    'holistic_subdimension_mean': round(
                        candidate['score_details'][label]['holistic_subdimension_mean']
                        - control['score_details'][label]['holistic_subdimension_mean'], 4),
                    'subdimensions': {
                        key: round(
                            candidate['score_details'][label]['subdimensions'][key]['fused_score']
                            - control['score_details'][label]['subdimensions'][key]['fused_score'], 4)
                        for key in control['score_details'][label]['subdimensions']
                    },
                } for label in LABELS
            },
        }
        print('candidate_minus_control: ' + json.dumps(paired['final_scores'], ensure_ascii=False))
    report = {'experiment': 'full_scene_gate_ab_v1/R01', 'evaluator': 'drama_evaluator_logic_quality',
              'rows': rows, 'candidate_minus_control': paired}
    if args.write:
        if paired is None:
            raise SystemExit('Both evaluations must be complete before writing the paired report')
        path = EXPERIMENT / 'reports/logic_quality_R01.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f'report={path}')


if __name__ == '__main__':
    main()
