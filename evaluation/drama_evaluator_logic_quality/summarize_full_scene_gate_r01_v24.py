"""Freeze, validate and summarize R01 evaluations under evaluator v2.4."""

import argparse
import hashlib
import json
import os
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent
SCRIPT_ROOT = Path(os.environ.get(
    'SCRIPT_ROOT', '/inspire/hdd/global_user/wangqiqi-CZXS25210124/ScriptPipeline'
))
EXPERIMENT = SCRIPT_ROOT / 'output/full_scene_gate_ab_v1'
ARMS = ('control_hybrid', 'candidate_gate')
LABELS = ('剧本逻辑总分', '剧本质量最终总分')
PROMPT_VERSION = 'logic-quality-ledger-v2.4-shared-module-baseline-20260921'
EVALUATION_MODE = 'logic_quality_shared_module_baseline_fusion_multi_agent_6_2_2_2'
PROTOCOL_PATH = EXPERIMENT / 'reports/logic_quality_R01_v24_protocol.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation():
    paths = [EVAL_ROOT / 'evaluate_multi_agent.py', EVAL_ROOT / 'drama_evaluator/llm.py',
             EVAL_ROOT / 'drama_evaluator/multi_agent.py', EVAL_ROOT / 'drama_evaluator/pipeline.py']
    paths += sorted((EVAL_ROOT / 'prompts').rglob('*.md'))
    return {str(path.relative_to(EVAL_ROOT)): checksum(path) for path in paths}


def prepare_protocol():
    protocol = {
        'prompt_version': PROMPT_VERSION, 'evaluation_mode': EVALUATION_MODE,
        'model': 'Qwen3.8-27B', 'temperature': 0, 'enable_thinking': False,
        'context_mode': 'direct', 'direct_char_limit': 300000, 'chunk_chars': 100000,
        'review_workers': 2, 'audit_workers': 2, 'arbitration_workers': 2,
        'max_output_tokens': 16000, 'timeout': 1200, 'retries': 3, 'parse_retries': 2,
        'implementation_sha256': implementation(),
    }
    if PROTOCOL_PATH.exists() and read(PROTOCOL_PATH) != protocol:
        raise ValueError('v2.4 evaluator code, prompts or protocol changed; use a new evaluation tag')
    PROTOCOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return protocol


def verify_protocol():
    if not PROTOCOL_PATH.exists():
        raise ValueError('Run prepare before evaluation')
    expected = read(PROTOCOL_PATH)
    if expected.get('implementation_sha256') != implementation():
        raise ValueError('v2.4 evaluator code or prompts changed after preparation')
    return expected


def inspect(arm):
    directory = EXPERIMENT / arm / 'R01'
    script = directory / 'drama_evaluator_logic_quality_input/full_60_episodes.txt'
    source = script.with_suffix('.source.json')
    output = directory / 'drama_evaluations_logic_quality_qwen38_v24/full_60_episodes'
    row = {'arm': arm, 'complete': False, 'input': str(script), 'output': str(output)}
    if not script.exists() or not source.exists():
        row['status'] = 'input_missing'
        return row
    provenance = read(source)
    script_hash = checksum(script)
    if provenance.get('script_sha256') != script_hash or len(provenance.get('episodes', [])) != 60:
        raise ValueError(f'{arm}: exported script provenance mismatch')
    if [item.get('episode') for item in provenance['episodes']] != list(range(1, 61)):
        raise ValueError(f'{arm}: episode order changed')
    row.update(script_sha256=script_hash, characters=provenance.get('characters'))
    row['artifacts'] = {name: len(list((output / 'multi_agent' / name).rglob('*.json')))
                        for name in ('reviews', 'audits', 'arbitrations', 'holistic_scores')}
    manifest_path = output / 'multi_agent/manifest.json'
    scores_path = output / 'scores.json'
    if not manifest_path.exists():
        row['status'] = 'not_started'
        return row
    manifest = read(manifest_path)
    expected = {'version': PROMPT_VERSION, 'script_sha256': script_hash, 'model': 'Qwen3.8-27B',
                'context_mode': 'direct', 'direct_char_limit': 300000, 'chunk_chars': 100000}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(f'{arm}: evaluator manifest differs from frozen v2.4 protocol')
    if not scores_path.exists():
        row['status'] = 'running_or_incomplete'
        return row
    scores = read(scores_path)
    if (scores.get('model') != 'Qwen3.8-27B' or scores.get('evaluation_mode') != EVALUATION_MODE
            or set(scores.get('scores', {})) != set(LABELS)):
        raise ValueError(f'{arm}: score dimensions, model or mode mismatch')
    details = {label: scores['scores'][label] for label in LABELS}
    values = {label: details[label].get('final_score') for label in LABELS}
    if any(not isinstance(value, (int, float)) or not 0 <= value <= 100 for value in values.values()):
        raise ValueError(f'{arm}: invalid final scores')
    row.update(complete=True, status='complete', scores=values, score_details=details,
               evaluation_mode=scores['evaluation_mode'], prompt_version=PROMPT_VERSION)
    return row


def paired_difference(control, candidate):
    return {
        'final_scores': {label: round(candidate['scores'][label] - control['scores'][label], 4)
                         for label in LABELS},
        'components': {
            label: {
                'ledger_subdimension_mean': round(
                    candidate['score_details'][label]['ledger_subdimension_mean']
                    - control['score_details'][label]['ledger_subdimension_mean'], 4),
                'holistic_subdimension_mean': round(
                    candidate['score_details'][label]['holistic_subdimension_mean']
                    - control['score_details'][label]['holistic_subdimension_mean'], 4),
                'subdimensions': {
                    key: round(candidate['score_details'][label]['subdimensions'][key]['fused_score']
                               - control['score_details'][label]['subdimensions'][key]['fused_score'], 4)
                    for key in control['score_details'][label]['subdimensions']
                },
            } for label in LABELS
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-protocol', action='store_true')
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    protocol = prepare_protocol() if args.prepare_protocol else verify_protocol()
    rows = [inspect(arm) for arm in ARMS]
    for row in rows:
        suffix = ' '.join(f'{label}={row.get("scores", {}).get(label)}' for label in LABELS if label in row.get('scores', {}))
        print(f"{row['arm']}: status={row['status']} artifacts={row.get('artifacts', {})} {suffix}".rstrip())
    paired = paired_difference(*rows) if all(row['complete'] for row in rows) else None
    if paired:
        print('candidate_minus_control: ' + json.dumps(paired['final_scores'], ensure_ascii=False))
    report = {'experiment': 'full_scene_gate_ab_v1/R01', 'evaluator': 'drama_evaluator_logic_quality',
              'protocol': protocol, 'rows': rows, 'candidate_minus_control': paired}
    if args.write:
        if paired is None:
            raise SystemExit('Both v2.4 evaluations must finish before writing the report')
        path = EXPERIMENT / 'reports/logic_quality_R01_v24.json'
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f'report={path}')


if __name__ == '__main__':
    main()
