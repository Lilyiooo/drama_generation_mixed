"""Validate and summarize R01-R03 under the v2.4 logic-quality evaluator."""

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path


EVAL_ROOT = Path(__file__).resolve().parent
SCRIPT_ROOT = Path(os.environ.get(
    "SCRIPT_ROOT", "/inspire/hdd/global_user/wangqiqi-CZXS25210124/ScriptPipeline"
))
EXPERIMENT = SCRIPT_ROOT / "output/full_scene_gate_ab_v1"
ARMS = ("control_hybrid", "candidate_gate")
RUNS = ("R01", "R02", "R03")
NEW_RUNS = ("R02", "R03")
LABELS = ("剧本逻辑总分", "剧本质量最终总分")
PROMPT_VERSION = "logic-quality-ledger-v2.4-shared-module-baseline-20260921"
EVALUATION_MODE = "logic_quality_shared_module_baseline_fusion_multi_agent_6_2_2_2"
R01_PROTOCOL = EXPERIMENT / "reports/logic_quality_R01_v24_protocol.json"
PROTOCOL_PATH = EXPERIMENT / "reports/logic_quality_R02_R03_v24_protocol.json"
REPORT_PATH = EXPERIMENT / "reports/logic_quality_R01_R02_R03_v24.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation() -> dict[str, str]:
    paths = [
        EVAL_ROOT / "evaluate_multi_agent.py",
        EVAL_ROOT / "drama_evaluator/llm.py",
        EVAL_ROOT / "drama_evaluator/multi_agent.py",
        EVAL_ROOT / "drama_evaluator/pipeline.py",
    ]
    paths += sorted((EVAL_ROOT / "prompts").rglob("*.md"))
    return {str(path.relative_to(EVAL_ROOT)): checksum(path) for path in paths}


def prepare_protocol() -> dict:
    if not R01_PROTOCOL.is_file():
        raise ValueError("R01 v2.4 protocol is missing")
    r01 = read(R01_PROTOCOL)
    if r01.get("implementation_sha256") != implementation():
        raise ValueError("Evaluator code or prompts differ from the frozen R01 v2.4 protocol")
    protocol = {
        "prompt_version": PROMPT_VERSION,
        "evaluation_mode": EVALUATION_MODE,
        "runs": list(NEW_RUNS),
        "model": "Qwen3.8-27B",
        "temperature": 0,
        "enable_thinking": False,
        "context_mode": "direct",
        "direct_char_limit": 300000,
        "chunk_chars": 100000,
        "review_workers": 2,
        "audit_workers": 2,
        "arbitration_workers": 2,
        "outer_workers": 2,
        "max_output_tokens": 24576,
        "timeout": 1200,
        "retries": 3,
        "parse_retries": 3,
        "r01_protocol_path": str(R01_PROTOCOL),
        "r01_protocol_sha256": checksum(R01_PROTOCOL),
        "implementation_sha256": implementation(),
    }
    if PROTOCOL_PATH.exists() and read(PROTOCOL_PATH) != protocol:
        raise ValueError("R02/R03 v2.4 protocol or evaluator implementation changed")
    PROTOCOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return protocol


def verify_protocol() -> dict:
    if not PROTOCOL_PATH.is_file():
        raise ValueError("Run prepare before R02/R03 evaluation")
    protocol = read(PROTOCOL_PATH)
    if protocol.get("implementation_sha256") != implementation():
        raise ValueError("Evaluator code or prompts changed after R02/R03 preparation")
    if protocol.get("r01_protocol_sha256") != checksum(R01_PROTOCOL):
        raise ValueError("Frozen R01 protocol changed")
    return protocol


def inspect(arm: str, run: str) -> dict:
    directory = EXPERIMENT / arm / run
    script = directory / "drama_evaluator_logic_quality_input/full_60_episodes.txt"
    source = script.with_suffix(".source.json")
    output = directory / "drama_evaluations_logic_quality_qwen38_v24/full_60_episodes"
    row = {"arm": arm, "run": run, "complete": False, "input": str(script), "output": str(output)}
    if not script.is_file() or not source.is_file():
        row["status"] = "input_missing"
        return row
    provenance = read(source)
    script_hash = checksum(script)
    if provenance.get("script_sha256") != script_hash or len(provenance.get("episodes", [])) != 60:
        raise ValueError(f"{arm}/{run}: exported script provenance mismatch")
    if [item.get("episode") for item in provenance["episodes"]] != list(range(1, 61)):
        raise ValueError(f"{arm}/{run}: episode order changed")
    row["script_sha256"] = script_hash
    row["artifacts"] = {
        name: len(list((output / "multi_agent" / name).rglob("*.json")))
        for name in ("reviews", "audits", "arbitrations", "holistic_scores")
    }
    manifest_path = output / "multi_agent/manifest.json"
    scores_path = output / "scores.json"
    if not manifest_path.is_file():
        row["status"] = "not_started"
        return row
    manifest = read(manifest_path)
    expected = {
        "version": PROMPT_VERSION,
        "script_sha256": script_hash,
        "model": "Qwen3.8-27B",
        "context_mode": "direct",
        "direct_char_limit": 300000,
        "chunk_chars": 100000,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{arm}/{run}: evaluator manifest differs from v2.4 protocol")
    if not scores_path.is_file():
        row["status"] = "running_or_incomplete"
        return row
    scores = read(scores_path)
    if scores.get("model") != "Qwen3.8-27B" or scores.get("evaluation_mode") != EVALUATION_MODE:
        raise ValueError(f"{arm}/{run}: score model or mode mismatch")
    values = {label: scores["scores"][label]["final_score"] for label in LABELS}
    if any(not isinstance(value, (int, float)) or not 0 <= value <= 100 for value in values.values()):
        raise ValueError(f"{arm}/{run}: invalid score")
    row.update(complete=True, status="complete", scores=values)
    return row


def aggregate(rows: list[dict]) -> dict | None:
    if not all(row["complete"] for row in rows):
        return None
    pairs = []
    for run in RUNS:
        control = next(row for row in rows if row["arm"] == "control_hybrid" and row["run"] == run)
        candidate = next(row for row in rows if row["arm"] == "candidate_gate" and row["run"] == run)
        pairs.append({
            "run": run,
            "control": control["scores"],
            "candidate": candidate["scores"],
            "candidate_minus_control": {
                label: round(candidate["scores"][label] - control["scores"][label], 4)
                for label in LABELS
            },
        })
    return {
        "paired_runs": pairs,
        "candidate_minus_control_summary": {
            label: {
                "mean": round(statistics.mean(pair["candidate_minus_control"][label] for pair in pairs), 4),
                "sample_std": round(statistics.stdev(pair["candidate_minus_control"][label] for pair in pairs), 4),
                "values": [pair["candidate_minus_control"][label] for pair in pairs],
            }
            for label in LABELS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-protocol", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    protocol = prepare_protocol() if args.prepare_protocol else verify_protocol()
    rows = [inspect(arm, run) for run in RUNS for arm in ARMS]
    for row in rows:
        scores = " ".join(f"{label}={row.get('scores', {}).get(label)}" for label in LABELS if label in row.get("scores", {}))
        print(f"{row['arm']}/{row['run']}: status={row['status']} artifacts={row.get('artifacts', {})} {scores}".rstrip())
    result = aggregate(rows)
    if result:
        print("paired_summary=" + json.dumps(result["candidate_minus_control_summary"], ensure_ascii=False))
    if args.write:
        if result is None:
            raise SystemExit("All six R01-R03 trajectories must be evaluated before writing the report")
        payload = {"experiment": "full_scene_gate_ab_v1", "protocol_r02_r03": protocol, "rows": rows, **result}
        REPORT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()
