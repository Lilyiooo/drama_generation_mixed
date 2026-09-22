"""Recover candidate logic/R03 while preserving the frozen v2.4 evaluator."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from drama_evaluator import (
    EvaluationConfig,
    EvaluationPipeline,
    MultiAgentEvaluationConfig,
    MultiAgentEvaluationPipeline,
)


SCRIPT_ROOT = Path(os.environ.get(
    "SCRIPT_ROOT", "/inspire/hdd/global_user/wangqiqi-CZXS25210124/ScriptPipeline"
))
EXPERIMENT = SCRIPT_ROOT / "output/full_scene_gate_ab_v1"
RUN_ROOT = EXPERIMENT / "candidate_gate/R01"
INPUT = RUN_ROOT / "drama_evaluator_logic_quality_input/full_60_episodes.txt"
OUTPUT = RUN_ROOT / "drama_evaluations_logic_quality_qwen38_v24/full_60_episodes"
TARGET = OUTPUT / "multi_agent/reviews/logic/R03.json"
RAW_ROOT = OUTPUT / "multi_agent/recovery_raw/logic_R03"
RECORD = EXPERIMENT / "reports/logic_quality_R01_v24_logic_R03_structural_recovery.json"
BASE_PROTOCOL = EXPERIMENT / "reports/logic_quality_R01_v24_protocol.json"
MAX_OUTPUT_TOKENS = 24576


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def normalize_payload(payload: Any) -> tuple[dict[str, Any], str]:
    if isinstance(payload, dict):
        return payload, "none_object_received"
    if not isinstance(payload, list):
        raise ValueError("top-level JSON is neither an object nor an array")
    if len(payload) == 1 and isinstance(payload[0], dict) and isinstance(payload[0].get("ledger_entries"), list):
        return payload[0], "unwrap_single_object_array"
    wrappers = [item for item in payload if isinstance(item, dict) and isinstance(item.get("ledger_entries"), list)]
    if wrappers and len(wrappers) == len(payload):
        entries = [entry for wrapper in wrappers for entry in wrapper["ledger_entries"]]
        rationales = [str(wrapper.get("rationale", "")).strip() for wrapper in wrappers]
        return {
            "dimension": "logic",
            "reviewer_id": "R03",
            "rationale": "\n".join(item for item in rationales if item),
            "plot_behavior_ratings": [],
            "ledger_entries": entries,
        }, "merge_object_array"
    if all(isinstance(item, dict) and "unit" in item and "kind" in item for item in payload):
        return {
            "dimension": "logic",
            "reviewer_id": "R03",
            "rationale": "模型返回了完整台账条目数组；恢复器仅补齐协议要求的顶层对象。",
            "plot_behavior_ratings": [],
            "ledger_entries": payload,
        }, "wrap_ledger_entry_array"
    raise ValueError("top-level array does not contain recognizable review objects or ledger entries")


def build_pipeline() -> MultiAgentEvaluationPipeline:
    config = MultiAgentEvaluationConfig(
        script_path=INPUT,
        output_dir=OUTPUT,
        model="Qwen3.8-27B",
        review_workers=2,
        audit_workers=2,
        arbitration_workers=2,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        timeout=1200,
        retries=3,
        parse_retries=2,
        context_mode="direct",
        direct_char_limit=300000,
        chunk_chars=100000,
    )
    return MultiAgentEvaluationPipeline(config)


def evaluation_source(pipeline: MultiAgentEvaluationPipeline, script: str) -> tuple[str, str]:
    source_pipeline = EvaluationPipeline(EvaluationConfig(
        script_path=INPUT,
        output_dir=pipeline.artifact_dir,
        model="Qwen3.8-27B",
        workers=2,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        timeout=1200,
        retries=3,
        context_mode="direct",
        direct_char_limit=300000,
        chunk_chars=100000,
    ))
    source_pipeline.client = pipeline.client
    return source_pipeline._evaluation_source(script)


def validate_existing(pipeline: MultiAgentEvaluationPipeline) -> bool:
    if not TARGET.exists():
        return False
    pipeline._validate_review(pipeline._read_json(TARGET), "logic", "R03")
    print(f"reuse={TARGET} sha256={checksum(TARGET)}")
    return True


def recover() -> None:
    if not BASE_PROTOCOL.is_file():
        raise ValueError("Frozen v2.4 protocol is missing")
    pipeline = build_pipeline()
    if validate_existing(pipeline):
        return
    script = pipeline._read_text(INPUT)
    pipeline._prepare_output(pipeline._manifest(script))
    pipeline._create_client()
    source_name, source = evaluation_source(pipeline, script)
    system, user = pipeline._review_prompt("logic", "R03", source_name, source)
    suffix = (
        "\n\n恢复格式要求：顶层必须是一个 JSON 对象，必须以 { 开始并以 } 结束；"
        "不要只返回 ledger_entries 数组。如果内容过长，可以减少冗余解释，但不能省略已识别的独立问题与亮点。"
    )
    last_error: Exception | None = None
    for attempt in range(1, 4):
        raw = pipeline.client.text(system, user + suffix, label=f"剧本逻辑评审 R03 专项恢复#{attempt}")
        raw_path = RAW_ROOT / f"attempt_{attempt}.txt"
        write_text(raw_path, raw)
        try:
            parsed = pipeline._extract_json(raw)
            normalized, normalization = normalize_payload(parsed)
            report = pipeline._validate_review(normalized, "logic", "R03")
            pipeline._write_json(TARGET, report)
            record = {
                "recovery_id": "logic-quality-r01-v24-candidate-logic-r03-structural-recovery",
                "base_protocol_sha256": checksum(BASE_PROTOCOL),
                "target": str(TARGET),
                "reason": "valid JSON used an array at the top level instead of the required review object",
                "semantic_policy": "preserve model ledger entries; only normalize the top-level JSON container",
                "normalization": normalization,
                "attempt": attempt,
                "raw_response": str(raw_path),
                "raw_response_sha256": checksum(raw_path),
                "target_sha256": checksum(TARGET),
                "max_output_tokens": MAX_OUTPUT_TOKENS,
            }
            write_text(RECORD, json.dumps(record, ensure_ascii=False, indent=2))
            print(f"recovered={TARGET} entries={len(report['ledger_entries'])} normalization={normalization}")
            print(f"record={RECORD}")
            return
        except (ValueError, json.JSONDecodeError) as error:
            last_error = error
            print(f"attempt={attempt} structural_error={error}", flush=True)
    raise RuntimeError(f"logic/R03 structural recovery failed: {last_error}")


def status() -> None:
    pipeline = build_pipeline()
    exists = validate_existing(pipeline)
    print(f"target_exists={exists} record_exists={RECORD.exists()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "status"), nargs="?", default="run")
    args = parser.parse_args()
    if args.action == "run":
        recover()
    else:
        status()


if __name__ == "__main__":
    main()
