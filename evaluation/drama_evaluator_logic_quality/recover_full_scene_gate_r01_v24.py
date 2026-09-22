"""Safely resume the single truncated v2.4 candidate review."""

import argparse
import hashlib
import json
from pathlib import Path

import summarize_full_scene_gate_r01_v24 as summary


EXPERIMENT = summary.EXPERIMENT
OUTPUT = (
    EXPERIMENT
    / "candidate_gate/R01/drama_evaluations_logic_quality_qwen38_v24/full_60_episodes"
)
ARTIFACTS = OUTPUT / "multi_agent"
RECOVERY_PROTOCOL = EXPERIMENT / "reports/logic_quality_R01_v24_recovery.json"
RECOVERY_RESULT = EXPERIMENT / "reports/logic_quality_R01_v24_recovery_result.json"
BASE_PROTOCOL = summary.PROTOCOL_PATH
PROTECTED_REVIEWS = (
    "reviews/logic/R01.json",
    "reviews/logic/R02.json",
    "reviews/quality/R01.json",
    "reviews/quality/R02.json",
    "reviews/quality/R03.json",
)
TARGET_REVIEW = "reviews/logic/R03.json"


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def current_protected_hashes() -> dict[str, str]:
    missing = [relative for relative in PROTECTED_REVIEWS if not (ARTIFACTS / relative).is_file()]
    if missing:
        raise ValueError(f"Expected completed reviews are missing: {missing}")
    return {relative: checksum(ARTIFACTS / relative) for relative in PROTECTED_REVIEWS}


def expected_protocol() -> dict:
    return {
        "recovery_id": "logic-quality-r01-v24-candidate-logic-r03-output-limit",
        "base_protocol_path": str(BASE_PROTOCOL),
        "base_protocol_sha256": checksum(BASE_PROTOCOL),
        "arm": "candidate_gate",
        "target_artifact": TARGET_REVIEW,
        "reason": "logic R03 exhausted three retries because the response was truncated at 16000 tokens",
        "only_override": {"max_output_tokens": {"from": 16000, "to": 24576}},
        "protected_review_sha256": current_protected_hashes(),
    }


def prepare() -> None:
    base = summary.verify_protocol()
    if base.get("max_output_tokens") != 16000:
        raise ValueError("Frozen v2.4 base protocol no longer has max_output_tokens=16000")
    protocol = expected_protocol()
    if RECOVERY_PROTOCOL.exists() and read(RECOVERY_PROTOCOL) != protocol:
        raise ValueError("Recovery protocol or protected review files changed")
    write(RECOVERY_PROTOCOL, protocol)
    target_state = "already_present" if (ARTIFACTS / TARGET_REVIEW).exists() else "missing"
    print(f"recovery_protocol={RECOVERY_PROTOCOL}")
    print(f"target={TARGET_REVIEW} state={target_state} protected_reviews={len(PROTECTED_REVIEWS)}")


def verify_protected() -> dict:
    if not RECOVERY_PROTOCOL.exists():
        raise ValueError("Run prepare before recovery")
    protocol = read(RECOVERY_PROTOCOL)
    if protocol.get("base_protocol_sha256") != checksum(BASE_PROTOCOL):
        raise ValueError("Frozen base protocol changed after recovery preparation")
    current = current_protected_hashes()
    if current != protocol.get("protected_review_sha256"):
        raise ValueError("One or more protected pre-recovery reviews changed")
    return protocol


def status(require_complete: bool = False) -> None:
    summary.verify_protocol()
    verify_protected()
    row = summary.inspect("candidate_gate")
    target_exists = (ARTIFACTS / TARGET_REVIEW).is_file()
    print(
        f"candidate_gate: status={row['status']} target_review={target_exists} "
        f"artifacts={row.get('artifacts', {})}"
    )
    if require_complete and not row["complete"]:
        raise SystemExit("Candidate v2.4 evaluation is still incomplete")
    if not row["complete"]:
        return
    if not require_complete:
        print(f"target_review_sha256={checksum(ARTIFACTS / TARGET_REVIEW)}")
        return
    result = {
        "recovery_protocol": str(RECOVERY_PROTOCOL),
        "recovery_protocol_sha256": checksum(RECOVERY_PROTOCOL),
        "protected_reviews_unchanged": True,
        "target_review_sha256": checksum(ARTIFACTS / TARGET_REVIEW),
        "scores_sha256": checksum(OUTPUT / "scores.json"),
        "artifacts": row["artifacts"],
        "scores": row["scores"],
    }
    write(RECOVERY_RESULT, result)
    print(f"recovery_result={RECOVERY_RESULT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "status", "verify"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        status(require_complete=args.action == "verify")


if __name__ == "__main__":
    main()
