"""Evaluate one complete ScriptPipeline run with the vendored v2.4 evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVALUATOR = (
    REPO_ROOT / "evaluation/drama_evaluator_logic_quality/evaluate_multi_agent.py"
)


def export_complete_script(run_dir: Path, episode_count: int) -> tuple[Path, str]:
    drama_dir = run_dir / "05_drama"
    paths = sorted(drama_dir.glob("episode_*.json"))
    if len(paths) != episode_count:
        raise ValueError(
            f"Expected {episode_count} episode JSON files in {drama_dir}; found {len(paths)}"
        )

    episodes = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        episode_id = payload.get("episode_id")
        content = payload.get("content")
        if not isinstance(episode_id, int) or episode_id in episodes:
            raise ValueError(f"Invalid or duplicate episode_id in {path}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Empty episode content in {path}")
        episodes[episode_id] = content.strip()

    if sorted(episodes) != list(range(episode_count)):
        raise ValueError("Episode IDs must be continuous and zero-based")

    script = "\n\n".join(
        f"第{episode_id + 1}集\n\n{episodes[episode_id]}"
        for episode_id in range(episode_count)
    ) + "\n"
    export_dir = run_dir / "drama_evaluator_input"
    export_dir.mkdir(parents=True, exist_ok=True)
    script_path = export_dir / f"full_{episode_count}_episodes.txt"
    script_path.write_text(script, encoding="utf-8")
    digest = hashlib.sha256(script.encode("utf-8")).hexdigest()
    manifest = {
        "episode_count": episode_count,
        "script_path": str(script_path),
        "script_sha256": digest,
        "characters": len(script),
    }
    (export_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return script_path, digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--episode-count", type=int, default=60)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--execute-api", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    evaluator = args.evaluator.resolve()
    if args.episode_count < 1:
        parser.error("--episode-count must be positive")
    if not evaluator.is_file():
        parser.error(f"Evaluator not found: {evaluator}")

    script_path, digest = export_complete_script(run_dir, args.episode_count)
    result_dir = (
        run_dir
        / "drama_evaluations_logic_quality_qwen38_v24"
        / f"full_{args.episode_count}_episodes"
    )
    protocol = {
        "model": "Qwen3.8-27B",
        "base_url": os.environ.get("DRAMA_EVAL_BASE_URL", "http://127.0.0.1:8001/v1"),
        "enable_thinking": os.environ.get("DRAMA_EVAL_ENABLE_THINKING", "false"),
        "temperature": os.environ.get("DRAMA_EVAL_TEMPERATURE", "0"),
        "context_mode": "direct",
        "max_output_tokens": 24576,
        "script_sha256": digest,
    }
    protocol_path = result_dir / "local_protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError(f"Evaluation protocol changed; use a fresh run directory: {result_dir}")
    if args.execute_api:
        result_dir.mkdir(parents=True, exist_ok=True)
        protocol_path.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    command = [
        sys.executable, str(evaluator), str(script_path),
        "--output-dir", str(result_dir), "--model", "Qwen3.8-27B",
        "--context-mode", "direct", "--direct-char-limit", "300000",
        "--chunk-chars", "100000", "--max-output-tokens", "24576",
        "--review-workers", "2", "--audit-workers", "2", "--arbitration-workers", "2",
        "--timeout", "1200", "--retries", "3", "--parse-retries", "3",
    ]
    if not args.execute_api:
        command.append("--dry-run")
    environment = dict(os.environ)
    environment.setdefault("DRAMA_EVAL_BASE_URL", "http://127.0.0.1:8001/v1")
    environment.setdefault("DRAMA_EVAL_API_KEY", "EMPTY")
    print(f"script={script_path} sha256={digest} execute_api={args.execute_api}", flush=True)
    subprocess.run(command, cwd=evaluator.parent, env=environment, check=True)


if __name__ == "__main__":
    main()
