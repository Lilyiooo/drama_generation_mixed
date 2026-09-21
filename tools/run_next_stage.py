#!/usr/bin/env python3
"""基于已有 pipeline 输出，继续生成下一个 stage（默认高潮），并复用已有的 trope_bank。

与 pipeline 的自动续传不同，本脚本会显式复用前序 stage 已建好的叙事套路 bank，
避免续传时丢掉 bank 导致 novelty 去重失效。

用法示例（接着跑高潮，max=4）：
    python run_next_stage.py \
        --output-dir /path/to/output/stage_eventline_pipeline_v8 \
        --stage 高潮 \
        --max-events-to-anchor 4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V8_SCRIPT = PROJECT_ROOT / "tools/stage_eventline_local_tree_v8.py"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output/stage_eventline_pipeline_v8"

STAGE_ORDER = ["开端", "发展", "高潮", "结局"]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def replace_stage_events(outline: dict[str, Any], stage_name: str, new_events: list[str]) -> dict[str, Any]:
    for stage in outline.get("framework", []):
        if isinstance(stage, dict) and str(stage.get("stage", "")).strip() == stage_name:
            stage["event_list"] = [{"event": e} for e in new_events]
            return outline
    raise ValueError(f"未找到 stage：{stage_name}")


def find_latest(output_dir: Path, pattern: str) -> Path | None:
    candidates = sorted(output_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def find_prev_stage(stage: str) -> str | None:
    if stage not in STAGE_ORDER:
        return None
    idx = STAGE_ORDER.index(stage)
    return STAGE_ORDER[idx - 1] if idx > 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description="基于已有 pipeline 输出继续生成下一个 stage")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="pipeline 输出目录")
    parser.add_argument("--stage", default="高潮", help="要生成的 stage 名称")
    parser.add_argument("--max-events-to-anchor", type=int, default=4)
    parser.add_argument("--branch-factor", type=int, default=4)
    parser.add_argument("--keep-top-k", type=int, default=2)
    parser.add_argument("--local-max-depth", type=int, default=4)
    parser.add_argument("--max-local-chains-per-anchor", type=int, default=4)
    parser.add_argument("--parallel-workers", type=int, default=8)
    parser.add_argument("--model-name", default="gemini-2.5-pro")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    stage = args.stage
    timestamp = datetime.now().strftime("%m%d_%H%M%S")

    # 1. 定位输入 outline：优先 outline_after_<前序stage>_*.json，其次 outline_before_<stage>_*.json
    outline_path = find_latest(output_dir, "outline_after_*.json")
    if outline_path is None:
        outline_path = find_latest(output_dir, f"outline_before_{stage}_*.json")
    if outline_path is None:
        print("错误：找不到输入 outline（outline_after_*.json 或 outline_before_*.json）", file=sys.stderr)
        sys.exit(1)
    outline = load_json(outline_path)
    print(f"📥 输入 outline: {outline_path}")

    # 2. 定位 bank：优先 <stage>_*_bank.json，其次从前序 stage 结果读 trope_bank
    bank_path: Path | None = find_latest(output_dir, f"v8_results/{stage}_*_bank.json")
    if bank_path is None:
        prev_stage = find_prev_stage(stage)
        if prev_stage:
            prev_result = find_latest(output_dir, f"v8_results/{prev_stage}_*.json")
            if prev_result is not None:
                data = load_json(prev_result)
                trope_bank = data.get("trope_bank", [])
                if trope_bank:
                    bank_path = output_dir / "v8_results" / f"{stage}_{timestamp}_bank.json"
                    save_json(bank_path, {"trope_bank": trope_bank})
    if bank_path is not None:
        print(f"📚 复用叙事套路 bank: {bank_path}")

    # 3. 运行 v8 树搜索生成该 stage
    result_path = output_dir / "v8_results" / f"{stage}_{timestamp}.json"
    cmd = [
        sys.executable,
        str(V8_SCRIPT),
        "--outline-path", str(outline_path),
        "--stage", stage,
        "--max-events-to-anchor", str(args.max_events_to_anchor),
        "--branch-factor", str(args.branch_factor),
        "--keep-top-k", str(args.keep_top_k),
        "--local-max-depth", str(args.local_max_depth),
        "--max-local-chains-per-anchor", str(args.max_local_chains_per_anchor),
        "--parallel-workers", str(args.parallel_workers),
        "--model-name", args.model_name,
        "--temperature", str(args.temperature),
        "--max-new-tokens", str(args.max_new_tokens),
        "--output", str(result_path),
    ]
    if bank_path is not None:
        cmd += ["--initial-trope-bank", str(bank_path)]

    print(f"\n▶ 正在处理 stage: {stage}（max_events_to_anchor={args.max_events_to_anchor}）")
    print(f"▶ 输出结果: {result_path}")
    subprocess.run(cmd, check=True, text=True)

    # 4. 替换 outline 中该 stage 的 event_list
    result = load_json(result_path)
    final_event_line = result.get("final_event_line", [])
    if not final_event_line:
        print("错误：v8 没有生成 final_event_line", file=sys.stderr)
        sys.exit(1)
    outline = replace_stage_events(outline, stage, final_event_line)

    # 5. 保存最终 outline
    final_path = output_dir / f"outline_final_{stage}_{timestamp}.json"
    save_json(final_path, outline)

    print(f"\n✅ 完成！{stage} 生成 {len(final_event_line)} 个事件")
    print(f"   最终 outline: {final_path}")
    print(f"   {stage} 树搜索结果: {result_path}")


if __name__ == "__main__":
    main()
