#!/usr/bin/env python3
"""多 stage 串联流水线（跳过已生成的 stage）：从已有 v4 输出读取 final_event_line，替换 outline 后继续下游 stage。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V5_SCRIPT = PROJECT_ROOT / "tools/stage_eventline_local_tree_v5.py"
DEFAULT_OUTLINE = PROJECT_ROOT / "output/60ep_0810_1946/03_story_outline/outline.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output/stage_eventline_pipeline_v5_max4"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def replace_stage_events(outline: dict[str, Any], stage_name: str, new_events: list[str]) -> dict[str, Any]:
    framework = outline.get("framework", [])
    for stage in framework:
        if not isinstance(stage, dict):
            continue
        if str(stage.get("stage", "")).strip() == stage_name:
            stage["event_list"] = [{"event": e} for e in new_events]
            return outline
    raise ValueError(f"未找到 stage：{stage_name}")


def run_v5_for_stage(
    *,
    outline_path: Path,
    stage: str,
    output_dir: Path,
    branch_factor: int,
    keep_top_k: int,
    local_max_depth: int,
    max_events_to_anchor: int,
    max_local_chains_per_anchor: int,
    parallel_workers: int,
    model_name: str,
    temperature: float,
    max_new_tokens: int,
) -> tuple[list[str], Path]:
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    safe_stage = stage.replace("/", "_").replace(" ", "_")
    v5_output = output_dir / "v5_results" / f"{safe_stage}_{timestamp}.json"
    v5_output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(V5_SCRIPT),
        "--outline-path", str(outline_path),
        "--stage", stage,
        "--branch-factor", str(branch_factor),
        "--keep-top-k", str(keep_top_k),
        "--local-max-depth", str(local_max_depth),
        "--max-events-to-anchor", str(max_events_to_anchor),
        "--max-local-chains-per-anchor", str(max_local_chains_per_anchor),
        "--parallel-workers", str(parallel_workers),
        "--model-name", model_name,
        "--temperature", str(temperature),
        "--max-new-tokens", str(max_new_tokens),
        "--output", str(v5_output),
    ]

    print(f"\n{'='*60}")
    print(f"▶ 正在处理 stage: {stage}")
    print(f"▶ 输入 outline: {outline_path}")
    print(f"▶ 输出结果: {v5_output}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, check=True, text=True)
    # stdout/stderr 直接流到父进程终端，实时显示进度

    v5_data = load_json(v5_output)
    final_event_line = v5_data.get("final_event_line", [])
    if not final_event_line:
        raise RuntimeError(f"v5 没有生成 final_event_line，请检查 {v5_output}")

    print(f"✅ {stage} 完成: 生成 {len(final_event_line)} 个事件")
    return final_event_line, v5_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="多 stage 串联流水线：跳过第一个 stage 的 v5 运行，从已有结果读入，再处理后续 stage"
    )
    parser.add_argument("--outline-path", default=str(DEFAULT_OUTLINE))
    parser.add_argument(
        "--skip-stage-result",
        required=True,
        help="已有 v5 运行结果 JSON 路径（跳过对应 stage 的生成，直接从中读取 final_event_line）",
    )
    parser.add_argument(
        "--skip-stage-name",
        required=True,
        help="跳过 stage 的名称（如 发展），与 --skip-stage-result 配合使用",
    )
    parser.add_argument(
        "--downstream-stages",
        default="高潮",
        help="跳过之后要处理的后续 stage，逗号分隔，默认 `高潮`",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--branch-factor", type=int, default=4)
    parser.add_argument("--keep-top-k", type=int, default=2)
    parser.add_argument("--local-max-depth", type=int, default=4)
    parser.add_argument("--max-events-to-anchor", type=int, default=9)
    parser.add_argument("--max-local-chains-per-anchor", type=int, default=4)
    parser.add_argument("--parallel-workers", type=int, default=8)
    parser.add_argument("--model-name", default="gemini-2.5-pro")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    args = parser.parse_args()

    downstream = [s.strip() for s in args.downstream_stages.split(",") if s.strip()]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")

    # 加载已有 v5 结果
    skip_path = Path(args.skip_stage_result)
    v5_data = load_json(skip_path)
    skip_events = v5_data.get("final_event_line", [])
    if not skip_events:
        print(f"错误：{skip_path} 中没有 final_event_line", file=sys.stderr)
        sys.exit(1)
    print(f"📥 从 {skip_path.name} 读取到 {len(skip_events)} 个事件（stage={args.skip_stage_name}）")

    # 加载原始 outline 并替换
    outline = load_json(Path(args.outline_path))
    save_json(output_dir / f"outline_original_{timestamp}.json", outline)

    outline = replace_stage_events(outline, args.skip_stage_name, skip_events)
    after_skip = output_dir / f"outline_after_{args.skip_stage_name}_{timestamp}.json"
    save_json(after_skip, outline)
    print(f"📄 已替换 {args.skip_stage_name} → {after_skip}")

    # 逐 downstream stage 跑 v5
    for idx, stage in enumerate(downstream, 1):
        print(f"\n{'#'*60}")
        print(f"## 后续阶段 {idx}/{len(downstream)}: {stage}")
        print(f"{'#'*60}")

        stage_input = output_dir / f"outline_before_{stage}_{timestamp}.json"
        save_json(stage_input, outline)

        final_event_line, _ = run_v5_for_stage(
            outline_path=stage_input,
            stage=stage,
            output_dir=output_dir,
            branch_factor=args.branch_factor,
            keep_top_k=args.keep_top_k,
            local_max_depth=args.local_max_depth,
            max_events_to_anchor=args.max_events_to_anchor,
            max_local_chains_per_anchor=args.max_local_chains_per_anchor,
            parallel_workers=args.parallel_workers,
            model_name=args.model_name,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )

        outline = replace_stage_events(outline, stage, final_event_line)
        after_path = output_dir / f"outline_after_{stage}_{timestamp}.json"
        save_json(after_path, outline)
        print(f"📄 已保存: {after_path}")

    # 最终
    final_path = output_dir / f"outline_final_{timestamp}.json"
    save_json(final_path, outline)
    print(f"\n🎉 完成！最终 outline: {final_path}")


if __name__ == "__main__":
    main()
