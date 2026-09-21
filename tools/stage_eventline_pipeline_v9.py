#!/usr/bin/env python3
"""多 stage 串联流水线：依次对 outline 中各 stage 执行 v9 树搜索，逐段替换 event_list。"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V9_SCRIPT = PROJECT_ROOT / "tools/stage_eventline_local_tree_v9.py"
DEFAULT_OUTLINE = PROJECT_ROOT / "output/60ep_0810_1946/03_story_outline/outline.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output/stage_eventline_pipeline_v9"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def replace_stage_events(outline: dict[str, Any], stage_name: str, new_events: list[str]) -> dict[str, Any]:
    """把 outline 中 stage_name 对应的 event_list 替换为 new_events（纯字符串列表）。"""
    framework = outline.get("framework", [])
    for stage in framework:
        if not isinstance(stage, dict):
            continue
        if str(stage.get("stage", "")).strip() == stage_name:
            stage["event_list"] = [{"event": e} for e in new_events]
            return outline
    raise ValueError(f"未找到 stage：{stage_name}")


def parse_stage_from_name(filename: str, prefix: str) -> str:
    """从文件名提取 stage 名。如 outline_after_发展_0812_211557.json -> 发展。"""
    stem = filename[:-5] if filename.endswith(".json") else filename
    if prefix and stem.startswith(prefix):
        stem = stem[len(prefix):]
    m = re.match(r"(.+)_(\d{4}_\d{6})$", stem)
    if m:
        return m.group(1)
    return stem


def drop_completed_stages(stages: list[str], done_stage: str) -> tuple[list[str], list[str]]:
    """从 stages 里移除 done_stage 及它之前的所有 stage，返回 (剩余 stages, 被跳过 stages)。"""
    if done_stage not in stages:
        return stages, []
    idx = stages.index(done_stage)
    skipped = stages[: idx + 1]
    remaining = stages[idx + 1:]
    if skipped:
        print(f"   跳过 stage: {', '.join(skipped)}")
    return remaining, skipped


def run_v9_for_stage(
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
    mutation_probability: float,
    initial_trope_bank: list[dict[str, str]] | None = None,
) -> tuple[list[str], Path, list[dict[str, str]]]:
    """运行 v9 为指定 stage 生成 final_event_line，返回 (事件列表, 输出路径, trope_bank)。"""
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    safe_stage = stage.replace("/", "_").replace(" ", "_")
    v9_output = output_dir / "v9_results" / f"{safe_stage}_{timestamp}.json"
    v9_output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(V9_SCRIPT),
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
        "--mutation-probability", str(mutation_probability),
        "--output", str(v9_output),
    ]
    if initial_trope_bank:
        bank_path = v9_output.parent / f"{safe_stage}_{timestamp}_bank.json"
        save_json(bank_path, {"trope_bank": initial_trope_bank})
        cmd += ["--initial-trope-bank", str(bank_path)]
        print(f"▶ 复用前序 bank（{len(initial_trope_bank)} 条）: {bank_path}")

    print(f"\n{'='*60}")
    print(f"▶ 正在处理 stage: {stage}")
    print(f"▶ 输入 outline: {outline_path}")
    print(f"▶ 输出结果: {v9_output}")
    print(f"=" * 60)

    result = subprocess.run(cmd, check=True, text=True)
    # stdout/stderr 直接流到父进程终端，实时显示进度

    # 读取结果
    v9_data = load_json(v9_output)
    final_event_line = v9_data.get("final_event_line", [])
    trope_bank = v9_data.get("trope_bank", [])
    if not final_event_line:
        raise RuntimeError(f"v9 没有生成 final_event_line，请检查 {v9_output}")

    print(f"✅ {stage} 完成: 生成 {len(final_event_line)} 个事件，bank {len(trope_bank)} 条")
    return final_event_line, v9_output, trope_bank


def main() -> None:
    parser = argparse.ArgumentParser(
        description="多 stage 串联流水线：依次执行 v9 树搜索并逐段替换 outline"
    )
    parser.add_argument(
        "--outline-path",
        default=None,
        help="显式指定 outline JSON 路径；不传则自动检测 output_dir 下已有进度并续传",
    )
    parser.add_argument(
        "--stages",
        default="发展,高潮",
        help="要按顺序处理的 stage，逗号分隔，默认 `发展,高潮`",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="流水线输出目录")
    # v9 参数透传
    parser.add_argument("--branch-factor", type=int, default=4)
    parser.add_argument("--keep-top-k", type=int, default=2)
    parser.add_argument("--local-max-depth", type=int, default=4)
    parser.add_argument("--max-events-to-anchor", type=int, default=9)
    parser.add_argument("--max-local-chains-per-anchor", type=int, default=4)
    parser.add_argument("--parallel-workers", type=int, default=32)
    parser.add_argument("--model-name", default="gemini-2.5-pro")
    parser.add_argument("--mutation-probability", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    args = parser.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if not stages:
        print("错误：至少指定一个 stage", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")

    explicit_outline = args.outline_path is not None
    current_outline: dict[str, Any] | None = None

    if explicit_outline:
        # 用户显式指定 outline，不自动续传
        current_outline = load_json(Path(args.outline_path))
        print(f"📥 使用指定 outline: {args.outline_path}")
    else:
        # 自动续传：优先找 outline_after_<stage>_*.json，其次复用 v9_results/<stage>_*.json
        after_candidates = sorted(
            output_dir.glob("outline_after_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if after_candidates:
            latest = after_candidates[0]
            done_stage = parse_stage_from_name(latest.name, prefix="outline_after_")
            current_outline = load_json(latest)
            print(f"🔁 检测到已完成 outline: {latest.name}")
            stages, _ = drop_completed_stages(stages, done_stage)
        else:
            v9_candidates = sorted(
                output_dir.glob("v9_results/*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if v9_candidates:
                latest = v9_candidates[0]
                done_stage = parse_stage_from_name(latest.name, prefix="")
                v9_data = load_json(latest)
                fel = v9_data.get("final_event_line", [])
                if fel:
                    current_outline = load_json(DEFAULT_OUTLINE)
                    current_outline = replace_stage_events(current_outline, done_stage, fel)
                    print(f"🔁 复用已有 v9 结果: {latest.name}（{done_stage}，{len(fel)} 事件）")
                    stages, _ = drop_completed_stages(stages, done_stage)

        if current_outline is None:
            current_outline = load_json(DEFAULT_OUTLINE)
            print(f"📥 使用默认 outline: {DEFAULT_OUTLINE}")

    # 如果所有 stage 都已完成，直接保存最终 outline 退出
    if not stages:
        print("✅ 所有指定 stage 均已完成，无需继续生成。")
        final_path = output_dir / f"outline_final_{timestamp}.json"
        save_json(final_path, current_outline)
        print(f"   最终 outline: {final_path}")
        return

    # 保存本次运行的起点 outline 副本
    save_json(output_dir / f"outline_original_{timestamp}.json", current_outline)

    current_trope_bank: list[dict[str, str]] | None = None

    for idx, stage in enumerate(stages, 1):
        print(f"\n{'#'*60}")
        print(f"## 阶段 {idx}/{len(stages)}: {stage}")
        print(f"{'#'*60}")

        # 保存当前 outline 供 v9 读取
        stage_input = output_dir / f"outline_before_{stage}_{timestamp}.json"
        save_json(stage_input, current_outline)

        # 运行 v9（把前序 stage 已建好的 bank 传下去，避免重复建库）
        final_event_line, v9_output_path, trope_bank = run_v9_for_stage(
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
            mutation_probability=args.mutation_probability,
            initial_trope_bank=current_trope_bank,
        )

        # 更新 bank，供下一个 stage 复用
        current_trope_bank = trope_bank if trope_bank else current_trope_bank

        # 替换该 stage 的 event_list
        current_outline = replace_stage_events(current_outline, stage, final_event_line)

        # 保存中间状态
        after_path = output_dir / f"outline_after_{stage}_{timestamp}.json"
        save_json(after_path, current_outline)
        print(f"📄 已保存替换后的 outline: {after_path}")

    # 保存最终 outline
    final_path = output_dir / f"outline_final_{timestamp}.json"
    save_json(final_path, current_outline)

    print(f"\n{'='*60}")
    print(f"🎉 v9 流水线完成！")
    print(f"   最终 outline: {final_path}")
    print(f"   处理 stage: {', '.join(stages)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
