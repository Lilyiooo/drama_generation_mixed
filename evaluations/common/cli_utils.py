from __future__ import annotations

import argparse
from pathlib import Path

from .annotations import load_or_create_annotations
from .io import load_scenes


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run_dir", help="生成结果目录，例如 output/60ep_0810_1946")
    parser.add_argument("--source", choices=["auto", "drama", "outline"], default="auto")
    parser.add_argument("--annotator", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--output", help="结果 JSON 路径；默认写入 run_dir/evaluation_results")


def prepare(args: argparse.Namespace, metric_name: str):
    scenes, source = load_scenes(args.run_dir, args.source)
    if len(scenes) < 2:
        raise RuntimeError(f"{source} 只提取到 {len(scenes)} 个场景/集，至少需要2个样本")
    result_dir = Path(args.run_dir).resolve() / "evaluation_results"
    cache_path = result_dir / f"annotations_{source}_{args.annotator}.json"
    annotations = load_or_create_annotations(scenes, cache_path, args.annotator)
    output = Path(args.output) if args.output else result_dir / f"{metric_name}_{source}.json"
    return scenes, annotations, source, output
