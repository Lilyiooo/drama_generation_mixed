from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

METRICS = [
    "functional_equivalence_reuse",
    "narrative_function_sequence",
    "structural_vendi",
    "semantic_homogeneity",
]
OPTIONAL_METRICS = [
    "chunk_tag_homogeneity",
    "dual_layer_trope_homogeneity",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="依次运行剧内同质化评测")
    parser.add_argument("run_dir")
    parser.add_argument("--source", choices=["auto", "drama", "outline"], default="auto")
    parser.add_argument("--annotator", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--semantic-backend", choices=["auto", "sentence-transformers", "tfidf-char"], default="auto")
    parser.add_argument("--include-chunk-tag-homogeneity", action="store_true", help="额外运行基于项目LLM的分段标签归并同质化评测")
    parser.add_argument("--include-dual-layer-trope-homogeneity", action="store_true", help="额外运行双层套路化评测（中观动作标签 + 微观因果链）")
    parser.add_argument("--chunk-count", type=int, default=12)
    parser.add_argument("--chunk-model-name", default="gemini-2.5-pro")
    parser.add_argument("--max-tags-per-chunk", type=int, default=6)
    parser.add_argument("--global-tag-min", type=int, default=10)
    parser.add_argument("--global-tag-max", type=int, default=18)
    parser.add_argument("--max-macro-tags", type=int, default=6)
    parser.add_argument("--max-micro-motifs", type=int, default=8)
    parser.add_argument("--chunk-refresh", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    selected_optional_metrics = []
    if args.include_chunk_tag_homogeneity:
        selected_optional_metrics.append("chunk_tag_homogeneity")
    if args.include_dual_layer_trope_homogeneity:
        selected_optional_metrics.append("dual_layer_trope_homogeneity")
    metrics = METRICS + selected_optional_metrics
    for metric in metrics:
        command = [
            sys.executable,
            str(root / metric / "evaluate.py"),
            args.run_dir,
            "--source",
            args.source,
        ]
        if metric in METRICS:
            command.extend([
                "--annotator",
                args.annotator,
            ])
        if metric == "semantic_homogeneity":
            command.extend(["--backend", args.semantic_backend])
        if metric == "chunk_tag_homogeneity":
            command.extend([
                "--chunk-count",
                str(args.chunk_count),
                "--model-name",
                args.chunk_model_name,
                "--max-tags-per-chunk",
                str(args.max_tags_per_chunk),
                "--global-tag-min",
                str(args.global_tag_min),
                "--global-tag-max",
                str(args.global_tag_max),
            ])
            if args.chunk_refresh:
                command.append("--refresh")
        if metric == "dual_layer_trope_homogeneity":
            command.extend([
                "--chunk-count",
                str(args.chunk_count),
                "--model-name",
                args.chunk_model_name,
                "--max-macro-tags",
                str(args.max_macro_tags),
                "--max-micro-motifs",
                str(args.max_micro_motifs),
            ])
            if args.chunk_refresh:
                command.append("--refresh")
        print(f"运行 {metric} ...")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
