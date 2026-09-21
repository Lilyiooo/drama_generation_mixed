from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from evaluations.common.cli_utils import add_common_arguments, prepare
from evaluations.common.io import save_json
from evaluations.common.math_utils import percentile
from evaluations.common.vectorize import cosine_matrix, encode_texts, l2_normalize


def centroid_homogeneity(vectors: np.ndarray) -> float:
    center = vectors.mean(axis=0, keepdims=True)
    center = l2_normalize(center)[0]
    return float(np.mean(vectors @ center))


def bootstrap_interval(vectors: np.ndarray, runs: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    scores = []
    for _ in range(runs):
        indices = [rng.randrange(len(vectors)) for _ in range(len(vectors))]
        scores.append(centroid_homogeneity(vectors[indices]))
    return [percentile(scores, 0.025), percentile(scores, 0.975)]


def summarize_level(vectors: np.ndarray, ids: list[str], bootstrap_runs: int, seed: int) -> dict:
    similarities = cosine_matrix(vectors)
    count = len(vectors)
    off_diagonal = similarities[~np.eye(count, dtype=bool)] if count > 1 else np.array([])
    nearest = []
    pairs = []
    for index in range(count):
        row = similarities[index].copy()
        row[index] = -np.inf
        nearest_index = int(np.argmax(row)) if count > 1 else index
        score = float(row[nearest_index]) if count > 1 else 1.0
        nearest.append(score)
        pairs.append({"item": ids[index], "nearest_item": ids[nearest_index], "similarity": score})
    return {
        "sample_count": count,
        "semantic_homogeneity": centroid_homogeneity(vectors),
        "semantic_homogeneity_95pct_bootstrap_interval": bootstrap_interval(vectors, bootstrap_runs, seed),
        "mean_pairwise_similarity": float(np.mean(off_diagonal)) if len(off_diagonal) else 1.0,
        "mean_nearest_neighbor_similarity": float(np.mean(nearest)),
        "nearest_neighbors": pairs,
    }


def evaluate(annotations, backend: str, model_name: str, bootstrap_runs: int, seed: int) -> dict:
    summaries = [annotation.structural_summary for annotation in annotations]
    vectors, backend_used, warnings = encode_texts(summaries, backend, model_name)
    episode_rows: dict[int, list[int]] = defaultdict(list)
    for index, annotation in enumerate(annotations):
        episode_rows[annotation.episode_id].append(index)
    episode_vectors = []
    episode_ids = []
    for episode_id in sorted(episode_rows):
        episode_vectors.append(vectors[episode_rows[episode_id]].mean(axis=0))
        episode_ids.append(f"E{episode_id:03d}")
    episode_matrix = l2_normalize(np.asarray(episode_vectors, dtype=np.float64))
    result = {
        "metric": "structural_summary_semantic_homogeneity",
        "method": "Wang & Kreminski centroid homogeneity adapted to within-drama structural summaries",
        "embedding_backend": backend_used,
        "scene_level": summarize_level(
            vectors, [annotation.scene_id for annotation in annotations], bootstrap_runs, seed
        ),
        "episode_level": summarize_level(
            episode_matrix, episode_ids, bootstrap_runs, seed + 1
        ),
        "interpretation": "semantic_homogeneity越接近1，结构化剧情摘要越集中；它可能混入题材一致性，必须结合功能序列指标解释。",
        "scientific_notes": [
            "输入是目标—阻碍—行动—转折—结果—状态变化—钩子的结构化摘要，不是原始台词。",
            "论文原指标比较同一段落位置的多个故事；这里是用于单剧内部比较的明确改造版。",
            "置信区间反映有限样本不确定性，不代表标注误差。",
        ],
    }
    if warnings:
        result["warnings"] = warnings
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="计算结构化摘要语义同质性")
    add_common_arguments(parser)
    parser.add_argument(
        "--backend",
        choices=["auto", "sentence-transformers", "tfidf-char"],
        default="auto",
    )
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2")
    parser.add_argument("--bootstrap-runs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    _, annotations, source, output = prepare(args, "semantic_homogeneity")
    result = evaluate(annotations, args.backend, args.model_name, args.bootstrap_runs, args.seed)
    result.update({"source": source, "annotator": args.annotator})
    if args.annotator == "heuristic":
        result.setdefault("warnings", []).append(
            "启发式结构标注仅用于管线验证；正式结果需使用LLM标注并抽样人工复核。"
        )
    save_json(output, result)
    print(f"结果已写入：{output}")


if __name__ == "__main__":
    main()
