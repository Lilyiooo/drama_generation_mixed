from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from evaluations.common.cli_utils import add_common_arguments, prepare
from evaluations.common.io import save_json
from evaluations.common.math_utils import mean, percentile, population_std
from evaluations.common.vectorize import char_tfidf, cosine_matrix

FIELD_WEIGHTS = {
    "goal": 0.15,
    "obstacle": 0.15,
    "action": 0.20,
    "turn": 0.20,
    "outcome": 0.15,
    "hook": 0.05,
    "state_change": 0.10,
}


def functional_similarity_matrix(annotations) -> np.ndarray:
    size = len(annotations)
    combined = np.zeros((size, size), dtype=np.float64)
    for field, weight in FIELD_WEIGHTS.items():
        vectors = char_tfidf([getattr(annotation, field) for annotation in annotations])
        combined += weight * np.maximum(cosine_matrix(vectors), 0.0)
    function_sets = [set(annotation.narrative_functions) for annotation in annotations]
    function_similarity = np.eye(size, dtype=np.float64)
    for left in range(size):
        for right in range(left + 1, size):
            union = function_sets[left] | function_sets[right]
            score = len(function_sets[left] & function_sets[right]) / len(union) if union else 1.0
            function_similarity[left, right] = function_similarity[right, left] = score
    return np.clip(0.8 * combined + 0.2 * function_similarity, 0.0, 1.0)


def sequential_distinct_count(order: list[int], similarities: np.ndarray, threshold: float) -> tuple[int, list[list[int]]]:
    clusters: list[list[int]] = []
    for index in order:
        for cluster in clusters:
            representative = cluster[0]
            if similarities[index, representative] >= threshold:
                cluster.append(index)
                break
        else:
            clusters.append([index])
    return len(clusters), clusters


def evaluate(annotations, threshold: float, permutations: int, seed: int) -> dict:
    similarities = functional_similarity_matrix(annotations)
    rng = random.Random(seed)
    orders = [list(range(len(annotations)))]
    for _ in range(max(permutations - 1, 0)):
        order = list(range(len(annotations)))
        rng.shuffle(order)
        orders.append(order)
    counts = []
    clusterings = []
    for order in orders:
        count, clusters = sequential_distinct_count(order, similarities, threshold)
        counts.append(count)
        clusterings.append(clusters)
    canonical_count, canonical_clusters = sequential_distinct_count(
        list(range(len(annotations))), similarities, threshold
    )
    pairs = []
    for left in range(len(annotations)):
        for right in range(left + 1, len(annotations)):
            score = float(similarities[left, right])
            if score >= threshold:
                pairs.append(
                    {
                        "left": annotations[left].scene_id,
                        "right": annotations[right].scene_id,
                        "similarity": round(score, 6),
                    }
                )
    pairs.sort(key=lambda item: item["similarity"], reverse=True)
    return {
        "metric": "scene_functional_equivalence_reuse",
        "method": "NoveltyBench distinct_k adapted to within-drama scenes",
        "sample_count": len(annotations),
        "equivalence_threshold": threshold,
        "canonical_distinct_classes": canonical_count,
        "canonical_distinct_ratio": canonical_count / len(annotations),
        "canonical_reuse_rate": 1.0 - canonical_count / len(annotations),
        "permutation_robustness": {
            "runs": len(counts),
            "distinct_classes_mean": mean(counts),
            "distinct_classes_std": population_std(counts),
            "distinct_classes_95pct_interval": [percentile(counts, 0.025), percentile(counts, 0.975)],
            "reuse_rate_mean": 1.0 - mean(counts) / len(annotations),
        },
        "clusters": [
            {
                "cluster_id": position + 1,
                "size": len(cluster),
                "members": [annotations[index].scene_id for index in cluster],
                "representative_summary": annotations[cluster[0]].structural_summary,
            }
            for position, cluster in enumerate(canonical_clusters)
        ],
        "equivalent_pairs": pairs,
        "interpretation": "复用率越高，表示越多场景仅是同一叙事功能模板的变体；阈值必须用人工场景对校准。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="计算场景功能等价复用率")
    add_common_arguments(parser)
    parser.add_argument("--threshold", type=float, default=0.82)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    _, annotations, source, output = prepare(args, "functional_equivalence_reuse")
    result = evaluate(annotations, args.threshold, args.permutations, args.seed)
    result.update({"source": source, "annotator": args.annotator})
    if args.annotator == "heuristic":
        result["warning"] = "启发式标注结果仅用于管线验证；正式科学评测应使用LLM标注并抽样人工复核。"
    save_json(output, result)
    print(f"结果已写入：{output}")


if __name__ == "__main__":
    main()
