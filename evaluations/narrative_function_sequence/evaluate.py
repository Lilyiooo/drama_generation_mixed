from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluations.common.annotations import FUNCTIONS
from evaluations.common.cli_utils import add_common_arguments, prepare
from evaluations.common.io import save_json
from evaluations.common.math_utils import lcs_similarity, levenshtein_similarity, mean


def ngrams(sequence: list[str], order: int):
    return [tuple(sequence[index : index + order]) for index in range(len(sequence) - order + 1)]


def repeated_excess_ratio(items) -> float:
    data = list(items)
    if not data:
        return 0.0
    counts = Counter(data)
    repeated_excess = sum(max(count - 1, 0) for count in counts.values())
    return repeated_excess / len(data)


def transition_statistics(sequences: list[list[str]]) -> dict:
    transitions: dict[str, Counter[str]] = defaultdict(Counter)
    for sequence in sequences:
        for left, right in zip(sequence, sequence[1:]):
            transitions[left][right] += 1
    total_transitions = sum(sum(counter.values()) for counter in transitions.values())
    weighted_entropy = 0.0
    deterministic = 0
    details = []
    for source, targets in sorted(transitions.items()):
        total = sum(targets.values())
        probabilities = [count / total for count in targets.values()]
        raw_entropy = -sum(probability * math.log(probability) for probability in probabilities)
        normalized = raw_entropy / math.log(len(targets)) if len(targets) > 1 else 0.0
        weighted_entropy += total * normalized
        if len(targets) == 1:
            deterministic += total
        details.append(
            {
                "source": source,
                "source_name": FUNCTIONS.get(source, source),
                "count": total,
                "normalized_entropy": normalized,
                "targets": dict(targets.most_common()),
            }
        )
    return {
        "weighted_normalized_transition_entropy": weighted_entropy / total_transitions if total_transitions else 0.0,
        "deterministic_transition_ratio": deterministic / total_transitions if total_transitions else 0.0,
        "transition_count": total_transitions,
        "by_source": details,
    }


def evaluate(annotations) -> dict:
    scene_sequences = [annotation.narrative_functions for annotation in annotations]
    signatures = [tuple(sequence) for sequence in scene_sequences]
    exact_counts = Counter(signatures)
    ngram_metrics = {}
    for order in (2, 3):
        all_ngrams = [gram for sequence in scene_sequences for gram in ngrams(sequence, order)]
        counts = Counter(all_ngrams)
        ngram_metrics[str(order)] = {
            "total_occurrences": len(all_ngrams),
            "unique_count": len(counts),
            "repeated_excess_ratio": repeated_excess_ratio(all_ngrams),
            "top_patterns": [
                {
                    "pattern": list(pattern),
                    "names": [FUNCTIONS.get(code, code) for code in pattern],
                    "count": count,
                }
                for pattern, count in counts.most_common(10)
            ],
        }
    pair_details = []
    nearest_edit = []
    nearest_lcs = []
    for left in range(len(scene_sequences)):
        candidates = []
        for right in range(len(scene_sequences)):
            if left == right:
                continue
            edit_score = levenshtein_similarity(scene_sequences[left], scene_sequences[right])
            lcs_score = lcs_similarity(scene_sequences[left], scene_sequences[right])
            candidates.append((max(edit_score, lcs_score), edit_score, lcs_score, right))
        best = max(candidates, default=(0.0, 0.0, 0.0, left))
        nearest_edit.append(best[1])
        nearest_lcs.append(best[2])
        pair_details.append(
            {
                "scene": annotations[left].scene_id,
                "nearest_scene": annotations[best[3]].scene_id,
                "edit_similarity": best[1],
                "lcs_similarity": best[2],
            }
        )
    episode_sequences: dict[int, list[str]] = defaultdict(list)
    for annotation in annotations:
        episode_sequences[annotation.episode_id].extend(annotation.narrative_functions)
    return {
        "metric": "narrative_function_sequence_reuse",
        "method": "34-function narrative representation with exact, n-gram, edit/LCS and transition metrics",
        "scene_count": len(annotations),
        "episode_count": len(episode_sequences),
        "exact_sequence_reuse_rate": repeated_excess_ratio(signatures),
        "exact_sequence_types": [
            {
                "sequence": list(signature),
                "names": [FUNCTIONS.get(code, code) for code in signature],
                "count": count,
            }
            for signature, count in exact_counts.most_common()
        ],
        "function_ngram_reuse": ngram_metrics,
        "mean_nearest_edit_similarity": mean(nearest_edit),
        "mean_nearest_lcs_similarity": mean(nearest_lcs),
        "nearest_scene_pairs": pair_details,
        "scene_transition_statistics": transition_statistics(scene_sequences),
        "episode_transition_statistics": transition_statistics(list(episode_sequences.values())),
        "interpretation": {
            "reuse": "exact_sequence_reuse_rate和功能n-gram重复率越高，套路复用越严重。",
            "predictability": "转移熵越低、确定性转移比例越高，后续剧情越可预测。",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="计算叙事功能序列重复率")
    add_common_arguments(parser)
    args = parser.parse_args()
    _, annotations, source, output = prepare(args, "narrative_function_sequence")
    result = evaluate(annotations)
    result.update({"source": source, "annotator": args.annotator})
    if args.annotator == "heuristic":
        result["warning"] = "功能序列由关键词启发式生成，仅用于管线验证；正式结论需用LLM标注并人工复核。"
    save_json(output, result)
    print(f"结果已写入：{output}")


if __name__ == "__main__":
    main()
