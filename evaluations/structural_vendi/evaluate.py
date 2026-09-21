from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from evaluations.common.annotations import FUNCTIONS
from evaluations.common.cli_utils import add_common_arguments, prepare
from evaluations.common.io import save_json
from evaluations.common.vectorize import char_tfidf, l2_normalize

FIELD_WEIGHTS = {
    "goal": 0.12,
    "obstacle": 0.16,
    "action": 0.18,
    "turn": 0.20,
    "outcome": 0.14,
    "hook": 0.08,
    "state_change": 0.07,
    "narrative_functions": 0.05,
}


def structural_features(annotations) -> np.ndarray:
    blocks = []
    for field, weight in FIELD_WEIGHTS.items():
        if field == "narrative_functions":
            codes = list(FUNCTIONS)
            index = {code: position for position, code in enumerate(codes)}
            block = np.zeros((len(annotations), len(codes)), dtype=np.float64)
            for row, annotation in enumerate(annotations):
                for code in annotation.narrative_functions:
                    if code in index:
                        block[row, index[code]] += 1.0
            block = l2_normalize(block)
        else:
            block = char_tfidf([getattr(annotation, field) for annotation in annotations])
        blocks.append(math.sqrt(weight) * block)
    return l2_normalize(np.concatenate(blocks, axis=1))


def effective_rank(eigenvalues: np.ndarray, q: float = 1.0) -> float:
    values = np.clip(np.asarray(eigenvalues, dtype=np.float64), 0.0, None)
    values = values / values.sum()
    values = values[values > 1e-12]
    if math.isclose(q, 1.0):
        return float(np.exp(-np.sum(values * np.log(values))))
    if math.isclose(q, 0.0):
        return float(len(values))
    if math.isinf(q):
        return float(1.0 / np.max(values))
    if q < 0:
        raise ValueError("q 必须大于等于0")
    return float(np.sum(values**q) ** (1.0 / (1.0 - q)))


def score_features(features: np.ndarray, q: float) -> dict:
    count = len(features)
    if features.shape[1] < count:
        spectral_matrix = features.T @ features / count
    else:
        spectral_matrix = features @ features.T / count
    eigenvalues = np.linalg.eigvalsh((spectral_matrix + spectral_matrix.T) / 2.0)
    score = effective_rank(eigenvalues, q)
    return {
        "sample_count": count,
        "vendi_score": score,
        "vendi_per_sample": score / count,
        "effective_redundancy": 1.0 - score / count,
        "q": q,
        "positive_eigenvalue_count": int(np.sum(eigenvalues > 1e-12)),
        "minimum_eigenvalue": float(np.min(eigenvalues)),
    }


def evaluate(annotations, q: float) -> dict:
    features = structural_features(annotations)
    episode_rows: dict[int, list[int]] = defaultdict(list)
    for index, annotation in enumerate(annotations):
        episode_rows[annotation.episode_id].append(index)
    episode_features = []
    for episode_id in sorted(episode_rows):
        episode_features.append(features[episode_rows[episode_id]].mean(axis=0))
    episode_matrix = l2_normalize(np.asarray(episode_features, dtype=np.float64))
    result = {
        "metric": "structural_vendi_score",
        "method": "Vendi Score over a PSD linear kernel of weighted structural features",
        "field_weights": FIELD_WEIGHTS,
        "scene_level": score_features(features, q),
        "episode_level": score_features(episode_matrix, q),
        "interpretation": "Vendi Score是有效不同模板数；vendi_per_sample越低、effective_redundancy越高，结构同质化越严重。",
        "scientific_notes": [
            "采用归一化特征内积核，确保核矩阵正半定且对角线为1。",
            "Vendi只衡量多样性，不衡量质量；需与剧本质量评分联合报告。",
            "不同字段权重或标注器下的绝对分数不可直接横比。",
        ],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="计算场景和分集结构Vendi Score")
    add_common_arguments(parser)
    parser.add_argument("--q", type=float, default=1.0, help="Vendi/Hill阶数，默认Shannon阶q=1")
    args = parser.parse_args()
    _, annotations, source, output = prepare(args, "structural_vendi")
    result = evaluate(annotations, args.q)
    result.update({"source": source, "annotator": args.annotator})
    if args.annotator == "heuristic":
        result["warning"] = "启发式结构标注仅用于管线验证；正式结果需使用LLM标注并抽样人工复核。"
    save_json(output, result)
    print(f"结果已写入：{output}")


if __name__ == "__main__":
    main()
