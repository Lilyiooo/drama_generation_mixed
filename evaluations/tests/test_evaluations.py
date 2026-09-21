from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluations.chunk_tag_homogeneity.evaluate import chunk_episodes, score_chunk_sequences
from evaluations.common.math_utils import lcs_similarity, levenshtein_similarity
from evaluations.dual_layer_trope_homogeneity.evaluate import score_dual_layer_patterns
from evaluations.common.models import SceneAnnotation
from evaluations.functional_equivalence_reuse.evaluate import evaluate as evaluate_equivalence
from evaluations.narrative_function_sequence.evaluate import evaluate as evaluate_sequences
from evaluations.semantic_homogeneity.evaluate import centroid_homogeneity
from evaluations.structural_vendi.evaluate import effective_rank, evaluate as evaluate_vendi


def annotation(index: int, pattern: str, functions: list[str]) -> SceneAnnotation:
    return SceneAnnotation(
        scene_id=f"E{index:03d}S001",
        episode_id=index,
        scene_number=1,
        goal=pattern,
        obstacle="共同阻碍",
        action=pattern,
        turn=pattern,
        outcome=pattern,
        hook="新危机",
        state_change=pattern,
        narrative_functions=functions,
        annotation_method="test",
    )


def test_sequence_similarity_extremes():
    assert levenshtein_similarity(["A", "Q", "S"], ["A", "Q", "S"]) == 1.0
    assert lcs_similarity(["A", "Q", "S"], ["A", "Q", "S"]) == 1.0


def test_equivalence_reuse_detects_duplicate_patterns():
    data = [
        annotation(1, "遇敌后隐藏能力反杀", ["A", "Q", "O", "S"]),
        annotation(2, "遇敌后隐藏能力反杀", ["A", "Q", "O", "S"]),
        annotation(3, "关系因误会而破裂", ["A", "Em", "Ch"]),
    ]
    result = evaluate_equivalence(data, threshold=0.8, permutations=5, seed=1)
    assert result["canonical_distinct_classes"] == 2
    assert np.isclose(result["canonical_reuse_rate"], 1 / 3)


def test_sequence_reuse_detects_exact_duplicate():
    data = [
        annotation(1, "甲", ["A", "Q", "S"]),
        annotation(2, "乙", ["A", "Q", "S"]),
        annotation(3, "丙", ["A", "Em", "Ch"]),
    ]
    result = evaluate_sequences(data)
    assert result["exact_sequence_reuse_rate"] == 1 / 3
    assert result["function_ngram_reuse"]["3"]["repeated_excess_ratio"] == 1 / 3


def test_vendi_effective_rank_extremes():
    assert np.isclose(effective_rank(np.array([1.0, 0.0])), 1.0)
    assert np.isclose(effective_rank(np.array([0.5, 0.5])), 2.0)


def test_structural_vendi_is_bounded():
    data = [
        annotation(1, "相同模板", ["A", "Q", "S"]),
        annotation(2, "相同模板", ["A", "Q", "S"]),
        annotation(3, "不同关系推进", ["A", "Em", "Ch"]),
    ]
    result = evaluate_vendi(data, q=1.0)
    assert 1.0 <= result["scene_level"]["vendi_score"] <= 3.0


def test_centroid_homogeneity_extremes():
    identical = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert np.isclose(centroid_homogeneity(identical), 1.0)


def test_chunk_episodes_keeps_order_and_balance():
    episodes = [
        {"episode_id": index, "scene_count": 1, "text": f"第{index}集"}
        for index in range(1, 8)
    ]
    chunks = chunk_episodes(episodes, 3)
    assert [chunk["episode_ids"] for chunk in chunks] == [[1, 2, 3], [4, 5], [6, 7]]


def test_chunk_tag_score_detects_repeated_sequences():
    chunks = [
        {"chunk_id": "C01", "episode_range": [1, 2], "scene_count": 2},
        {"chunk_id": "C02", "episode_range": [3, 4], "scene_count": 2},
        {"chunk_id": "C03", "episode_range": [5, 6], "scene_count": 2},
    ]
    global_tags = [
        {"tag_id": "T01", "name": "身份压制制造危机"},
        {"tag_id": "T02", "name": "嘴炮反制翻盘"},
        {"tag_id": "T03", "name": "新危机钩子收尾"},
        {"tag_id": "T04", "name": "关系试探推进"},
    ]
    relabeled = [
        {
            "chunk_id": "C01",
            "chunk_summary": "甲",
            "ordered_global_tags": [
                {"tag_id": "T01", "tag_name": "身份压制制造危机"},
                {"tag_id": "T02", "tag_name": "嘴炮反制翻盘"},
                {"tag_id": "T03", "tag_name": "新危机钩子收尾"},
            ],
        },
        {
            "chunk_id": "C02",
            "chunk_summary": "乙",
            "ordered_global_tags": [
                {"tag_id": "T01", "tag_name": "身份压制制造危机"},
                {"tag_id": "T02", "tag_name": "嘴炮反制翻盘"},
                {"tag_id": "T03", "tag_name": "新危机钩子收尾"},
            ],
        },
        {
            "chunk_id": "C03",
            "chunk_summary": "丙",
            "ordered_global_tags": [
                {"tag_id": "T04", "tag_name": "关系试探推进"},
                {"tag_id": "T03", "tag_name": "新危机钩子收尾"},
            ],
        },
    ]
    result = score_chunk_sequences(chunks, global_tags, relabeled)
    assert np.isclose(result["exact_chunk_sequence_reuse_rate"], 1 / 3)
    assert result["chunk_tag_ngram_reuse"]["2"]["repeated_excess_ratio"] > 0
    assert result["global_tag_count"] == 4


def test_dual_layer_score_detects_micro_causal_reuse():
    chunks = [
        {"chunk_id": "C01", "episode_range": [1, 2], "episode_ids": [1, 2], "scene_count": 2},
        {"chunk_id": "C02", "episode_range": [3, 4], "episode_ids": [3, 4], "scene_count": 2},
        {"chunk_id": "C03", "episode_range": [5, 6], "episode_ids": [5, 6], "scene_count": 2},
    ]
    global_macro_tags = [
        {"tag_id": "MT01", "name": "公开施压求解"},
        {"tag_id": "MT02", "name": "受挫后转入查证"},
        {"tag_id": "MT03", "name": "爆牌逆转"},
    ]
    global_micro_taxonomy = {
        "causes": [
            {"cause_id": "C01", "cause_name": "高位打压"},
            {"cause_id": "C02", "cause_name": "旧法失灵"},
        ],
        "methods": [
            {"method_id": "A01", "method_name": "公开施压"},
            {"method_id": "A02", "method_name": "转入查证"},
        ],
        "outcomes": [
            {"outcome_id": "O01", "outcome_name": "施压失败"},
            {"outcome_id": "O02", "outcome_name": "获得线索"},
        ],
        "responses": [
            {"response_id": "R01", "response_name": "被迫变招"},
            {"response_id": "R02", "response_name": "乘胜追击"},
        ],
    }
    relabeled_chunks = [
        {
            "chunk_id": "C01",
            "chunk_summary": "甲",
            "ordered_macro_tags": [
                {"tag_id": "MT01", "tag_name": "公开施压求解", "reason": "公开施压未奏效"},
                {"tag_id": "MT02", "tag_name": "受挫后转入查证", "reason": "失败后改从查证入手"},
            ],
            "ordered_micro_motifs": [
                {
                    "macro_anchor_tag_id": "MT01",
                    "cause_id": "C01",
                    "cause_name": "高位打压",
                    "method_id": "A01",
                    "method_name": "公开施压",
                    "outcome_id": "O01",
                    "outcome_name": "施压失败",
                    "response_id": "R01",
                    "response_name": "被迫变招",
                    "reason": "施压失败后只得变招",
                },
                {
                    "macro_anchor_tag_id": "MT02",
                    "cause_id": "C02",
                    "cause_name": "旧法失灵",
                    "method_id": "A02",
                    "method_name": "转入查证",
                    "outcome_id": "O02",
                    "outcome_name": "获得线索",
                    "response_id": "R02",
                    "response_name": "乘胜追击",
                    "reason": "查证后掌握线索继续推进",
                },
            ],
        },
        {
            "chunk_id": "C02",
            "chunk_summary": "乙",
            "ordered_macro_tags": [
                {"tag_id": "MT01", "tag_name": "公开施压求解", "reason": "再次尝试公开施压"},
                {"tag_id": "MT02", "tag_name": "受挫后转入查证", "reason": "受阻后只能回头查证"},
            ],
            "ordered_micro_motifs": [
                {
                    "macro_anchor_tag_id": "MT01",
                    "cause_id": "C01",
                    "cause_name": "高位打压",
                    "method_id": "A01",
                    "method_name": "公开施压",
                    "outcome_id": "O01",
                    "outcome_name": "施压失败",
                    "response_id": "R01",
                    "response_name": "被迫变招",
                    "reason": "这次公开施压也失败，只能换招",
                },
                {
                    "macro_anchor_tag_id": "MT02",
                    "cause_id": "C02",
                    "cause_name": "旧法失灵",
                    "method_id": "A02",
                    "method_name": "转入查证",
                    "outcome_id": "O02",
                    "outcome_name": "获得线索",
                    "response_id": "R02",
                    "response_name": "乘胜追击",
                    "reason": "改走查证线后再次拿到线索",
                },
            ],
        },
        {
            "chunk_id": "C03",
            "chunk_summary": "丙",
            "ordered_macro_tags": [
                {"tag_id": "MT03", "tag_name": "爆牌逆转", "reason": "改为直接爆牌反杀"},
            ],
            "ordered_micro_motifs": [
                {
                    "macro_anchor_tag_id": "MT03",
                    "cause_id": "C02",
                    "cause_name": "旧法失灵",
                    "method_id": "A02",
                    "method_name": "转入查证",
                    "outcome_id": "O02",
                    "outcome_name": "获得线索",
                    "response_id": "R02",
                    "response_name": "乘胜追击",
                    "reason": "线索坐实后直接推进反杀",
                },
            ],
        },
    ]
    result = score_dual_layer_patterns(chunks, global_macro_tags, global_micro_taxonomy, relabeled_chunks)
    assert result["macro_layer"]["global_tag_count"] == 3
    assert result["micro_layer"]["causal_motif_reuse"]["exact_motif_reuse_rate"] > 0
    assert result["micro_layer"]["causal_motif_reuse"]["method_outcome_response_reuse_rate"] > 0
    assert result["micro_layer"]["conditional_transition_entropy"]["transition_count"] == 5
    assert result["micro_layer"]["motif_phase_coverage"]["micro_methods"]["weighted_phase_coverage_ratio"] > 0
