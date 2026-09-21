from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluations.chunk_tag_homogeneity.evaluate import (
    chunk_episodes,
    load_cache,
    load_episode_units,
    model_slug,
    save_cache,
    score_chunk_sequences,
    source_hash,
)
from evaluations.common.io import save_json
from evaluations.common.math_utils import mean

try:
    from evaluations.common.project_llm_trpc import query_json
except ImportError:
    from evaluations.common.project_llm import query_json

from evaluations.narrative_function_sequence.evaluate import repeated_excess_ratio

MICRO_SLOT_SPECS: dict[str, dict[str, str]] = {
    "causes": {"id_key": "cause_id", "name_key": "cause_name", "short": "C01", "label": "起因"},
    "methods": {"id_key": "method_id", "name_key": "method_name", "short": "A01", "label": "手段"},
    "outcomes": {"id_key": "outcome_id", "name_key": "outcome_name", "short": "O01", "label": "结果"},
    "responses": {"id_key": "response_id", "name_key": "response_name", "short": "R01", "label": "反应"},
}
PHASE_NAMES = ("early", "middle", "late")


def _safe_episode_ids(values: list[Any]) -> list[int]:
    result = []
    for value in values:
        if str(value).isdigit():
            result.append(int(value))
    return result


def prompt_local_dual_layer(chunk: dict[str, Any], max_macro_tags: int, max_micro_motifs: int) -> list[dict[str, str]]:
    schema = {
        "chunk_summary": "一句话概括这一段主线推进",
        "macro_tags": [
            {
                "label": "带推进关系的中观套路标签",
                "definition": "一句话说明这个标签对应的叙事推进机制",
                "evidence_episodes": [chunk["episode_range"][0]],
                "reason": "为什么该片段适合这个标签",
            }
        ],
        "micro_motifs": [
            {
                "cause": "高位打压",
                "method": "公开施压",
                "outcome": "施压失败",
                "response": "被迫变招",
                "evidence_episodes": [chunk["episode_range"][0]],
                "reason": "一句话说明这个微观因果链为何成立",
            }
        ],
    }
    prompt = f"""你是短剧套路分析员。现在只分析一个剧本片段，不评价文笔。
任务：为这个片段提炼一套双层套路表示，用于后续剧内套路化检测。

第一层（macro_tags）要求：
1. 标签必须是带推进关系的中观叙事动作，不能只是互相孤立的功能分类。
2. 标签风格示例：公开施压求解、受挫后转入查证、示弱潜伏等待翻盘、设局推进最终收网、阶段胜利后引出更大危机。
3. 标签要能和前后标签形成语义上的推进链条，而不是一堆并列名词。
4. 不要写人名地名道具，不要过细，不要写成剧情摘要。
5. 按阅读顺序排列。

第二层（micro_motifs）要求：
1. 每个 motif 须写成 起因 -> 手段 -> 结果 -> 反应 的抽象因果链。
2. 每个字段必须用简短词组概括（2-8字），不能写长句。示例：高位打压、公开施压、施压失败、被迫变招。
3. 足够笼统便于跨片段聚类，但不能空泛到只剩"冲突/处理/结局/继续"。
4. 不要出现具体人名、地点、道具、事件名。
5. 若同一片段内出现多个明显不同的微观套路链，可以写多个。
6. 按阅读顺序排列。

归纳原则：能合理归并就归并，不能归并就保留差异，不要为了追求数量而硬拆或硬并。
只输出 JSON 对象，不要额外解释。
JSON格式示例：{schema}
片段编号：{chunk['chunk_id']}
覆盖集数：{chunk['episode_range'][0]}-{chunk['episode_range'][1]}
片段正文：
{chunk['text'][:24000]}
"""
    return [{"role": "user", "content": prompt}]


def prompt_global_macro_tagbook(local_results: list[dict[str, Any]]) -> list[dict[str, str]]:
    compact = []
    for item in local_results:
        compact.append(
            {
                "chunk_id": item["chunk_id"],
                "episode_range": item["episode_range"],
                "chunk_summary": item["chunk_summary"],
                "macro_tags": [
                    {"label": tag["label"], "definition": tag["definition"]} for tag in item["macro_tags"]
                ],
            }
        )
    schema = {
        "global_macro_tags": [
            {
                "tag_id": "MT01",
                "name": "标签名",
                "definition": "标签定义",
                "includes": ["包含哪些局部叫法"],
                "excludes": ["容易混淆但不应并入的模式"],
                "merged_local_tags": ["被并入的原始局部标签"],
            }
        ]
    }
    prompt = f"""你是短剧标签体系设计员。下面给你的是同一部剧多个片段各自提炼出的第一层中观套路标签。
任务：将这些局部中观标签归并成一套全剧统一的第一层标签表。
要求：
1. 标签必须保留"推进关系/叙事动作"的语义，不要退化成孤立功能名词。
2. 能合理归并就归并，不能归并就保留差异；不要为了追求数量而硬拆或硬并。
3. tag_id 必须连续编号 MT01, MT02 ...
4. includes / merged_local_tags 写被并入的局部说法；excludes 写边界。
5. 只输出 JSON 对象。
JSON格式示例：{schema}
局部标签输入：
{compact}
"""
    return [{"role": "user", "content": prompt}]


def prompt_global_micro_taxonomy(local_results: list[dict[str, Any]]) -> list[dict[str, str]]:
    compact = []
    for item in local_results:
        compact.append(
            {
                "chunk_id": item["chunk_id"],
                "episode_range": item["episode_range"],
                "micro_motifs": [
                    {
                        "cause": motif["cause"],
                        "method": motif["method"],
                        "outcome": motif["outcome"],
                        "response": motif["response"],
                    }
                    for motif in item["micro_motifs"]
                ],
            }
        )
    schema = {
        "causes": [
            {
                "cause_id": "C01",
                "name": "起因类名",
                "definition": "定义",
                "includes": ["被并入的局部表述"],
                "excludes": ["不应并入的边界"],
                "merged_local_values": ["原始值"],
            }
        ],
        "methods": [
            {
                "method_id": "A01",
                "name": "手段类名",
                "definition": "定义",
                "includes": ["被并入的局部表述"],
                "excludes": ["不应并入的边界"],
                "merged_local_values": ["原始值"],
            }
        ],
        "outcomes": [
            {
                "outcome_id": "O01",
                "name": "结果类名",
                "definition": "定义",
                "includes": ["被并入的局部表述"],
                "excludes": ["不应并入的边界"],
                "merged_local_values": ["原始值"],
            }
        ],
        "responses": [
            {
                "response_id": "R01",
                "name": "反应类名",
                "definition": "定义",
                "includes": ["被并入的局部表述"],
                "excludes": ["不应并入的边界"],
                "merged_local_values": ["原始值"],
            }
        ],
    }
    prompt = f"""你是短剧套路分析体系设计员。下面给你的是同一部剧多个片段提炼出的第二层微观因果链。
任务：分别把 起因 / 手段 / 结果 / 反应 四个槽位中的局部表述归并成全剧统一词表。
要求：
1. 四个槽位分别独立归并，不能混槽位。
2. 每个类都要保持抽象、可复用、便于后续聚类和 n-gram 统计。
3. 能合理归并就归并，不能归并就保留差异；不要为了追求数量而硬拆或硬并。
4. ID 规则：cause_id 用 C01...；method_id 用 A01...；outcome_id 用 O01...；response_id 用 R01...
5. 只输出 JSON 对象。
JSON格式示例：{schema}
局部微观因果链输入：
{compact}
"""
    return [{"role": "user", "content": prompt}]


def prompt_relabel_dual_layer(
    chunk: dict[str, Any],
    global_macro_tags: list[dict[str, Any]],
    global_micro_taxonomy: dict[str, list[dict[str, Any]]],
    max_macro_tags: int,
    max_micro_motifs: int,
) -> list[dict[str, str]]:
    schema = {
        "chunk_summary": "一句话概括",
        "ordered_macro_tags": [
            {
                "tag_id": "MT01",
                "tag_name": "标签名",
                "evidence_episodes": [chunk["episode_range"][0]],
                "reason": "为什么这里出现该标签",
            }
        ],
        "ordered_micro_motifs": [
            {
                "macro_anchor_tag_id": "MT01",
                "cause_id": "C01",
                "method_id": "A01",
                "outcome_id": "O01",
                "response_id": "R01",
                "evidence_episodes": [chunk["episode_range"][0]],
                "reason": "为什么这里可以抽象成这条微观因果链",
            }
        ],
    }
    prompt = f"""你是短剧套路回标员。现在要用固定的双层标签体系，对一个剧本片段重新打标。
要求：
1. 第一层只能从给定 macro tag_id 中选择，不得自创。
2. 第二层四个槽位只能从给定词表 ID 中选择，不得自创。
3. 按阅读顺序排列。
4. micro motif 的 macro_anchor_tag_id 填写它最贴近的第一层标签 ID。
5. 只输出 JSON 对象。
第一层标签表：
{global_macro_tags}
第二层词表：
{global_micro_taxonomy}
JSON格式示例：{schema}
片段编号：{chunk['chunk_id']}
覆盖集数：{chunk['episode_range'][0]}-{chunk['episode_range'][1]}
片段正文：
{chunk['text'][:24000]}
"""
    return [{"role": "user", "content": prompt}]


def run_local_dual_extraction(
    chunks: list[dict[str, Any]],
    cache_path: Path,
    expected_hash: str,
    model_name: str,
    max_macro_tags: int,
    max_micro_motifs: int,
    refresh: bool,
) -> list[dict[str, Any]]:
    cached = None if refresh else load_cache(cache_path, expected_hash)
    if cached:
        return cached["chunk_local_dual"]
    results = []
    for chunk in chunks:
        parsed, raw = query_json(
            prompt_local_dual_layer(chunk, max_macro_tags, max_micro_motifs),
            model_name=model_name,
            max_new_tokens=12288,
        )
        macro_tags = parsed.get("macro_tags", [])
        micro_motifs = parsed.get("micro_motifs", [])
        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "episode_range": chunk["episode_range"],
                "chunk_summary": str(parsed.get("chunk_summary", "")).strip(),
                "macro_tags": [
                    {
                        "label": str(tag.get("label", "")).strip(),
                        "definition": str(tag.get("definition", "")).strip(),
                        "evidence_episodes": _safe_episode_ids(tag.get("evidence_episodes", [])),
                        "reason": str(tag.get("reason", "")).strip(),
                    }
                    for tag in macro_tags
                    if str(tag.get("label", "")).strip()
                ],
                "micro_motifs": [
                    {
                        "cause": str(motif.get("cause", "")).strip(),
                        "method": str(motif.get("method", "")).strip(),
                        "outcome": str(motif.get("outcome", "")).strip(),
                        "response": str(motif.get("response", "")).strip(),
                        "evidence_episodes": _safe_episode_ids(motif.get("evidence_episodes", [])),
                        "reason": str(motif.get("reason", "")).strip(),
                    }
                    for motif in micro_motifs
                    if all(str(motif.get(field, "")).strip() for field in ("cause", "method", "outcome", "response"))
                ],
                "raw_response": raw,
            }
        )
    save_cache(cache_path, expected_hash, {"chunk_local_dual": results, "model_name": model_name})
    return results


def build_global_macro_tagbook(
    local_results: list[dict[str, Any]],
    cache_path: Path,
    expected_hash: str,
    model_name: str,
    refresh: bool,
) -> list[dict[str, Any]]:
    cached = None if refresh else load_cache(cache_path, expected_hash)
    if cached:
        return cached["global_macro_tags"]
    parsed, raw = query_json(prompt_global_macro_tagbook(local_results), model_name=model_name, max_new_tokens=12288)
    tags = []
    for index, item in enumerate(parsed.get("global_macro_tags", []), start=1):
        tag_id = str(item.get("tag_id", f"MT{index:02d}")).strip() or f"MT{index:02d}"
        tags.append(
            {
                "tag_id": tag_id,
                "name": str(item.get("name", "")).strip(),
                "definition": str(item.get("definition", "")).strip(),
                "includes": [str(value).strip() for value in item.get("includes", []) if str(value).strip()],
                "excludes": [str(value).strip() for value in item.get("excludes", []) if str(value).strip()],
                "merged_local_tags": [
                    str(value).strip() for value in item.get("merged_local_tags", []) if str(value).strip()
                ],
            }
        )
    save_cache(cache_path, expected_hash, {"global_macro_tags": tags, "raw_response": raw, "model_name": model_name})
    return tags


def _parse_slot_items(slot: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spec = MICRO_SLOT_SPECS[slot]
    parsed = []
    for index, item in enumerate(items, start=1):
        item_id = str(item.get(spec["id_key"], f"{spec['short'][0]}{index:02d}")).strip() or f"{spec['short'][0]}{index:02d}"
        parsed.append(
            {
                spec["id_key"]: item_id,
                spec["name_key"]: str(item.get("name", "")).strip(),
                "definition": str(item.get("definition", "")).strip(),
                "includes": [str(value).strip() for value in item.get("includes", []) if str(value).strip()],
                "excludes": [str(value).strip() for value in item.get("excludes", []) if str(value).strip()],
                "merged_local_values": [
                    str(value).strip() for value in item.get("merged_local_values", []) if str(value).strip()
                ],
            }
        )
    return parsed


def build_global_micro_taxonomy(
    local_results: list[dict[str, Any]],
    cache_path: Path,
    expected_hash: str,
    model_name: str,
    refresh: bool,
) -> dict[str, list[dict[str, Any]]]:
    cached = None if refresh else load_cache(cache_path, expected_hash)
    if cached:
        return cached["global_micro_taxonomy"]
    parsed, raw = query_json(prompt_global_micro_taxonomy(local_results), model_name=model_name, max_new_tokens=12288)
    if not isinstance(parsed, dict):
        parsed = parsed[0] if isinstance(parsed, list) and parsed else {"causes": [], "methods": [], "outcomes": [], "responses": []}
    taxonomy = {slot: _parse_slot_items(slot, parsed.get(slot, [])) for slot in MICRO_SLOT_SPECS}
    save_cache(
        cache_path,
        expected_hash,
        {"global_micro_taxonomy": taxonomy, "raw_response": raw, "model_name": model_name},
    )
    return taxonomy


def run_dual_relabeling(
    chunks: list[dict[str, Any]],
    global_macro_tags: list[dict[str, Any]],
    global_micro_taxonomy: dict[str, list[dict[str, Any]]],
    cache_path: Path,
    expected_hash: str,
    model_name: str,
    max_macro_tags: int,
    max_micro_motifs: int,
    refresh: bool,
) -> list[dict[str, Any]]:
    cached = None if refresh else load_cache(cache_path, expected_hash)
    if cached:
        return cached["chunk_dual_labels"]
    valid_macro_ids = {item["tag_id"] for item in global_macro_tags}
    macro_name_map = {item["tag_id"]: item["name"] for item in global_macro_tags}
    slot_valid_ids = {
        slot: {item[spec["id_key"]] for item in global_micro_taxonomy.get(slot, [])}
        for slot, spec in MICRO_SLOT_SPECS.items()
    }
    slot_name_map = {
        slot: {item[spec["id_key"]]: item[spec["name_key"]] for item in global_micro_taxonomy.get(slot, [])}
        for slot, spec in MICRO_SLOT_SPECS.items()
    }
    results = []
    for chunk in chunks:
        parsed, raw = query_json(
            prompt_relabel_dual_layer(
                chunk,
                global_macro_tags,
                global_micro_taxonomy,
                max_macro_tags,
                max_micro_motifs,
            ),
            model_name=model_name,
            max_new_tokens=12288,
        )
        macro_tags = []
        for item in parsed.get("ordered_macro_tags", []):
            tag_id = str(item.get("tag_id", "")).strip()
            if tag_id not in valid_macro_ids:
                continue
            macro_tags.append(
                {
                    "tag_id": tag_id,
                    "tag_name": macro_name_map.get(tag_id, str(item.get("tag_name", "")).strip()),
                    "evidence_episodes": _safe_episode_ids(item.get("evidence_episodes", [])),
                    "reason": str(item.get("reason", "")).strip(),
                }
            )
        micro_motifs = []
        for item in parsed.get("ordered_micro_motifs", []):
            cause_id = str(item.get("cause_id", "")).strip()
            method_id = str(item.get("method_id", "")).strip()
            outcome_id = str(item.get("outcome_id", "")).strip()
            response_id = str(item.get("response_id", "")).strip()
            if not (
                cause_id in slot_valid_ids["causes"]
                and method_id in slot_valid_ids["methods"]
                and outcome_id in slot_valid_ids["outcomes"]
                and response_id in slot_valid_ids["responses"]
            ):
                continue
            macro_anchor_tag_id = str(item.get("macro_anchor_tag_id", "")).strip()
            if macro_anchor_tag_id and macro_anchor_tag_id not in valid_macro_ids:
                macro_anchor_tag_id = ""
            micro_motifs.append(
                {
                    "macro_anchor_tag_id": macro_anchor_tag_id,
                    "cause_id": cause_id,
                    "cause_name": slot_name_map["causes"].get(cause_id, cause_id),
                    "method_id": method_id,
                    "method_name": slot_name_map["methods"].get(method_id, method_id),
                    "outcome_id": outcome_id,
                    "outcome_name": slot_name_map["outcomes"].get(outcome_id, outcome_id),
                    "response_id": response_id,
                    "response_name": slot_name_map["responses"].get(response_id, response_id),
                    "evidence_episodes": _safe_episode_ids(item.get("evidence_episodes", [])),
                    "reason": str(item.get("reason", "")).strip(),
                }
            )
        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "episode_range": chunk["episode_range"],
                "chunk_summary": str(parsed.get("chunk_summary", "")).strip(),
                "ordered_macro_tags": macro_tags,
                "ordered_micro_motifs": micro_motifs,
                "raw_response": raw,
            }
        )
    save_cache(cache_path, expected_hash, {"chunk_dual_labels": results, "model_name": model_name})
    return results


def _motif_signature(motif: dict[str, Any]) -> tuple[str, str, str, str]:
    return (motif["cause_id"], motif["method_id"], motif["outcome_id"], motif["response_id"])


def _pattern_to_named_list(pattern: tuple[str, ...], name_maps: dict[str, dict[str, str]], kind: str) -> list[str]:
    if kind == "exact":
        cause_id, method_id, outcome_id, response_id = pattern
        return [
            name_maps["causes"].get(cause_id, cause_id),
            name_maps["methods"].get(method_id, method_id),
            name_maps["outcomes"].get(outcome_id, outcome_id),
            name_maps["responses"].get(response_id, response_id),
        ]
    if kind == "cmo":
        cause_id, method_id, outcome_id = pattern
        return [
            name_maps["causes"].get(cause_id, cause_id),
            name_maps["methods"].get(method_id, method_id),
            name_maps["outcomes"].get(outcome_id, outcome_id),
        ]
    method_id, outcome_id, response_id = pattern
    return [
        name_maps["methods"].get(method_id, method_id),
        name_maps["outcomes"].get(outcome_id, outcome_id),
        name_maps["responses"].get(response_id, response_id),
    ]


def causal_motif_reuse_statistics(
    motifs: list[dict[str, Any]], name_maps: dict[str, dict[str, str]]
) -> dict[str, Any]:
    exact = [_motif_signature(motif) for motif in motifs]
    cmo = [(motif["cause_id"], motif["method_id"], motif["outcome_id"]) for motif in motifs]
    mor = [(motif["method_id"], motif["outcome_id"], motif["response_id"]) for motif in motifs]
    exact_counts = Counter(exact)
    cmo_counts = Counter(cmo)
    mor_counts = Counter(mor)
    return {
        "motif_count": len(motifs),
        "exact_motif_reuse_rate": repeated_excess_ratio(exact),
        "cause_method_outcome_reuse_rate": repeated_excess_ratio(cmo),
        "method_outcome_response_reuse_rate": repeated_excess_ratio(mor),
        "top_exact_motifs": [
            {"pattern": list(pattern), "names": _pattern_to_named_list(pattern, name_maps, "exact"), "count": count}
            for pattern, count in exact_counts.most_common(10)
        ],
        "top_cause_method_outcome_patterns": [
            {"pattern": list(pattern), "names": _pattern_to_named_list(pattern, name_maps, "cmo"), "count": count}
            for pattern, count in cmo_counts.most_common(10)
        ],
        "top_method_outcome_response_patterns": [
            {"pattern": list(pattern), "names": _pattern_to_named_list(pattern, name_maps, "mor"), "count": count}
            for pattern, count in mor_counts.most_common(10)
        ],
    }


def conditional_transition_entropy(motifs: list[dict[str, Any]], name_maps: dict[str, dict[str, str]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for motif in motifs:
        grouped[(motif["method_id"], motif["outcome_id"])] [motif["response_id"]] += 1
    total = sum(sum(counter.values()) for counter in grouped.values())
    weighted_entropy = 0.0
    deterministic = 0
    details = []
    for (method_id, outcome_id), targets in sorted(grouped.items()):
        count = sum(targets.values())
        if len(targets) == 1:
            normalized = 0.0
            deterministic += count
        else:
            probabilities = [value / count for value in targets.values()]
            raw_entropy = -sum(probability * __import__("math").log(probability) for probability in probabilities)
            normalized = raw_entropy / __import__("math").log(len(targets))
        weighted_entropy += count * normalized
        details.append(
            {
                "method_id": method_id,
                "method_name": name_maps["methods"].get(method_id, method_id),
                "outcome_id": outcome_id,
                "outcome_name": name_maps["outcomes"].get(outcome_id, outcome_id),
                "count": count,
                "normalized_entropy": normalized,
                "responses": {
                    name_maps["responses"].get(response_id, response_id): value for response_id, value in targets.most_common()
                },
            }
        )
    return {
        "weighted_method_outcome_to_response_entropy": weighted_entropy / total if total else 0.0,
        "deterministic_method_outcome_to_response_ratio": deterministic / total if total else 0.0,
        "transition_count": total,
        "by_method_outcome": details,
    }


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text.strip())


def _char_ngrams(text: str, n: int = 2) -> set[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return set()
    if len(normalized) <= n:
        return {normalized}
    return {normalized[index : index + n] for index in range(len(normalized) - n + 1)}


def _text_similarity(left: str, right: str) -> float:
    left_grams = _char_ngrams(left)
    right_grams = _char_ngrams(right)
    if not left_grams and not right_grams:
        return 1.0
    if not left_grams or not right_grams:
        return 0.0
    overlap = len(left_grams & right_grams)
    union = len(left_grams | right_grams)
    return overlap / union if union else 0.0


def _group_similarity(entries: list[dict[str, str]], id_key: str, name_key: str) -> dict[str, Any]:
    grouped: dict[str, list[str]] = defaultdict(list)
    names: dict[str, str] = {}
    for entry in entries:
        grouped[entry[id_key]].append(entry.get("reason", ""))
        names[entry[id_key]] = entry.get(name_key, entry[id_key])
    details = []
    weighted_scores = []
    weights = []
    for item_id, reasons in grouped.items():
        if len(reasons) < 2:
            continue
        similarities = [_text_similarity(left, right) for left, right in combinations(reasons, 2)]
        score = mean(similarities)
        details.append(
            {
                "id": item_id,
                "name": names.get(item_id, item_id),
                "count": len(reasons),
                "mean_pairwise_similarity": score,
            }
        )
        weighted_scores.append(score * len(reasons))
        weights.append(len(reasons))
    details.sort(key=lambda item: (-item["mean_pairwise_similarity"], -item["count"], item["id"]))
    return {
        "weighted_mean_similarity": sum(weighted_scores) / sum(weights) if weights else 0.0,
        "groups_with_reuse": len(details),
        "details": details,
    }


def same_tag_variant_similarity(relabeled_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    macro_entries = []
    micro_entries = []
    for chunk in relabeled_chunks:
        macro_entries.extend(chunk.get("ordered_macro_tags", []))
        micro_entries.extend(chunk.get("ordered_micro_motifs", []))
    return {
        "macro_tags": _group_similarity(macro_entries, "tag_id", "tag_name"),
        "micro_methods": _group_similarity(micro_entries, "method_id", "method_name"),
    }


def _phase_for_index(index: int, total: int) -> str:
    if total <= 1:
        return PHASE_NAMES[0]
    ratio = index / total
    if ratio < 1 / 3:
        return PHASE_NAMES[0]
    if ratio < 2 / 3:
        return PHASE_NAMES[1]
    return PHASE_NAMES[2]


def motif_phase_coverage(chunks: list[dict[str, Any]], relabeled_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    phase_count = len(PHASE_NAMES)
    chunk_phase = {chunk["chunk_id"]: _phase_for_index(index, len(chunks)) for index, chunk in enumerate(chunks)}

    macro_occurrences: dict[str, set[str]] = defaultdict(set)
    macro_counts: Counter[str] = Counter()
    macro_names: dict[str, str] = {}
    method_occurrences: dict[str, set[str]] = defaultdict(set)
    method_counts: Counter[str] = Counter()
    method_names: dict[str, str] = {}

    for chunk in relabeled_chunks:
        phase = chunk_phase.get(chunk["chunk_id"], PHASE_NAMES[0])
        for tag in chunk.get("ordered_macro_tags", []):
            macro_occurrences[tag["tag_id"]].add(phase)
            macro_counts[tag["tag_id"]] += 1
            macro_names[tag["tag_id"]] = tag.get("tag_name", tag["tag_id"])
        for motif in chunk.get("ordered_micro_motifs", []):
            method_occurrences[motif["method_id"]].add(phase)
            method_counts[motif["method_id"]] += 1
            method_names[motif["method_id"]] = motif.get("method_name", motif["method_id"])

    def build(items: dict[str, set[str]], counts: Counter[str], names: dict[str, str]) -> dict[str, Any]:
        details = []
        weighted_scores = []
        weights = []
        for item_id, phases in items.items():
            if counts[item_id] < 2:
                continue
            coverage = len(phases) / phase_count
            details.append(
                {
                    "id": item_id,
                    "name": names.get(item_id, item_id),
                    "count": counts[item_id],
                    "phases": sorted(phases),
                    "phase_coverage_ratio": coverage,
                }
            )
            weighted_scores.append(coverage * counts[item_id])
            weights.append(counts[item_id])
        details.sort(key=lambda item: (-item["phase_coverage_ratio"], -item["count"], item["id"]))
        return {
            "weighted_phase_coverage_ratio": sum(weighted_scores) / sum(weights) if weights else 0.0,
            "details": details,
        }

    return {
        "phases": list(PHASE_NAMES),
        "macro_tags": build(macro_occurrences, macro_counts, macro_names),
        "micro_methods": build(method_occurrences, method_counts, method_names),
    }


def score_dual_layer_patterns(
    chunks: list[dict[str, Any]],
    global_macro_tags: list[dict[str, Any]],
    global_micro_taxonomy: dict[str, list[dict[str, Any]]],
    relabeled_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    macro_relabeled = [
        {
            "chunk_id": chunk["chunk_id"],
            "chunk_summary": chunk.get("chunk_summary", ""),
            "ordered_global_tags": chunk.get("ordered_macro_tags", []),
        }
        for chunk in relabeled_chunks
    ]
    macro_layer = score_chunk_sequences(chunks, global_macro_tags, macro_relabeled)
    macro_layer["metric"] = "dual_layer_macro_tag_homogeneity"
    macro_layer["method"] = "dual-layer macro narrative action tagbook with sequence repetition statistics"

    motifs = [motif for chunk in relabeled_chunks for motif in chunk.get("ordered_micro_motifs", [])]
    name_maps = {
        slot: {
            item[MICRO_SLOT_SPECS[slot]["id_key"]]: item[MICRO_SLOT_SPECS[slot]["name_key"]]
            for item in global_micro_taxonomy.get(slot, [])
        }
        for slot in MICRO_SLOT_SPECS
    }
    micro_layer = {
        "metric": "dual_layer_micro_causal_motif_homogeneity",
        "method": "clustered cause-method-outcome-response motifs with reuse, entropy, variant and phase coverage statistics",
        "taxonomy_sizes": {slot: len(global_micro_taxonomy.get(slot, [])) for slot in MICRO_SLOT_SPECS},
        "causal_motif_reuse": causal_motif_reuse_statistics(motifs, name_maps),
        "conditional_transition_entropy": conditional_transition_entropy(motifs, name_maps),
        "same_tag_variant_similarity": same_tag_variant_similarity(relabeled_chunks),
        "motif_phase_coverage": motif_phase_coverage(chunks, relabeled_chunks),
    }

    return {
        "metric": "dual_layer_trope_homogeneity",
        "method": "episode-chunk dual-layer extraction -> open-ended macro tagbook + clustered micro causal taxonomy -> relabeling -> macro/micro homogeneity statistics",
        "chunk_count": len(chunks),
        "macro_layer": macro_layer,
        "micro_layer": micro_layer,
        "global_macro_tags": global_macro_tags,
        "global_micro_taxonomy": global_micro_taxonomy,
        "chunk_assignments": [
            {
                "chunk_id": chunk["chunk_id"],
                "episode_range": chunk["episode_range"],
                "scene_count": chunk["scene_count"],
                "chunk_summary": relabeled["chunk_summary"],
                "ordered_macro_tags": relabeled["ordered_macro_tags"],
                "ordered_micro_motifs": relabeled["ordered_micro_motifs"],
            }
            for chunk, relabeled in zip(chunks, relabeled_chunks)
        ],
        "interpretation": {
            "macro_layer": "第一层更关注关系化中观标签的复用与序列趋同，适合看阶段性剧情弧线是否换皮重组。",
            "micro_layer": "第二层更关注 起因-手段-结果-反应 的微观因果链复用，适合看具体情节发动机是否反复使用。",
        },
    }


def evaluate(
    run_dir: str | Path,
    source: str,
    chunk_count: int,
    model_name: str,
    max_macro_tags: int,
    max_micro_motifs: int,
    refresh: bool = False,
) -> tuple[dict[str, Any], str]:
    episodes, resolved_source = load_episode_units(run_dir, source)
    chunks = chunk_episodes(episodes, chunk_count)
    result_dir = Path(run_dir).resolve() / "evaluation_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = source_hash(episodes, len(chunks), model_name)
    slug = model_slug(model_name)
    local_cache = result_dir / f"dual_layer_local_tags_{resolved_source}_{slug}.json"
    macro_cache = result_dir / f"dual_layer_macro_tagbook_{resolved_source}_{slug}.json"
    micro_cache = result_dir / f"dual_layer_micro_taxonomy_{resolved_source}_{slug}.json"
    relabel_cache = result_dir / f"dual_layer_relabels_{resolved_source}_{slug}.json"

    local_results = run_local_dual_extraction(
        chunks,
        local_cache,
        fingerprint,
        model_name,
        max_macro_tags,
        max_micro_motifs,
        refresh,
    )
    global_macro_tags = build_global_macro_tagbook(local_results, macro_cache, fingerprint, model_name, refresh)
    global_micro_taxonomy = build_global_micro_taxonomy(local_results, micro_cache, fingerprint, model_name, refresh)
    relabeled = run_dual_relabeling(
        chunks,
        global_macro_tags,
        global_micro_taxonomy,
        relabel_cache,
        fingerprint,
        model_name,
        max_macro_tags,
        max_micro_motifs,
        refresh,
    )
    result = score_dual_layer_patterns(chunks, global_macro_tags, global_micro_taxonomy, relabeled)
    result.update(
        {
            "source": resolved_source,
            "llm_backend": "project_venus_client",
            "model_name": model_name,
            "chunk_layout": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "episode_range": chunk["episode_range"],
                    "episode_ids": chunk["episode_ids"],
                    "scene_count": chunk["scene_count"],
                }
                for chunk in chunks
            ],
            "cache_files": {
                "local_tags": str(local_cache),
                "macro_tagbook": str(macro_cache),
                "micro_taxonomy": str(micro_cache),
                "dual_relabels": str(relabel_cache),
            },
        }
    )
    return result, resolved_source


def main() -> None:
    parser = argparse.ArgumentParser(description="计算双层套路化同质化评测（保留原有评测并新增微观因果链分析）")
    parser.add_argument("run_dir", help="生成结果目录，例如 output/60ep_0810_1946")
    parser.add_argument("--source", choices=["auto", "drama", "outline"], default="drama")
    parser.add_argument("--chunk-count", type=int, default=12, help="按集均匀切成多少段，默认12段")
    parser.add_argument("--model-name", default="gemini-2.5-pro", help="复用项目 VenusClient 的模型名")
    parser.add_argument("--max-macro-tags", type=int, default=6)
    parser.add_argument("--max-micro-motifs", type=int, default=8)
    parser.add_argument("--refresh", action="store_true", help="忽略缓存并重新请求LLM")
    parser.add_argument("--output", help="结果 JSON 路径；默认写入 run_dir/evaluation_results")
    args = parser.parse_args()

    result, source = evaluate(
        run_dir=args.run_dir,
        source=args.source,
        chunk_count=args.chunk_count,
        model_name=args.model_name,
        max_macro_tags=args.max_macro_tags,
        max_micro_motifs=args.max_micro_motifs,
        refresh=args.refresh,
    )
    output = (
        Path(args.output)
        if args.output
        else Path(args.run_dir).resolve() / "evaluation_results" / f"dual_layer_trope_homogeneity_{source}.json"
    )
    save_json(output, result)
    print(f"结果已写入：{output}")


if __name__ == "__main__":
    main()
