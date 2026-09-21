from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluations.common.io import load_scenes, save_json
from evaluations.common.math_utils import (
    lcs_similarity,
    levenshtein_similarity,
    mean,
    normalized_entropy,
)
from evaluations.common.project_llm import query_json
from evaluations.narrative_function_sequence.evaluate import ngrams, repeated_excess_ratio, transition_statistics


def model_slug(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", model_name).strip("-") or "model"


def load_episode_units(run_dir: str | Path, source: str = "auto") -> tuple[list[dict[str, Any]], str]:
    scenes, resolved_source = load_scenes(run_dir, source)
    grouped: dict[int, list[Any]] = defaultdict(list)
    for scene in scenes:
        grouped[scene.episode_id].append(scene)
    episodes = []
    for episode_id in sorted(grouped):
        parts = []
        for scene in sorted(grouped[episode_id], key=lambda item: item.scene_number):
            heading = f"{scene.scene_id} {scene.heading}".strip()
            parts.append(f"{heading}\n{scene.text}".strip())
        text = "\n\n---\n\n".join(part for part in parts if part.strip())
        episodes.append(
            {
                "episode_id": episode_id,
                "scene_count": len(grouped[episode_id]),
                "text": text,
            }
        )
    if len(episodes) < 2:
        raise RuntimeError(f"{resolved_source} 只提取到 {len(episodes)} 集，至少需要2集才能分段评测")
    return episodes, resolved_source


def chunk_episodes(episodes: list[dict[str, Any]], chunk_count: int) -> list[dict[str, Any]]:
    if not episodes:
        return []
    count = max(1, min(chunk_count, len(episodes)))
    base = len(episodes) // count
    remainder = len(episodes) % count
    chunks = []
    cursor = 0
    for index in range(count):
        size = base + (1 if index < remainder else 0)
        current = episodes[cursor : cursor + size]
        cursor += size
        chunk_id = f"C{index + 1:02d}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "chunk_index": index + 1,
                "episode_ids": [item["episode_id"] for item in current],
                "episode_range": [current[0]["episode_id"], current[-1]["episode_id"]],
                "scene_count": sum(item["scene_count"] for item in current),
                "text": "\n\n".join(
                    f"【第{item['episode_id']}集】\n{item['text']}" for item in current if item["text"].strip()
                ),
            }
        )
    return chunks


def source_hash(episodes: list[dict[str, Any]], chunk_count: int, model_name: str) -> str:
    payload = {
        "chunk_count": chunk_count,
        "model_name": model_name,
        "episodes": [{"episode_id": item["episode_id"], "text": item["text"]} for item in episodes],
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def load_cache(path: Path, expected_hash: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("source_hash") != expected_hash:
        return None
    return payload


def save_cache(path: Path, expected_hash: str, payload: dict[str, Any]) -> None:
    result = {"source_hash": expected_hash}
    result.update(payload)
    save_json(path, result)


def prompt_local_chunk_tags(chunk: dict[str, Any], max_tags: int) -> list[dict[str, str]]:
    schema = {
        "chunk_summary": "一句话概括这一段主线推进",
        "ordered_tags": [
            {
                "label": "4到12个汉字的粗粒度套路标签",
                "definition": "一句话说明该标签代表什么模式",
                "evidence_episodes": [chunk["episode_range"][0]],
                "reason": "为什么这一段适合这个标签",
            }
        ],
    }
    prompt = f"""你是短剧套路分析员。现在只分析一个剧本片段，不评价文笔。
目标：提炼这一片段内部按阅读顺序出现的粗粒度套路标签，用于后续全剧内部同质化分析。
要求：
1. 标签必须偏笼统，描述中观叙事机制，不要写具体人名地名，不要写过细事件。
2. 标签示例风格：身份压制制造危机、嘴炮反制翻盘、借围观舆论施压、条件交易建立联盟、新危机钩子收尾。
3. 输出 3 到 {max_tags} 个标签，按阅读顺序排列。
4. 同一标签若在该段内再次明显出现，可以重复一次；若只是轻微呼应，不要重复。
5. 每个 evidence_episodes 只填写相关集号整数列表。
6. 只输出 JSON 对象，不要额外解释。
JSON格式示例：{json.dumps(schema, ensure_ascii=False)}
片段编号：{chunk['chunk_id']}
覆盖集数：{chunk['episode_range'][0]}-{chunk['episode_range'][1]}
片段正文：
{chunk['text'][:24000]}
"""
    return [{"role": "user", "content": prompt}]


def prompt_global_tagbook(local_results: list[dict[str, Any]], min_tags: int, max_tags: int) -> list[dict[str, str]]:
    compact = []
    for item in local_results:
        compact.append(
            {
                "chunk_id": item["chunk_id"],
                "episode_range": item["episode_range"],
                "chunk_summary": item["chunk_summary"],
                "ordered_tags": [
                    {
                        "label": tag["label"],
                        "definition": tag["definition"],
                    }
                    for tag in item["ordered_tags"]
                ],
            }
        )
    schema = {
        "global_tags": [
            {
                "tag_id": "T01",
                "name": "标签名",
                "definition": "标签定义",
                "includes": ["包含哪些局部叫法"],
                "excludes": ["容易混淆但不应并入的模式"],
                "merged_local_tags": ["被并入的原始局部标签"],
            }
        ]
    }
    prompt = f"""你是短剧标签体系设计员。下面给你的是同一部剧多个片段各自提炼出的局部套路标签。
任务：将这些局部标签归并成一套全剧统一标签表，用于后续每个片段的统一回标和同质化统计。
要求：
1. 生成 {min_tags} 到 {max_tags} 个全局标签。
2. 标签要足够宽，能跨多个片段复用；但不要宽到只剩“冲突”“反转”这种无信息词。
3. 禁止保留人名、地名、具体道具；必须抽象成叙事机制。
4. 若两个局部标签本质相同，只保留一个更稳的叫法。
5. includes / merged_local_tags 写被并入的局部说法；excludes 写边界。
6. tag_id 必须连续编号 T01, T02 ...
7. 只输出 JSON 对象。
JSON格式示例：{json.dumps(schema, ensure_ascii=False)}
局部标签输入：
{json.dumps(compact, ensure_ascii=False, indent=2)}
"""
    return [{"role": "user", "content": prompt}]


def prompt_relabel_chunk(chunk: dict[str, Any], global_tags: list[dict[str, Any]], max_tags: int) -> list[dict[str, str]]:
    schema = {
        "chunk_summary": "一句话概括",
        "ordered_global_tags": [
            {
                "tag_id": "T01",
                "tag_name": "标签名",
                "evidence_episodes": [chunk["episode_range"][0]],
                "reason": "为什么这里出现该标签",
            }
        ],
    }
    prompt = f"""你是短剧套路回标员。现在要用固定的全局标签表，对一个剧本片段重新打标签。
要求：
1. 只能从给定 tag_id 中选择，不得自创标签。
2. 输出 3 到 {max_tags} 个标签，按片段阅读顺序排列。
3. 若同一全局标签在该段中明显重复出现，可以重复写一次；否则不要重复。
4. 标签仍然是粗粒度套路机制，不是逐场细拆。
5. evidence_episodes 只填写相关集号整数列表。
6. 只输出 JSON 对象。
全局标签表：
{json.dumps(global_tags, ensure_ascii=False, indent=2)}
JSON格式示例：{json.dumps(schema, ensure_ascii=False)}
片段编号：{chunk['chunk_id']}
覆盖集数：{chunk['episode_range'][0]}-{chunk['episode_range'][1]}
片段正文：
{chunk['text'][:24000]}
"""
    return [{"role": "user", "content": prompt}]


def run_local_tagging(
    chunks: list[dict[str, Any]],
    cache_path: Path,
    expected_hash: str,
    model_name: str,
    max_tags: int,
    refresh: bool,
) -> list[dict[str, Any]]:
    cached = None if refresh else load_cache(cache_path, expected_hash)
    if cached:
        return cached["chunk_local_tags"]
    results = []
    for chunk in chunks:
        parsed, raw = query_json(prompt_local_chunk_tags(chunk, max_tags), model_name=model_name, max_new_tokens=8192)
        ordered_tags = parsed.get("ordered_tags", [])[:max_tags]
        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "episode_range": chunk["episode_range"],
                "chunk_summary": str(parsed.get("chunk_summary", "")).strip(),
                "ordered_tags": [
                    {
                        "label": str(tag.get("label", "")).strip(),
                        "definition": str(tag.get("definition", "")).strip(),
                        "evidence_episodes": [int(ep) for ep in tag.get("evidence_episodes", []) if str(ep).isdigit()],
                        "reason": str(tag.get("reason", "")).strip(),
                    }
                    for tag in ordered_tags
                    if str(tag.get("label", "")).strip()
                ],
                "raw_response": raw,
            }
        )
    save_cache(cache_path, expected_hash, {"chunk_local_tags": results, "model_name": model_name})
    return results


def build_global_tagbook(
    local_results: list[dict[str, Any]],
    cache_path: Path,
    expected_hash: str,
    model_name: str,
    min_tags: int,
    max_tags: int,
    refresh: bool,
) -> list[dict[str, Any]]:
    cached = None if refresh else load_cache(cache_path, expected_hash)
    if cached:
        return cached["global_tags"]
    parsed, raw = query_json(
        prompt_global_tagbook(local_results, min_tags, max_tags),
        model_name=model_name,
        max_new_tokens=12288,
    )
    tags = []
    for index, item in enumerate(parsed.get("global_tags", []), start=1):
        tag_id = str(item.get("tag_id", f"T{index:02d}")).strip() or f"T{index:02d}"
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
    save_cache(cache_path, expected_hash, {"global_tags": tags, "raw_response": raw, "model_name": model_name})
    return tags


def run_global_relabeling(
    chunks: list[dict[str, Any]],
    global_tags: list[dict[str, Any]],
    cache_path: Path,
    expected_hash: str,
    model_name: str,
    max_tags: int,
    refresh: bool,
) -> list[dict[str, Any]]:
    cached = None if refresh else load_cache(cache_path, expected_hash)
    if cached:
        return cached["chunk_global_labels"]
    results = []
    valid_tag_ids = {item["tag_id"] for item in global_tags}
    name_map = {item["tag_id"]: item["name"] for item in global_tags}
    for chunk in chunks:
        parsed, raw = query_json(prompt_relabel_chunk(chunk, global_tags, max_tags), model_name=model_name, max_new_tokens=8192)
        ordered = []
        for item in parsed.get("ordered_global_tags", [])[:max_tags]:
            tag_id = str(item.get("tag_id", "")).strip()
            if tag_id not in valid_tag_ids:
                continue
            ordered.append(
                {
                    "tag_id": tag_id,
                    "tag_name": name_map.get(tag_id, str(item.get("tag_name", "")).strip()),
                    "evidence_episodes": [int(ep) for ep in item.get("evidence_episodes", []) if str(ep).isdigit()],
                    "reason": str(item.get("reason", "")).strip(),
                }
            )
        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "episode_range": chunk["episode_range"],
                "chunk_summary": str(parsed.get("chunk_summary", "")).strip(),
                "ordered_global_tags": ordered,
                "raw_response": raw,
            }
        )
    save_cache(cache_path, expected_hash, {"chunk_global_labels": results, "model_name": model_name})
    return results


def top_patterns(sequences: list[list[str]], order: int) -> dict[str, Any]:
    all_grams = [gram for sequence in sequences for gram in ngrams(sequence, order)]
    counts = Counter(all_grams)
    return {
        "total_occurrences": len(all_grams),
        "unique_count": len(counts),
        "repeated_excess_ratio": repeated_excess_ratio(all_grams),
        "top_patterns": [
            {"pattern": list(pattern), "count": count}
            for pattern, count in counts.most_common(10)
        ],
    }


def nearest_sequence_pairs(chunk_ids: list[str], sequences: list[list[str]]) -> tuple[list[dict[str, Any]], float, float]:
    details = []
    nearest_edit = []
    nearest_lcs = []
    for left in range(len(sequences)):
        candidates = []
        for right in range(len(sequences)):
            if left == right:
                continue
            edit_score = levenshtein_similarity(sequences[left], sequences[right])
            lcs_score = lcs_similarity(sequences[left], sequences[right])
            candidates.append((max(edit_score, lcs_score), edit_score, lcs_score, right))
        best = max(candidates, default=(0.0, 0.0, 0.0, left))
        nearest_edit.append(best[1])
        nearest_lcs.append(best[2])
        details.append(
            {
                "chunk_id": chunk_ids[left],
                "nearest_chunk_id": chunk_ids[best[3]],
                "edit_similarity": best[1],
                "lcs_similarity": best[2],
            }
        )
    return details, mean(nearest_edit), mean(nearest_lcs)


def score_chunk_sequences(
    chunks: list[dict[str, Any]],
    global_tags: list[dict[str, Any]],
    relabeled_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    sequences = [[item["tag_id"] for item in chunk["ordered_global_tags"]] for chunk in relabeled_chunks]
    chunk_ids = [chunk["chunk_id"] for chunk in relabeled_chunks]
    signatures = [tuple(sequence) for sequence in sequences]
    exact_counts = Counter(signatures)
    all_tags = [tag for sequence in sequences for tag in sequence]
    tag_counts = Counter(all_tags)
    details, mean_edit, mean_lcs = nearest_sequence_pairs(chunk_ids, sequences)
    return {
        "metric": "chunk_global_tag_homogeneity",
        "method": "episode-chunk local tagging -> global tagbook -> chunk relabeling -> sequence repetition statistics",
        "chunk_count": len(chunks),
        "global_tag_count": len(global_tags),
        "global_tag_entropy": normalized_entropy(all_tags),
        "global_tag_reuse_rate": repeated_excess_ratio(all_tags),
        "top_global_tags": [
            {
                "tag_id": tag["tag_id"],
                "tag_name": tag["name"],
                "count": tag_counts.get(tag["tag_id"], 0),
            }
            for tag in sorted(global_tags, key=lambda item: (-tag_counts.get(item["tag_id"], 0), item["tag_id"]))
        ],
        "exact_chunk_sequence_reuse_rate": repeated_excess_ratio(signatures),
        "exact_chunk_sequence_types": [
            {"sequence": list(signature), "count": count}
            for signature, count in exact_counts.most_common()
        ],
        "chunk_tag_ngram_reuse": {
            "2": top_patterns(sequences, 2),
            "3": top_patterns(sequences, 3),
        },
        "mean_nearest_edit_similarity": mean_edit,
        "mean_nearest_lcs_similarity": mean_lcs,
        "nearest_chunk_pairs": details,
        "transition_statistics": transition_statistics(sequences),
        "global_tags": global_tags,
        "chunk_assignments": [
            {
                "chunk_id": chunk["chunk_id"],
                "episode_range": chunk["episode_range"],
                "scene_count": chunk["scene_count"],
                "chunk_summary": relabeled["chunk_summary"],
                "ordered_global_tags": relabeled["ordered_global_tags"],
            }
            for chunk, relabeled in zip(chunks, relabeled_chunks)
        ],
        "interpretation": {
            "reuse": "global_tag_reuse_rate、exact_chunk_sequence_reuse_rate、n-gram重复率越高，说明剧内片段复用相同套路越明显。",
            "predictability": "transition_statistics 中转移熵越低、最近邻相似度越高，说明片段之间的套路路径越趋同。",
        },
    }


def evaluate(
    run_dir: str | Path,
    source: str,
    chunk_count: int,
    model_name: str,
    max_tags_per_chunk: int,
    global_tag_min: int,
    global_tag_max: int,
    refresh: bool = False,
) -> tuple[dict[str, Any], str]:
    episodes, resolved_source = load_episode_units(run_dir, source)
    chunks = chunk_episodes(episodes, chunk_count)
    result_dir = Path(run_dir).resolve() / "evaluation_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = source_hash(episodes, len(chunks), model_name)
    slug = model_slug(model_name)
    local_cache = result_dir / f"chunk_local_tags_{resolved_source}_{slug}.json"
    tagbook_cache = result_dir / f"chunk_global_tagbook_{resolved_source}_{slug}.json"
    relabel_cache = result_dir / f"chunk_global_relabels_{resolved_source}_{slug}.json"
    local_results = run_local_tagging(chunks, local_cache, fingerprint, model_name, max_tags_per_chunk, refresh)
    global_tags = build_global_tagbook(
        local_results,
        tagbook_cache,
        fingerprint,
        model_name,
        global_tag_min,
        global_tag_max,
        refresh,
    )
    relabeled = run_global_relabeling(
        chunks,
        global_tags,
        relabel_cache,
        fingerprint,
        model_name,
        max_tags_per_chunk,
        refresh,
    )
    result = score_chunk_sequences(chunks, global_tags, relabeled)
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
                "global_tagbook": str(tagbook_cache),
                "global_relabels": str(relabel_cache),
            },
        }
    )
    return result, resolved_source


def main() -> None:
    parser = argparse.ArgumentParser(description="计算基于LLM分段标签归并的剧内套路化同质化评测")
    parser.add_argument("run_dir", help="生成结果目录，例如 output/60ep_0810_1946")
    parser.add_argument("--source", choices=["auto", "drama", "outline"], default="drama")
    parser.add_argument("--chunk-count", type=int, default=12, help="按集均匀切成多少段，默认12段")
    parser.add_argument("--model-name", default="gemini-2.5-pro", help="复用项目 VenusClient 的模型名")
    parser.add_argument("--max-tags-per-chunk", type=int, default=6)
    parser.add_argument("--global-tag-min", type=int, default=10)
    parser.add_argument("--global-tag-max", type=int, default=18)
    parser.add_argument("--refresh", action="store_true", help="忽略缓存并重新请求LLM")
    parser.add_argument("--output", help="结果 JSON 路径；默认写入 run_dir/evaluation_results")
    args = parser.parse_args()

    result, source = evaluate(
        run_dir=args.run_dir,
        source=args.source,
        chunk_count=args.chunk_count,
        model_name=args.model_name,
        max_tags_per_chunk=args.max_tags_per_chunk,
        global_tag_min=args.global_tag_min,
        global_tag_max=args.global_tag_max,
        refresh=args.refresh,
    )
    output = (
        Path(args.output)
        if args.output
        else Path(args.run_dir).resolve() / "evaluation_results" / f"chunk_tag_homogeneity_{source}.json"
    )
    save_json(output, result)
    print(f"结果已写入：{output}")


if __name__ == "__main__":
    main()
