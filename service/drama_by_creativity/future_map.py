"""在完整集大纲生成后构建 Future Map 与分集贡献映射。"""

import json
from typing import Any, Dict, List, Optional, Set, Tuple

import jinja2
from .local_llm import LLM
from drama_local.runtime import logger

from .llm_inference import llm_inference_json
from .prompts.future_map import (
    GENERATE_EPISODE_CONTRIBUTIONS_PROMPT,
    GENERATE_FUTURE_MAP_PROMPT,
)
from .realtime_output import save_realtime
from .utils import get_model_config


def _build_llm_service(prompt: str) -> LLM:
    config = get_model_config("future_map_model")
    return LLM(
        model=config["model"],
        system_prompt=config["system_prompt"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        top_k=config["top_k"],
        max_length=config["max_length"],
        max_new_tokens=config["max_new_tokens"],
        template=jinja2.Template(prompt),
    )


def _json_text(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _normalize_future_map(raw: Any) -> Tuple[Optional[Dict[str, Any]], str]:
    if not isinstance(raw, dict) or not isinstance(raw.get("nodes"), list):
        return None, "返回结果缺少 nodes 数组"

    nodes_by_id: Dict[str, Dict[str, Any]] = {}
    ordered_ids: List[str] = []
    for raw_node in raw["nodes"]:
        if not isinstance(raw_node, dict):
            continue
        node_id = str(raw_node.get("node_id", "")).strip()
        if not node_id or node_id in nodes_by_id:
            continue
        parent_id = raw_node.get("parent_id")
        if parent_id is not None:
            parent_id = str(parent_id).strip() or None
        node = {
            "node_id": node_id,
            "parent_id": parent_id,
            "depth": raw_node.get("depth"),
            "goal": str(raw_node.get("goal", "")).strip(),
            "requirement": str(raw_node.get("requirement", "")).strip(),
            "evidence": str(raw_node.get("evidence", "")).strip(),
        }
        if not node["goal"]:
            continue
        nodes_by_id[node_id] = node
        ordered_ids.append(node_id)

    if "FM001" not in nodes_by_id:
        return None, "Future Map 缺少根节点 FM001"

    root = nodes_by_id["FM001"]
    root["parent_id"] = None
    root["depth"] = 0

    visiting: Set[str] = set()
    resolved_depths: Dict[str, int] = {"FM001": 0}

    def resolve_depth(node_id: str) -> Optional[int]:
        if node_id in resolved_depths:
            return resolved_depths[node_id]
        if node_id in visiting:
            return None
        node = nodes_by_id[node_id]
        parent_id = node["parent_id"]
        if not parent_id or parent_id not in nodes_by_id or parent_id == node_id:
            return None
        visiting.add(node_id)
        parent_depth = resolve_depth(parent_id)
        visiting.discard(node_id)
        if parent_depth is None:
            return None
        resolved_depths[node_id] = parent_depth + 1
        return parent_depth + 1

    valid_nodes: List[Dict[str, Any]] = []
    for node_id in ordered_ids:
        depth = resolve_depth(node_id)
        if depth is None:
            continue
        node = nodes_by_id[node_id]
        node["depth"] = depth
        valid_nodes.append(node)

    if len(valid_nodes) < 2:
        return None, "Future Map 未形成有效的需求分解树"

    return {
        "map_type": "backward_requirement_tree",
        "root_id": "FM001",
        "nodes": valid_nodes,
    }, ""


def _normalize_episode_contributions(
    raw: Any,
    episode_outline: List[Dict[str, Any]],
    valid_node_ids: Set[str],
) -> Tuple[Optional[Dict[str, Any]], str]:
    if not isinstance(raw, dict) or not isinstance(raw.get("episodes"), list):
        return None, "返回结果缺少 episodes 数组"

    expected_ids = [episode.get("episode_id") for episode in episode_outline]
    if any(not isinstance(episode_id, int) for episode_id in expected_ids):
        return None, "集大纲包含无效 episode_id"

    raw_by_episode: Dict[int, Dict[str, Any]] = {}
    for item in raw["episodes"]:
        if isinstance(item, dict) and isinstance(item.get("episode_id"), int):
            raw_by_episode[item["episode_id"]] = item

    if set(raw_by_episode) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(raw_by_episode))
        extra = sorted(set(raw_by_episode) - set(expected_ids))
        return None, f"分集贡献映射集数不完整，缺失={missing}，多余={extra}"

    allowed_types = {"setup", "advance", "turn", "complete", "payoff"}
    episodes: List[Dict[str, Any]] = []
    for episode_id in sorted(expected_ids):
        item = raw_by_episode[episode_id]
        raw_contributions = item.get("contributions", [])
        if not isinstance(raw_contributions, list):
            raw_contributions = []
        contributions: List[Dict[str, str]] = []
        seen_node_ids: Set[str] = set()
        primary_used = False
        for raw_contribution in raw_contributions:
            if not isinstance(raw_contribution, dict):
                continue
            node_id = str(raw_contribution.get("node_id", "")).strip()
            contribution_type = str(raw_contribution.get("contribution_type", "")).strip()
            contribution = str(raw_contribution.get("contribution", "")).strip()
            if (
                node_id not in valid_node_ids
                or node_id in seen_node_ids
                or contribution_type not in allowed_types
                or not contribution
            ):
                continue
            importance = str(raw_contribution.get("importance", "secondary")).strip()
            if importance not in {"primary", "secondary"}:
                importance = "secondary"
            if importance == "primary":
                if primary_used:
                    importance = "secondary"
                else:
                    primary_used = True
            seen_node_ids.add(node_id)
            contributions.append({
                "node_id": node_id,
                "contribution_type": contribution_type,
                "importance": importance,
                "contribution": contribution,
            })
        episodes.append({"episode_id": episode_id, "contributions": contributions})

    return {"episodes": episodes}, ""


async def generate_future_map_artifacts(
    ctx,
    episode_outline: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """依次生成 Future Map 和分集贡献映射；失败不阻断原集大纲流程。"""
    if not episode_outline:
        logger.warning_context(ctx, "集大纲为空，跳过 Future Map 生成。")
        return None

    outline_text = _json_text(episode_outline)
    try:
        logger.info_context(ctx, "完整集大纲已生成，开始构建 Future Map。")
        raw_future_map = await llm_inference_json(
            params={"EpisodeOutline": outline_text},
            llm_service=_build_llm_service(GENERATE_FUTURE_MAP_PROMPT),
            task_name="生成 Future Map",
            ctx=ctx,
        )
        future_map, error = _normalize_future_map(raw_future_map)
        if future_map is None:
            logger.error_context(ctx, f"Future Map 校验失败：{error}")
            return None
        save_realtime(["04_future_map", "future_map.json"], future_map)

        logger.info_context(ctx, "Future Map 已生成，开始构建分集贡献映射。")
        raw_contributions = await llm_inference_json(
            params={
                "EpisodeOutline": outline_text,
                "FutureMap": _json_text(future_map),
            },
            llm_service=_build_llm_service(GENERATE_EPISODE_CONTRIBUTIONS_PROMPT),
            task_name="生成分集 Future Map 贡献映射",
            ctx=ctx,
        )
        contributions, error = _normalize_episode_contributions(
            raw_contributions,
            episode_outline,
            {node["node_id"] for node in future_map["nodes"]},
        )
        if contributions is None:
            logger.error_context(ctx, f"分集贡献映射校验失败：{error}")
            return {"future_map": future_map, "episode_contributions": None}
        save_realtime(["04_future_map", "episode_contributions.json"], contributions)
        logger.info_context(ctx, "Future Map 与分集贡献映射生成完成。")
        return {
            "future_map": future_map,
            "episode_contributions": contributions,
        }
    except Exception as exc:
        logger.error_context(ctx, f"Future Map 生成失败，不影响集大纲返回：{exc}")
        return None
