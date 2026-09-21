from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from evaluations.common.project_llm_trpc import query_json
except ImportError:
    from evaluations.common.project_llm import query_json

from evaluations.common.io import save_json

DEFAULT_OUTLINE_PATH = PROJECT_ROOT / "output/60ep_0810_1946/03_story_outline/outline.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output/stage_eventline_local_tree"


def load_outline(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"outline 文件不存在：{target}")
    with target.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("outline 文件内容必须是 JSON 对象")
    return payload


def extract_stage_payload(outline: dict[str, Any], stage_name: str) -> dict[str, Any]:
    framework = outline.get("framework", [])
    if not isinstance(framework, list):
        raise ValueError("outline.framework 必须是数组")

    for stage in framework:
        if not isinstance(stage, dict):
            continue
        if str(stage.get("stage", "")).strip() == stage_name:
            events = [
                str(item.get("event", "")).strip()
                for item in stage.get("event_list", [])
                if isinstance(item, dict) and str(item.get("event", "")).strip()
            ]
            if not events:
                raise ValueError(f"stage `{stage_name}` 中没有可用 event")
            return {
                "stage": stage_name,
                "seed_events": events,
            }
    raise ValueError(f"未找到 stage：{stage_name}")


class NodeIdFactory:
    def __init__(self) -> None:
        self._next = 1

    def new_id(self) -> str:
        value = self._next
        self._next += 1
        return f"n{value:04d}"


def sanitize_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def sanitize_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "是"}:
            return True
        if lowered in {"false", "no", "0", "否"}:
            return False
    return default


def build_candidate_messages(
    *,
    title: str,
    background: str,
    highlights: str,
    summary: str,
    stage_name: str,
    seed_events: list[str],
    fixed_prefix: list[str],
    current_segment: list[str],
    target_anchor: str,
    future_anchors: list[str],
    branch_factor: int,
    rollout_depth_remaining: int,
    max_events_to_anchor: int,
) -> list[dict[str, str]]:
    schema = {
        "candidates": [
            {
                "event": "一句话概括下一步事件",
                "progress_to_anchor": 72,
                "coherence": 8,
                "novelty": 7,
                "anchor_reached": False,
                "reason": "一句话说明这个候选为什么值得展开",
            }
        ]
    }

    prompt = f"""你现在扮演短剧剧情规划器，要为某个 stage 做“局部树搜索”的下一步扩展。

你的任务不是一次性写完整阶段，而是：
- 基于当前已经固定的事件前缀；
- 只提出“下一步可能发生的事件”候选；
- 每个候选只用一句话概括；
- 候选之间要有明显推进机制差异；
- 候选要尽量帮助剧情自然通往当前目标锚点。

背景信息：
- 标题：{title}
- 故事背景：{background}
- 卖点：{highlights}
- 摘要：{summary}
- 当前 stage：{stage_name}
- 本 stage 的粗粒度锚点：{seed_events}
- 已固定的完整前缀事件：{fixed_prefix}
- 当前这个锚点段内，已经暂时选中的事件：{current_segment}
- 当前要逼近的目标锚点：{target_anchor}
- 后续锚点（供你避免走偏）：{future_anchors}

约束：
1. 只输出接下来的 {branch_factor} 个候选“下一事件”。
2. 每个候选必须是一句话，不写对白，不写镜头，不写分集。
3. 候选之间不要只是同义改写，要在推进方式上拉开差异。
4. 允许候选是桥接、受挫、反打、关系推进、线索浮现、代价暴露中的任意一种。
5. `progress_to_anchor` 取 0-100，表示该事件发生后距离目标锚点有多近。
6. `coherence` 取 1-10，表示逻辑和人物动机是否自然。
7. `novelty` 取 1-10，表示相对常规爽文推进是否有变化，但不能为猎奇而猎奇。
8. `anchor_reached=true` 只在该事件已经可以视作抵达当前目标锚点时使用。
9. 如果当前很适合直接落到目标锚点，可以把某个候选写成目标锚点的实现版本。
10. 当前最多还允许为这一锚点再写 {max_events_to_anchor} 个事件；本次搜索还剩 {rollout_depth_remaining} 层前瞻预算。
11. 只输出 JSON，不要解释。

JSON 格式：{schema}
"""
    return [{"role": "user", "content": prompt}]


def parse_candidates(payload: Any, branch_factor: int) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if not isinstance(payload, dict):
        raise ValueError("模型返回的候选不是 JSON 对象")
    raw_candidates = payload.get("candidates", [])
    if not isinstance(raw_candidates, list):
        raise ValueError("candidates 必须是数组")

    normalized: list[dict[str, Any]] = []
    seen_events: set[str] = set()
    for item in raw_candidates:
        if not isinstance(item, dict):
            continue
        event = str(item.get("event", "")).strip()
        if not event or event in seen_events:
            continue
        seen_events.add(event)
        normalized.append(
            {
                "event": event,
                "progress_to_anchor": sanitize_int(item.get("progress_to_anchor"), default=50, minimum=0, maximum=100),
                "coherence": sanitize_int(item.get("coherence"), default=7, minimum=1, maximum=10),
                "novelty": sanitize_int(item.get("novelty"), default=6, minimum=1, maximum=10),
                "anchor_reached": sanitize_bool(item.get("anchor_reached"), default=False),
                "reason": str(item.get("reason", "")).strip(),
            }
        )
        if len(normalized) >= branch_factor:
            break
    if not normalized:
        raise ValueError("模型没有返回可用候选")
    return normalized


def compute_local_score(candidate: dict[str, Any], depth: int) -> float:
    coherence = float(candidate["coherence"])
    novelty = float(candidate["novelty"])
    progress = float(candidate["progress_to_anchor"]) / 10.0
    anchor_bonus = 7.0 if candidate["anchor_reached"] else 0.0
    depth_discount = math.pow(0.93, max(depth - 1, 0))
    return round((0.42 * coherence + 0.33 * novelty + 0.25 * progress + anchor_bonus) * depth_discount, 4)


def choose_children(candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    ranked = sorted(
        candidates,
        key=lambda item: (
            item["local_score"],
            item["progress_to_anchor"],
            item["novelty"],
            item["coherence"],
        ),
        reverse=True,
    )
    return ranked[: max(1, top_k)]


def build_tree_for_anchor(
    *,
    node_factory: NodeIdFactory,
    title: str,
    background: str,
    highlights: str,
    summary: str,
    stage_name: str,
    seed_events: list[str],
    fixed_prefix: list[str],
    committed_segment: list[str],
    target_anchor: str,
    future_anchors: list[str],
    branch_factor: int,
    keep_top_k: int,
    rollout_depth: int,
    max_events_to_anchor: int,
    model_name: str,
    temperature: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    current_best_path: list[dict[str, Any]] = []

    def expand(path_candidates: list[dict[str, Any]], depth: int) -> dict[str, Any]:
        nonlocal current_best_path

        if path_candidates:
            last = path_candidates[-1]
            if last["anchor_reached"] or depth > rollout_depth:
                total_score = round(sum(item["local_score"] for item in path_candidates), 4)
                if total_score > sum(item["local_score"] for item in current_best_path):
                    current_best_path = [dict(item) for item in path_candidates]
                return {
                    "node_id": last["node_id"],
                    "event": last["event"],
                    "depth": last["depth"],
                    "progress_to_anchor": last["progress_to_anchor"],
                    "coherence": last["coherence"],
                    "novelty": last["novelty"],
                    "anchor_reached": last["anchor_reached"],
                    "reason": last["reason"],
                    "local_score": last["local_score"],
                    "aggregate_score": total_score,
                    "children": [],
                }

        local_prefix = committed_segment + [item["event"] for item in path_candidates]
        remaining_budget = max_events_to_anchor - len(local_prefix)
        if remaining_budget <= 0:
            if path_candidates:
                total_score = round(sum(item["local_score"] for item in path_candidates), 4)
                if total_score > sum(item["local_score"] for item in current_best_path):
                    current_best_path = [dict(item) for item in path_candidates]
                last = path_candidates[-1]
                return {
                    "node_id": last["node_id"],
                    "event": last["event"],
                    "depth": last["depth"],
                    "progress_to_anchor": last["progress_to_anchor"],
                    "coherence": last["coherence"],
                    "novelty": last["novelty"],
                    "anchor_reached": last["anchor_reached"],
                    "reason": last["reason"],
                    "local_score": last["local_score"],
                    "aggregate_score": total_score,
                    "children": [],
                }
            return {
                "node_id": node_factory.new_id(),
                "event": "ROOT",
                "depth": 0,
                "progress_to_anchor": 0,
                "coherence": 0,
                "novelty": 0,
                "anchor_reached": False,
                "reason": "budget reached",
                "local_score": 0.0,
                "aggregate_score": 0.0,
                "children": [],
            }

        messages = build_candidate_messages(
            title=title,
            background=background,
            highlights=highlights,
            summary=summary,
            stage_name=stage_name,
            seed_events=seed_events,
            fixed_prefix=fixed_prefix,
            current_segment=local_prefix,
            target_anchor=target_anchor,
            future_anchors=future_anchors,
            branch_factor=branch_factor,
            rollout_depth_remaining=max(0, rollout_depth - depth + 1),
            max_events_to_anchor=remaining_budget,
        )
        parsed, raw_response = query_json(
            messages,
            model_name=model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        candidates = parse_candidates(parsed, branch_factor)
        enriched: list[dict[str, Any]] = []
        for candidate in candidates:
            node_depth = depth
            candidate["node_id"] = node_factory.new_id()
            candidate["depth"] = node_depth
            candidate["local_score"] = compute_local_score(candidate, node_depth)
            candidate["raw_response"] = raw_response
            enriched.append(candidate)

        selected = choose_children(enriched, keep_top_k)
        children = [expand(path_candidates + [child], depth + 1) for child in selected]

        if path_candidates:
            last = path_candidates[-1]
            aggregate_score = round(last["local_score"] + max((child["aggregate_score"] for child in children), default=0.0), 4)
            return {
                "node_id": last["node_id"],
                "event": last["event"],
                "depth": last["depth"],
                "progress_to_anchor": last["progress_to_anchor"],
                "coherence": last["coherence"],
                "novelty": last["novelty"],
                "anchor_reached": last["anchor_reached"],
                "reason": last["reason"],
                "local_score": last["local_score"],
                "aggregate_score": aggregate_score,
                "children": children,
            }

        return {
            "node_id": node_factory.new_id(),
            "event": "ROOT",
            "depth": 0,
            "progress_to_anchor": 0,
            "coherence": 0,
            "novelty": 0,
            "anchor_reached": False,
            "reason": "root",
            "local_score": 0.0,
            "aggregate_score": max((child["aggregate_score"] for child in children), default=0.0),
            "children": children,
        }

    tree = expand([], 1)
    best_path = [
        {
            "node_id": item["node_id"],
            "event": item["event"],
            "depth": item["depth"],
            "progress_to_anchor": item["progress_to_anchor"],
            "coherence": item["coherence"],
            "novelty": item["novelty"],
            "anchor_reached": item["anchor_reached"],
            "reason": item["reason"],
            "local_score": item["local_score"],
        }
        for item in current_best_path
    ]
    return {"tree": tree, "best_path": best_path}


def pick_frozen_prefix(best_path: list[dict[str, Any]], freeze_steps: int) -> tuple[list[dict[str, Any]], bool]:
    if not best_path:
        return [], False
    anchor_position = next((index for index, item in enumerate(best_path) if item.get("anchor_reached")), None)
    if anchor_position is not None:
        return best_path[: anchor_position + 1], True
    frozen = best_path[: max(1, min(freeze_steps, len(best_path)))]
    return frozen, False


def generate_stage_eventline_with_local_tree(
    *,
    outline_path: str | Path,
    stage_name: str,
    branch_factor: int,
    keep_top_k: int,
    rollout_depth: int,
    freeze_steps: int,
    max_events_to_anchor: int,
    model_name: str,
    temperature: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    outline = load_outline(outline_path)
    stage_payload = extract_stage_payload(outline, stage_name)
    seed_events = stage_payload["seed_events"]

    title = str(outline.get("title", "")).strip()
    background = str(outline.get("background", "")).strip()
    highlights = str(outline.get("highlights", "")).strip()
    summary = str(outline.get("summary", "")).strip()

    node_factory = NodeIdFactory()
    final_event_line: list[str] = []
    segment_results: list[dict[str, Any]] = []

    for anchor_index, target_anchor in enumerate(seed_events):
        segment_committed: list[str] = []
        rounds: list[dict[str, Any]] = []
        anchor_reached = False
        force_appended_anchor = False

        for round_index in range(1, max_events_to_anchor + 1):
            tree_result = build_tree_for_anchor(
                node_factory=node_factory,
                title=title,
                background=background,
                highlights=highlights,
                summary=summary,
                stage_name=stage_name,
                seed_events=seed_events,
                fixed_prefix=final_event_line,
                committed_segment=segment_committed,
                target_anchor=target_anchor,
                future_anchors=seed_events[anchor_index + 1 :],
                branch_factor=branch_factor,
                keep_top_k=keep_top_k,
                rollout_depth=rollout_depth,
                max_events_to_anchor=max_events_to_anchor,
                model_name=model_name,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            best_path = tree_result["best_path"]
            frozen_nodes, reached_now = pick_frozen_prefix(best_path, freeze_steps)
            frozen_events = [item["event"] for item in frozen_nodes]
            if not frozen_events:
                break

            segment_committed.extend(frozen_events)
            final_event_line.extend(frozen_events)
            rounds.append(
                {
                    "round": round_index,
                    "tree": tree_result["tree"],
                    "best_path": best_path,
                    "frozen_nodes": frozen_nodes,
                    "frozen_events": frozen_events,
                    "anchor_reached_in_this_round": reached_now,
                }
            )

            if reached_now:
                anchor_reached = True
                break
            if len(segment_committed) >= max_events_to_anchor:
                break

        if not anchor_reached:
            final_event_line.append(target_anchor)
            segment_committed.append(target_anchor)
            force_appended_anchor = True

        segment_results.append(
            {
                "anchor_index": anchor_index,
                "target_anchor": target_anchor,
                "rounds": rounds,
                "segment_committed": segment_committed,
                "anchor_reached_by_search": anchor_reached,
                "force_appended_anchor": force_appended_anchor,
            }
        )

    return {
        "meta": {
            "mode": "local_tree_search_experiment",
            "outline_path": str(Path(outline_path).expanduser().resolve()),
            "stage": stage_name,
            "title": title,
            "branch_factor": branch_factor,
            "keep_top_k": keep_top_k,
            "rollout_depth": rollout_depth,
            "freeze_steps": freeze_steps,
            "max_events_to_anchor": max_events_to_anchor,
            "model_name": model_name,
            "temperature": temperature,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "source_stage": stage_payload,
        "final_event_line": final_event_line,
        "segments": segment_results,
    }


def build_output_path(output_dir: str | Path, stage_name: str) -> Path:
    safe_stage = "".join(ch if ch.isalnum() else "_" for ch in stage_name).strip("_") or "stage"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_dir).expanduser().resolve() / f"{safe_stage}_local_tree_{timestamp}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对指定 stage 执行局部树搜索式事件线生成实验")
    parser.add_argument("--outline-path", default=str(DEFAULT_OUTLINE_PATH), help="story outline JSON 路径")
    parser.add_argument("--stage", default="发展", help="要扩写的 stage 名称，默认‘发展’")
    parser.add_argument("--branch-factor", type=int, default=5, help="每个节点生成多少个下一事件候选")
    parser.add_argument("--keep-top-k", type=int, default=2, help="每层保留多少个候选继续 rollout")
    parser.add_argument("--rollout-depth", type=int, default=3, help="每轮局部树向前 rollout 几层")
    parser.add_argument("--freeze-steps", type=int, default=1, help="每轮固定最佳路径前几步；若提前到锚点则固定到锚点")
    parser.add_argument("--max-events-to-anchor", type=int, default=6, help="每个粗粒度锚点前最多补充多少个事件")
    parser.add_argument("--model-name", default="gemini-2.5-pro", help="模型名")
    parser.add_argument("--temperature", type=float, default=0.9, help="采样温度")
    parser.add_argument("--max-new-tokens", type=int, default=8192, help="最大生成 token 数")
    parser.add_argument("--output", help="输出 JSON 路径；不传则自动落到 output/stage_eventline_local_tree")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_stage_eventline_with_local_tree(
        outline_path=args.outline_path,
        stage_name=args.stage,
        branch_factor=args.branch_factor,
        keep_top_k=args.keep_top_k,
        rollout_depth=args.rollout_depth,
        freeze_steps=args.freeze_steps,
        max_events_to_anchor=args.max_events_to_anchor,
        model_name=args.model_name,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    output_path = Path(args.output).expanduser().resolve() if args.output else build_output_path(DEFAULT_OUTPUT_DIR, args.stage)
    save_json(output_path, result)
    print(f"已完成 `{args.stage}` stage 的局部树搜索实验")
    print(f"最终事件数：{len(result['final_event_line'])}")
    print(f"结果已写入：{output_path}")


if __name__ == "__main__":
    main()
