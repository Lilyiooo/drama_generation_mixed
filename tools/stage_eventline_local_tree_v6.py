from __future__ import annotations

import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from stage_eventline_local_tree_v2 import (
    DEFAULT_OUTLINE_PATH,
    NodeIdFactory,
    choose_top_candidates_for_expansion,
    collect_previous_stage_events,
    compute_candidate_expansion_score,
    create_node_payload,
    extract_stage_payload,
    load_outline,
    parse_candidates,
    sanitize_int,
    score_candidates_for_expansion_once,
)


STEP_SCORING_FEWSHOT = """
Few-shot 示例 1：
前序已发生情节：
- 女主靠撒泼和舆论表演夺回将军府控制权。
目标锚点是第一笔讨债。
当前分支：
- 团队准备去找礼部尚书追债。
同一父节点下的候选下一事件：
候选 1：女主继续在尚书府门口哭嚎骂街，逼对方拿钱息事宁人。
候选 2：团队先摸清尚书最爱名画、最怕赝品丑闻，再借一场假拍卖逼他露怯。
评分参考：
- 候选 1：logic = 7, drive = 7, novelty = 3, appeal = 5
- 说明：能推动讨债，但机制仍是"公开闹场施压"，只是换个对象重复旧套路。这属于复用已固定的完整前缀事件中的叙事套路（包含前序 stage 与本 stage 已完成部分），novelty必须给低分。
- 候选 2：logic = 8, drive = 8, novelty = 8, appeal = 8
- 说明：既顺着讨债目标推进，又从硬闹切换到设局智取，较明显跳脱前序套路，novelty给高分。

Few-shot 示例 2：
前序已发生情节：
- 男女主刚建立互信，继母开始反扑。
- 宁国公仍是后续才该正式浮出的更大反派。
当前分支：
- 继母准备利用现有局面继续打压女主。
同一父节点下的候选下一事件：
候选 1：继母直接当众说出宁国公就是幕后真凶。
候选 2：继母借账目问题封掉女主刚收回的铺面，逼她先自救。
评分参考：
- 候选 1：logic = 4, drive = 5, novelty = 2, appeal = 4
- 说明：看似刺激，但它粗暴消耗后续悬念，属于跳步，不应高分。
- 候选 2：logic = 8, drive = 8, novelty = 6, appeal = 7
- 说明：反扑方式和当前局势一致，也给后续更大阴谋保留了空间。
""".strip()


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
    remaining_budget: int,
) -> list[dict[str, str]]:
    schema = {
        "candidates": [
            {
                "event": "一句话概括下一步事件",
                "progress_to_anchor": 72,
                "anchor_reached": False,
                "reason": "一句话说明这个候选为什么值得继续展开",
            }
        ]
    }

    prompt = f"""你现在扮演短剧剧情规划器，要为某个 stage 做"局部树搜索"的下一步扩展。

你的任务不是一次性写完整阶段，而是：
- 基于当前已经固定的前序剧情；
- 只提出"下一步可能发生的事件"候选；
- 每个候选只用一句话概括；
- 候选之间要有明显推进机制差异；
- 候选要尽量帮助剧情自然通往当前目标锚点；
- 你只负责提出候选，不负责给这些候选打分。

背景信息：
- 标题：{title}
- 故事背景：{background}
- 卖点：{highlights}
- 摘要：{summary}
- 当前 stage：{stage_name}
- 本 stage 的粗粒度锚点：{seed_events}
- 已固定的完整前缀事件（包含前序 stage 与本 stage 已完成部分）：{fixed_prefix}
- 当前这条分支已经走到的事件链：{current_segment}
- 当前要逼近的目标锚点：{target_anchor}
- 后续锚点（供你避免走偏和提前消耗悬念）：{future_anchors}

约束：
1. 只输出接下来的 {branch_factor} 个候选"下一事件"。
2. 每个候选必须是一句话，不写对白，不写镜头，不写分集。
3. 候选之间不要只是同义改写，要在推进方式上拉开差异。
4. `progress_to_anchor` 取 0-100，表示该事件发生后距离目标锚点有多近。
5. `anchor_reached=true` 只在该事件已经可以视作抵达当前目标锚点时使用。
6. 尽量不要复用已固定的完整前缀事件中的叙事套路（包含前序 stage 与本 stage 已完成部分）。
7. 不要为了显得新就跳步，不要提前把后续锚点的核心悬念直接消耗掉。
8. 当前这条分支最多还允许再写 {remaining_budget} 个事件，务必控制在预算内自然抵达目标锚点。（这只是最多事件预算，可以早停）
9. 只输出 JSON，不要解释。

JSON 格式：{schema}
"""
    return [{"role": "user", "content": prompt}]


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output/stage_eventline_local_tree_v6"

LOCAL_CHAIN_SCORING_FEWSHOT = """
Few-shot 示例 1：
前序已发生情节：
- 女主靠舆论表演和撒泼夺回将军府控制权。
- 当前目标锚点是"礼部尚书第一次被迫正面应对讨债"。
候选链 A（未到锚点）：
- 团队先查到尚书府最近在秘密处理一幅赝品名画。
- 女主故意放风说拍卖行要验画，逼得尚书府下人先行灭火。
候选链 B（已到锚点）：
- 女主继续去尚书府门口哭嚎骂街。
- 尚书嫌丢人，出来见她。
评分参考：
- 链 A：coherence = 8, novelty = 8, target_fit = 7, continuation_value = 9
- 说明：虽然还没直接碰到锚点，但它明显把局面推向"尚书不得不出面"，而且推进机制从公开闹场切到设局智取，没有重复之前的叙事套路，适合作为下一轮的强承接点。
- 链 B：coherence = 7, novelty = 3, target_fit = 8, continuation_value = 5
- 说明：它确实接触到了目标锚点，但推进方式仍是重复既有套路，新意和后续展开空间都偏弱，即机制仍是"公开闹场施压"，只是换个对象重复旧套路。这属于复用已固定的完整前缀事件中的叙事套路（包含前序 stage 与本 stage 已完成部分），novelty必须给低分。

Few-shot 示例 2：
前序已发生情节：
- 男女主刚追回第一笔债，继母准备反扑。
- 当前目标锚点是"继母发动一轮有效打压"。
候选链 C（未到锚点）：
- 继母先收买账房，悄悄做空账本。
- 女主发现店里银钱对不上，只能先自查漏洞。
候选链 D（未到锚点）：
- 继母突然公开宣布宁国公才是真凶。
- 满朝震动。
评分参考：
- 链 C：coherence = 8, novelty = 6, target_fit = 7, continuation_value = 8
- 说明：尚未真正碰到目标锚点，但已为继母的实质打压完成铺垫，下一轮很容易自然进入反扑。
- 链 D：coherence = 3, novelty = 2, target_fit = 2, continuation_value = 2
- 说明：看似刺激，实则严重跳步，还提前消耗后续大反派悬念，不是好的局部链条。
""".strip()


class ProgressLogger:
    def __init__(self) -> None:
        self._call_index = 0
        self._lock = threading.Lock()

    def log(self, stage: str, anchor_index: int, total_anchors: int, phase: str, detail: str) -> None:
        with self._lock:
            self._call_index += 1
            call_index = self._call_index
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(
                f"[{timestamp}] LLM调用#{call_index} | stage={stage} | 段={anchor_index + 1}/{total_anchors} | {phase} | {detail}",
                flush=True,
            )

    def info(self, message: str) -> None:
        with self._lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}", flush=True)

    def log_segment_start(self, stage: str, anchor_index: int, total_anchors: int, target_anchor: str) -> None:
        self.info(
            f"开始处理锚点段 {anchor_index + 1}/{total_anchors} | stage={stage} | target={target_anchor}"
        )

    def log_local_chain_start(
        self,
        stage: str,
        anchor_index: int,
        total_anchors: int,
        round_index: int,
        local_max_depth: int,
        prefix_events: list[str],
    ) -> None:
        self.info(
            f"开始局部链轮次 {round_index} | stage={stage} | 段={anchor_index + 1}/{total_anchors} | depth上限={local_max_depth} | 当前已提交事件数={len(prefix_events)}"
        )

    def log_local_chain_finish(
        self,
        stage: str,
        anchor_index: int,
        total_anchors: int,
        round_index: int,
        committed_events: list[str],
        anchor_reached: bool,
    ) -> None:
        self.info(
            f"完成局部链轮次 {round_index} | stage={stage} | 段={anchor_index + 1}/{total_anchors} | 本轮提交事件数={len(committed_events)} | reached={anchor_reached}"
        )

    def log_segment_finish(self, stage: str, anchor_index: int, total_anchors: int, committed_events: list[str]) -> None:
        self.info(
            f"完成锚点段 {anchor_index + 1}/{total_anchors} | stage={stage} | 提交事件数={len(committed_events)}"
        )

    def log_write(self, output_path: Path, completed_segments: int, total_segments: int) -> None:
        self.info(
            f"已落盘 | progress={completed_segments}/{total_segments} | file={output_path}"
        )


def abstract_micro_motifs_once(
    *,
    events: list[str],
    source_label: str,
    model_name: str,
    max_new_tokens: int,
    progress_logger: ProgressLogger,
    stage_name: str,
    anchor_index: int,
    total_anchors: int,
) -> list[dict[str, str]]:
    """从一段已提交事件中，用 LLM 提取 起因->手段->结果->反应 的微观因果链套路。"""
    if not events:
        return []

    schema = {
        "motifs": [
            {"cause": "高位打压", "method": "公开施压", "outcome": "施压失败", "response": "被迫变招"},
        ]
    }
    events_text = "\n".join(f"- {event}" for event in events)
    prompt = f"""你是短剧套路分析员。下面是一段已经写好的剧情事件链，请提炼其中涉及的"叙事套路"。

每个套路用 起因 -> 手段 -> 结果 -> 反应 四个槽位表示，每个槽位用 2-8 字的抽象词组概括。
要求：
1. 不要出现具体人名、地点、道具。
2. 可以相对笼统一些，但需要展现出大致的情节发展方式。
3. 事件链中可能存在多条叙事套路链条，需要逐一罗列。
4. 只输出 JSON 对象，不要解释。

剧情事件链（{source_label}）：
{events_text}

JSON格式：{schema}
"""
    progress_logger.log(
        stage=stage_name,
        anchor_index=anchor_index,
        total_anchors=total_anchors,
        phase="套路提取",
        detail=f"source={source_label} | events={len(events)}",
    )
    parsed, _raw = query_json(
        [{"role": "user", "content": prompt}],
        model_name=model_name,
        temperature=0.2,
        max_new_tokens=max_new_tokens,
    )
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else {}
    motifs: list[dict[str, str]] = []
    if isinstance(parsed, dict):
        for item in parsed.get("motifs", []):
            if not isinstance(item, dict):
                continue
            motif = {
                "cause": str(item.get("cause", "")).strip(),
                "method": str(item.get("method", "")).strip(),
                "outcome": str(item.get("outcome", "")).strip(),
                "response": str(item.get("response", "")).strip(),
            }
            if all(motif.values()):
                motifs.append(motif)
    return motifs


def format_trope_bank(trope_bank: list[dict[str, str]]) -> str:
    """把套路 bank 格式化成评分 prompt 里的可读文本。"""
    if not trope_bank:
        return "（暂无，这是第一次提交）"
    lines = []
    for index, motif in enumerate(trope_bank, 1):
        lines.append(
            f"{index}. 起因:{motif['cause']} -> 手段:{motif['method']} -> 结果:{motif['outcome']} -> 反应:{motif['response']}"
        )
    return "\n".join(lines)


def build_local_chain_score_messages(
    *,
    title: str,
    background: str,
    highlights: str,
    summary: str,
    stage_name: str,
    seed_events: list[str],
    fixed_prefix: list[str],
    target_anchor: str,
    future_anchors: list[str],
    terminal_chains: list[dict[str, Any]],
    trope_bank: list[dict[str, str]],
) -> list[dict[str, str]]:
    chain_lines: list[str] = []
    for index, chain in enumerate(terminal_chains, start=1):
        chain_lines.append(f"候选链 {index}：")
        for event in chain["events"]:
            chain_lines.append(f"- {event}")
        chain_lines.append(f"- anchor_reached: {chain['anchor_reached']}")
        chain_lines.append(f"- final_progress_to_anchor: {chain['final_progress_to_anchor']}")
        chain_lines.append(f"- terminal_reason: {chain['terminal_reason']}")

    chains_text = "\n".join(chain_lines)
    trope_bank_text = format_trope_bank(trope_bank)
    schema = {
        "scores": [
            {
                "chain_index": 1,
                "coherence": 8,
                "novelty": 7,
                "target_fit": 8,
                "continuation_value": 8,
                "reason": "两三句话说明为什么这条局部链适合或不适合在当前轮次被提交。",
            }
        ]
    }

    prompt = f"""你现在是短剧剧情局部树搜索中的独立评委，负责给"当前轮次产生的多条局部候选链"分别打分，并决定哪一条最值得在本轮被提交。

注意：
- 有些候选链已经抵达当前目标锚点，有些还没有。
- 你的任务不是强行偏向"已经到锚点"的链，而是评估哪条链作为"这一轮应提交的最佳局部链"最合适。
- 如果一条链虽然还没到锚点，但已经明显逼近目标、推进逻辑顺、且为下一轮续链打开了更强局面，它也可以拿高分。
- 但如果一条链原地热闹、机制重复、人物动机不顺、强行跳步，或者提前消费 future_anchors 的核心悬念，就不能高分。
- novelty 不是猎奇，而是相对于"已用叙事套路 bank"里已记录的因果套路，是否出现了真实的新推进机制。
- continuation_value 特别关注：这条链在结尾处是否形成了一个适合下一轮继续扩展的状态。

{STEP_SCORING_FEWSHOT}

{LOCAL_CHAIN_SCORING_FEWSHOT}

待评估的故事信息：
- 标题：{title}
- 故事背景：{background}
- 卖点：{highlights}
- 摘要：{summary}
- 当前 stage：{stage_name}
- 本 stage 的粗粒度锚点：{seed_events}
- 已固定的完整前缀事件（包含前序 stage 与本 stage 已完成部分）：{fixed_prefix}
- 当前要逼近的目标锚点：{target_anchor}
- 后续锚点（必须避免提前消耗其核心悬念）：{future_anchors}
- 已用过的叙事套路 bank（起因->手段->结果->反应，供 novelty 去重）：
{trope_bank_text}

待评估候选链如下：
{chains_text}

评分维度：
1. `coherence`：1-10，评价这条局部链的人物动机、因果衔接、事件逻辑是否自然。
2. `novelty`：1-10，评价它相对"已用叙事套路 bank"中记录的因果链（起因->手段->结果->反应），是否引入了新的因果推进模式，而非复用已固定的完整前缀事件中的叙事套路（包含前序 stage 与本 stage 已完成部分）；若本条链的核心套路与 bank 中某条高度雷同或相似度较高（换皮复用），novelty 必须给低分。
3. `target_fit`：1-10，若已到锚点，评价到达方式是否自然；若未到锚点，评价它是否实质性逼近目标而不是原地打转。
4. `continuation_value`：1-10，评价它是否把局面推到了一个适合下一轮续链的状态。
5. `reason`：2-3 句话，明确说明它为什么顺或不顺、为什么值得或不值得提交，尤其说明有没有重复套路、跳步或提前消费后续悬念。

只输出 JSON，不要解释。
JSON 格式：{schema}
"""
    return [{"role": "user", "content": prompt}]


def parse_local_chain_scores(payload: Any, expected_count: int) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if not isinstance(payload, dict):
        raise ValueError("局部链评分返回不是 JSON 对象")
    raw_scores = payload.get("scores", [])
    if not isinstance(raw_scores, list):
        raise ValueError("scores 必须是数组")

    score_map: dict[int, dict[str, Any]] = {}
    for item in raw_scores:
        if not isinstance(item, dict):
            continue
        chain_index = sanitize_int(item.get("chain_index"), default=-1, minimum=-1, maximum=10_000)
        if chain_index < 1 or chain_index > expected_count:
            continue
        score_map[chain_index - 1] = {
            "coherence": sanitize_int(item.get("coherence"), default=7, minimum=1, maximum=10),
            "novelty": sanitize_int(item.get("novelty"), default=6, minimum=1, maximum=10),
            "target_fit": sanitize_int(item.get("target_fit"), default=7, minimum=1, maximum=10),
            "continuation_value": sanitize_int(item.get("continuation_value"), default=7, minimum=1, maximum=10),
            "reason": str(item.get("reason", "")).strip(),
        }

    normalized: list[dict[str, Any]] = []
    for index in range(expected_count):
        normalized.append(
            score_map.get(
                index,
                {
                    "coherence": 7,
                    "novelty": 6,
                    "target_fit": 7,
                    "continuation_value": 7,
                    "reason": "评分模型未返回该链分数，使用默认值。",
                },
            )
        )
    return normalized


def compute_local_chain_score(score_payload: dict[str, Any], anchor_reached: bool) -> float:
    coherence = float(score_payload["coherence"])
    novelty = float(score_payload["novelty"])
    target_fit = float(score_payload["target_fit"])
    continuation_value = float(score_payload["continuation_value"])
    reached_bonus = 0.35 if anchor_reached else 0.0
    return round(0.30 * coherence + 0.30 * novelty + 0.30 * target_fit + 0.10 * continuation_value + reached_bonus, 4)


def score_local_chains_once(
    *,
    title: str,
    background: str,
    highlights: str,
    summary: str,
    stage_name: str,
    seed_events: list[str],
    fixed_prefix: list[str],
    target_anchor: str,
    future_anchors: list[str],
    terminal_chains: list[dict[str, Any]],
    trope_bank: list[dict[str, str]],
    model_name: str,
    max_new_tokens: int,
    progress_logger: ProgressLogger,
    anchor_index: int,
    total_anchors: int,
    round_index: int,
) -> tuple[list[dict[str, Any]], str]:
    messages = build_local_chain_score_messages(
        title=title,
        background=background,
        highlights=highlights,
        summary=summary,
        stage_name=stage_name,
        seed_events=seed_events,
        fixed_prefix=fixed_prefix,
        target_anchor=target_anchor,
        future_anchors=future_anchors,
        terminal_chains=terminal_chains,
        trope_bank=trope_bank,
    )
    progress_logger.log(
        stage=stage_name,
        anchor_index=anchor_index,
        total_anchors=total_anchors,
        phase="局部链评分",
        detail=f"round={round_index} | candidate_chains={len(terminal_chains)}",
    )
    parsed, raw_response = query_json(
        messages,
        model_name=model_name,
        temperature=0.2,
        max_new_tokens=max_new_tokens,
    )
    scores = parse_local_chain_scores(parsed, len(terminal_chains))
    return scores, raw_response


def choose_best_local_chain(scored_chains: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not scored_chains:
        return None
    ranked = sorted(
        scored_chains,
        key=lambda item: (
            item["score"],
            1 if item["anchor_reached"] else 0,
            item["target_fit"],
            item["continuation_value"],
            item["final_progress_to_anchor"],
            item["coherence"],
            item["novelty"],
        ),
        reverse=True,
    )
    return ranked[0]


def build_output_path(output_dir: str | Path, stage_name: str) -> Path:
    safe_stage = "".join(ch if ch.isalnum() else "_" for ch in stage_name).strip("_") or "stage"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_dir).expanduser().resolve() / f"{safe_stage}_local_tree_v6_{timestamp}.json"


def generate_candidates_for_state_once(
    *,
    position: int,
    total_states: int,
    state: dict[str, Any],
    title: str,
    background: str,
    highlights: str,
    summary: str,
    stage_name: str,
    seed_events: list[str],
    fixed_prefix: list[str],
    target_anchor: str,
    future_anchors: list[str],
    branch_factor: int,
    local_max_depth: int,
    depth: int,
    model_name: str,
    temperature: float,
    max_new_tokens: int,
    progress_logger: ProgressLogger,
    anchor_index: int,
    total_anchors: int,
    round_index: int,
    anchor_remaining_budget: int,
) -> dict[str, Any]:
    path = state["path"]
    current_segment = [item["event"] for item in path]
    remaining_budget = local_max_depth - len(current_segment)
    prompt_budget = max(0, anchor_remaining_budget - len(current_segment))
    if remaining_budget <= 0:
        return {
            "position": position,
            "state": state,
            "current_segment": current_segment,
            "remaining_budget": remaining_budget,
            "candidates": [],
            "raw_generation_response": "",
            "skipped": True,
        }

    messages = build_candidate_messages(
        title=title,
        background=background,
        highlights=highlights,
        summary=summary,
        stage_name=stage_name,
        seed_events=seed_events,
        fixed_prefix=fixed_prefix,
        current_segment=current_segment,
        target_anchor=target_anchor,
        future_anchors=future_anchors,
        branch_factor=branch_factor,
        remaining_budget=prompt_budget,
    )
    progress_logger.log(
        stage=stage_name,
        anchor_index=anchor_index,
        total_anchors=total_anchors,
        phase="候选生成",
        detail=(
            f"round={round_index} | depth={depth} | leaf={position + 1}/{total_states} "
            f"| parent_path_len={len(current_segment)} | remaining_budget={remaining_budget}"
        ),
    )
    parsed, raw_generation_response = query_json(
        messages,
        model_name=model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    candidates = parse_candidates(parsed, branch_factor)
    return {
        "position": position,
        "state": state,
        "current_segment": current_segment,
        "remaining_budget": remaining_budget,
        "candidates": candidates,
        "raw_generation_response": raw_generation_response,
        "skipped": False,
    }


def score_candidates_for_state_once(
    *,
    generation_result: dict[str, Any],
    title: str,
    background: str,
    highlights: str,
    summary: str,
    stage_name: str,
    seed_events: list[str],
    fixed_prefix: list[str],
    target_anchor: str,
    future_anchors: list[str],
    model_name: str,
    max_new_tokens: int,
    progress_logger: ProgressLogger,
    anchor_index: int,
    total_anchors: int,
    depth: int,
) -> dict[str, Any]:
    if generation_result["skipped"]:
        return {
            **generation_result,
            "step_scores": [],
            "raw_step_scoring_response": "",
        }

    step_scores, raw_step_scoring_response = score_candidates_for_expansion_once(
        title=title,
        background=background,
        highlights=highlights,
        summary=summary,
        stage_name=stage_name,
        seed_events=seed_events,
        fixed_prefix=fixed_prefix,
        current_segment=generation_result["current_segment"],
        target_anchor=target_anchor,
        future_anchors=future_anchors,
        candidates=generation_result["candidates"],
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        progress_logger=progress_logger,
        anchor_index=anchor_index,
        total_anchors=total_anchors,
        depth=depth,
    )
    return {
        **generation_result,
        "step_scores": step_scores,
        "raw_step_scoring_response": raw_step_scoring_response,
    }


def build_tree_with_parallel_generation(
    *,
    node_factory: NodeIdFactory,
    title: str,
    background: str,
    highlights: str,
    summary: str,
    stage_name: str,
    seed_events: list[str],
    fixed_prefix: list[str],
    target_anchor: str,
    future_anchors: list[str],
    branch_factor: int,
    keep_top_k: int,
    local_max_depth: int,
    model_name: str,
    temperature: float,
    max_new_tokens: int,
    parallel_workers: int,
    progress_logger: ProgressLogger,
    anchor_index: int,
    total_anchors: int,
    round_index: int,
    anchor_remaining_budget: int,
) -> dict[str, Any]:
    root = {
        "node_id": node_factory.new_id(),
        "event": "ROOT",
        "depth": 0,
        "progress_to_anchor": 0,
        "anchor_reached": False,
        "generation_reason": "root",
        "selected_for_expansion": True,
        "logic": None,
        "drive": None,
        "novelty": None,
        "appeal": None,
        "evaluation_reason": "",
        "expansion_score": None,
        "children": [],
    }

    active_states: list[dict[str, Any]] = [{"path": [], "tree_node": root}]
    completed_paths: list[dict[str, Any]] = []
    incomplete_paths: list[dict[str, Any]] = []
    layer_summaries: list[dict[str, Any]] = []

    for depth in range(1, local_max_depth + 1):
        if not active_states:
            break

        layer_info: dict[str, Any] = {"depth": depth, "expanded_paths": []}
        generation_inputs: list[dict[str, Any]] = []
        for state in active_states:
            current_segment = [item["event"] for item in state["path"]]
            remaining_budget = local_max_depth - len(current_segment)
            if remaining_budget <= 0:
                incomplete_paths.append(
                    {
                        "events": current_segment,
                        "path": state["path"],
                        "terminal_reason": "budget_exhausted_before_generation",
                    }
                )
                continue
            generation_inputs.append(state)

        if not generation_inputs:
            break

        max_workers = max(1, min(parallel_workers, len(generation_inputs)))
        generation_results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_position = {
                executor.submit(
                    generate_candidates_for_state_once,
                    position=position,
                    total_states=len(generation_inputs),
                    state=state,
                    title=title,
                    background=background,
                    highlights=highlights,
                    summary=summary,
                    stage_name=stage_name,
                    seed_events=seed_events,
                    fixed_prefix=fixed_prefix,
                    target_anchor=target_anchor,
                    future_anchors=future_anchors,
                    branch_factor=branch_factor,
                    local_max_depth=local_max_depth,
                    depth=depth,
                    model_name=model_name,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    progress_logger=progress_logger,
                    anchor_index=anchor_index,
                    total_anchors=total_anchors,
                    round_index=round_index,
                    anchor_remaining_budget=anchor_remaining_budget,
                ): position
                for position, state in enumerate(generation_inputs)
            }
            for future in as_completed(future_to_position):
                generation_results.append(future.result())

        generation_results.sort(key=lambda item: item["position"])

        scoring_inputs = [result for result in generation_results if not result["skipped"]]
        scoring_results_by_position: dict[int, dict[str, Any]] = {}
        if scoring_inputs:
            scoring_workers = max(1, min(parallel_workers, len(scoring_inputs)))
            with ThreadPoolExecutor(max_workers=scoring_workers) as executor:
                future_to_position = {
                    executor.submit(
                        score_candidates_for_state_once,
                        generation_result=result,
                        title=title,
                        background=background,
                        highlights=highlights,
                        summary=summary,
                        stage_name=stage_name,
                        seed_events=seed_events,
                        fixed_prefix=fixed_prefix,
                        target_anchor=target_anchor,
                        future_anchors=future_anchors,
                        model_name=model_name,
                        max_new_tokens=max_new_tokens,
                        progress_logger=progress_logger,
                        anchor_index=anchor_index,
                        total_anchors=total_anchors,
                        depth=depth,
                    ): result["position"]
                    for result in scoring_inputs
                }
                for future in as_completed(future_to_position):
                    scored_result = future.result()
                    scoring_results_by_position[scored_result["position"]] = scored_result

        next_active_states: list[dict[str, Any]] = []

        for result in generation_results:
            state = result["state"]
            path = state["path"]
            parent_node = state["tree_node"]
            current_segment = result["current_segment"]
            if result["skipped"]:
                incomplete_paths.append(
                    {
                        "events": current_segment,
                        "path": path,
                        "terminal_reason": "budget_exhausted_before_generation",
                    }
                )
                continue

            scored_result = scoring_results_by_position[result["position"]]
            candidates = scored_result["candidates"]
            step_scores = scored_result["step_scores"]
            raw_step_scoring_response = scored_result["raw_step_scoring_response"]

            scored_candidates: list[dict[str, Any]] = []
            for index, candidate in enumerate(candidates):
                score_payload = step_scores[index]
                scored_candidates.append(
                    {
                        **candidate,
                        "logic": score_payload["logic"],
                        "drive": score_payload["drive"],
                        "novelty": score_payload["novelty"],
                        "appeal": score_payload["appeal"],
                        "evaluation_reason": score_payload["reason"],
                        "expansion_score": compute_candidate_expansion_score(score_payload),
                    }
                )

            selected_candidates = choose_top_candidates_for_expansion(scored_candidates, keep_top_k)
            selected_events = {candidate["event"] for candidate in selected_candidates}

            expansion_record = {
                "parent_path": current_segment,
                "generation_raw_response": result["raw_generation_response"],
                "step_score_raw_response": raw_step_scoring_response,
                "generated_candidates": [],
            }

            for candidate in scored_candidates:
                node = create_node_payload(
                    node_id=node_factory.new_id(),
                    event=candidate["event"],
                    depth=depth,
                    progress_to_anchor=candidate["progress_to_anchor"],
                    anchor_reached=candidate["anchor_reached"],
                    reason=candidate.get("reason", ""),
                    selected_for_expansion=candidate["event"] in selected_events,
                    logic=candidate["logic"],
                    drive=candidate["drive"],
                    novelty=candidate["novelty"],
                    appeal=candidate["appeal"],
                    evaluation_reason=candidate["evaluation_reason"],
                    expansion_score=candidate["expansion_score"],
                )
                parent_node["children"].append(node)
                candidate_state = {
                    "node_id": node["node_id"],
                    "event": node["event"],
                    "depth": depth,
                    "progress_to_anchor": node["progress_to_anchor"],
                    "anchor_reached": node["anchor_reached"],
                    "generation_reason": node["generation_reason"],
                    "logic": node["logic"],
                    "drive": node["drive"],
                    "novelty": node["novelty"],
                    "appeal": node["appeal"],
                    "evaluation_reason": node["evaluation_reason"],
                    "expansion_score": node["expansion_score"],
                    "selected_for_expansion": node["selected_for_expansion"],
                }
                expansion_record["generated_candidates"].append(candidate_state)

                if not node["selected_for_expansion"]:
                    continue

                new_path = path + [candidate_state]
                if node["anchor_reached"]:
                    completed_paths.append(
                        {
                            "events": [item["event"] for item in new_path],
                            "path": new_path,
                            "terminal_reason": "anchor_reached",
                        }
                    )
                else:
                    next_active_states.append({"path": new_path, "tree_node": node})

            layer_info["expanded_paths"].append(expansion_record)

        layer_summaries.append(layer_info)
        active_states = next_active_states

    for state in active_states:
        incomplete_paths.append(
            {
                "events": [item["event"] for item in state["path"]],
                "path": state["path"],
                "terminal_reason": "local_max_depth_reached_without_anchor",
            }
        )

    return {
        "tree": root,
        "layer_summaries": layer_summaries,
        "completed_paths": completed_paths,
        "incomplete_paths": incomplete_paths,
    }


def collect_terminal_chains(search_result: dict[str, Any]) -> list[dict[str, Any]]:
    terminal_chains: list[dict[str, Any]] = []
    for item in search_result.get("completed_paths", []):
        events = item.get("events", [])
        path = item.get("path", [])
        if not events or not path:
            continue
        terminal_chains.append(
            {
                "events": events,
                "path": path,
                "terminal_reason": str(item.get("terminal_reason", "anchor_reached")),
                "anchor_reached": True,
                "final_progress_to_anchor": sanitize_int(path[-1].get("progress_to_anchor"), default=100, minimum=0, maximum=100),
            }
        )

    for item in search_result.get("incomplete_paths", []):
        events = item.get("events", [])
        path = item.get("path", [])
        if not events or not path:
            continue
        terminal_chains.append(
            {
                "events": events,
                "path": path,
                "terminal_reason": str(item.get("terminal_reason", "incomplete")),
                "anchor_reached": False,
                "final_progress_to_anchor": sanitize_int(path[-1].get("progress_to_anchor"), default=0, minimum=0, maximum=100),
            }
        )
    return terminal_chains


def assemble_generation_result(
    *,
    outline_path: str | Path,
    stage_name: str,
    title: str,
    branch_factor: int,
    keep_top_k: int,
    local_max_depth: int,
    max_events_to_anchor: int,
    max_local_chains_per_anchor: int,
    parallel_workers: int,
    model_name: str,
    temperature: float,
    generated_at: str,
    stage_payload: dict[str, Any],
    previous_stage_events: list[str],
    final_event_line: list[str],
    segment_results: list[dict[str, Any]],
    trope_bank: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "meta": {
            "mode": "local_tree_search_experiment_v6",
            "outline_path": str(Path(outline_path).expanduser().resolve()),
            "stage": stage_name,
            "title": title,
            "branch_factor": branch_factor,
            "keep_top_k": keep_top_k,
            "local_max_depth": local_max_depth,
            "max_events_to_anchor": max_events_to_anchor,
            "max_local_chains_per_anchor": max_local_chains_per_anchor,
            "parallel_workers": parallel_workers,
            "model_name": model_name,
            "temperature": temperature,
            "generated_at": generated_at,
            "completed_segments": len(segment_results),
            "total_segments": len(stage_payload.get("seed_events", [])),
            "is_finished": len(segment_results) == len(stage_payload.get("seed_events", [])),
        },
        "source_stage": stage_payload,
        "previous_stage_events": previous_stage_events,
        "final_event_line": final_event_line,
        "trope_bank": trope_bank,
        "segments": segment_results,
    }


def generate_stage_eventline_with_local_tree_v6(
    *,
    outline_path: str | Path,
    stage_name: str,
    branch_factor: int,
    keep_top_k: int,
    local_max_depth: int,
    max_events_to_anchor: int,
    max_local_chains_per_anchor: int,
    parallel_workers: int,
    model_name: str,
    temperature: float,
    max_new_tokens: int,
    output_path: str | Path | None = None,
    progress_logger: ProgressLogger | None = None,
) -> dict[str, Any]:
    outline = load_outline(outline_path)
    stage_payload = extract_stage_payload(outline, stage_name)
    seed_events = stage_payload["seed_events"]
    previous_stage_events = collect_previous_stage_events(outline, stage_name)

    title = str(outline.get("title", "")).strip()
    background = str(outline.get("background", "")).strip()
    highlights = str(outline.get("highlights", "")).strip()
    summary = str(outline.get("summary", "")).strip()
    generated_at = datetime.now().isoformat(timespec="seconds")
    resolved_output_path = Path(output_path).expanduser().resolve() if output_path else None

    progress_logger = progress_logger or ProgressLogger()
    node_factory = NodeIdFactory()
    final_event_line: list[str] = []
    segment_results: list[dict[str, Any]] = []

    # v6：初始化叙事套路 bank；先对历史阶段（前序 stage）事件建立套路
    trope_bank: list[dict[str, str]] = []
    if previous_stage_events:
        progress_logger.info(f"对历史阶段（前序 stage）提取叙事套路，共 {len(previous_stage_events)} 个事件...")
        historical_motifs = abstract_micro_motifs_once(
            events=previous_stage_events,
            source_label="历史阶段",
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            progress_logger=progress_logger,
            stage_name=stage_name,
            anchor_index=0,
            total_anchors=len(seed_events),
        )
        trope_bank.extend(historical_motifs)
        progress_logger.info(f"历史阶段套路已入库：{len(historical_motifs)} 条，bank 累计 {len(trope_bank)} 条")

    for anchor_index, target_anchor in enumerate(seed_events):
        progress_logger.log_segment_start(stage_name, anchor_index, len(seed_events), target_anchor)
        future_anchors = seed_events[anchor_index + 1 :]
        segment_prefix = previous_stage_events + final_event_line
        segment_committed: list[str] = []
        local_chain_rounds: list[dict[str, Any]] = []
        anchor_reached_by_search = False
        force_appended_anchor = False

        for round_index in range(1, max_local_chains_per_anchor + 1):
            if len(segment_committed) >= max_events_to_anchor:
                break

            remaining_total_budget = max_events_to_anchor - len(segment_committed)
            effective_local_depth = min(local_max_depth, remaining_total_budget)
            if effective_local_depth <= 0:
                break

            fixed_prefix_for_round = segment_prefix + segment_committed
            progress_logger.log_local_chain_start(
                stage=stage_name,
                anchor_index=anchor_index,
                total_anchors=len(seed_events),
                round_index=round_index,
                local_max_depth=effective_local_depth,
                prefix_events=segment_committed,
            )
            search_result = build_tree_with_parallel_generation(
                node_factory=node_factory,
                title=title,
                background=background,
                highlights=highlights,
                summary=summary,
                stage_name=stage_name,
                seed_events=seed_events,
                fixed_prefix=fixed_prefix_for_round,
                target_anchor=target_anchor,
                future_anchors=future_anchors,
                branch_factor=branch_factor,
                keep_top_k=keep_top_k,
                local_max_depth=effective_local_depth,
                model_name=model_name,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                parallel_workers=parallel_workers,
                progress_logger=progress_logger,
                anchor_index=anchor_index,
                total_anchors=len(seed_events),
                round_index=round_index,
                anchor_remaining_budget=remaining_total_budget,
            )

            terminal_chains = collect_terminal_chains(search_result)
            local_chain_scores: list[dict[str, Any]] = []
            local_chain_score_raw_response = ""
            selected_local_chain: dict[str, Any] | None = None

            if terminal_chains:
                scores, local_chain_score_raw_response = score_local_chains_once(
                    title=title,
                    background=background,
                    highlights=highlights,
                    summary=summary,
                    stage_name=stage_name,
                    seed_events=seed_events,
                    fixed_prefix=fixed_prefix_for_round,
                    target_anchor=target_anchor,
                    future_anchors=future_anchors,
                    terminal_chains=terminal_chains,
                    model_name=model_name,
                    max_new_tokens=max_new_tokens,
                    progress_logger=progress_logger,
                    anchor_index=anchor_index,
                    total_anchors=len(seed_events),
                    round_index=round_index,
                    trope_bank=trope_bank,
                )
                for index, chain in enumerate(terminal_chains):
                    score_payload = scores[index]
                    local_chain_scores.append(
                        {
                            "chain_index": index + 1,
                            "events": chain["events"],
                            "path": chain["path"],
                            "terminal_reason": chain["terminal_reason"],
                            "anchor_reached": chain["anchor_reached"],
                            "final_progress_to_anchor": chain["final_progress_to_anchor"],
                            "coherence": score_payload["coherence"],
                            "novelty": score_payload["novelty"],
                            "target_fit": score_payload["target_fit"],
                            "continuation_value": score_payload["continuation_value"],
                            "reason": score_payload["reason"],
                            "score": compute_local_chain_score(score_payload, chain["anchor_reached"]),
                        }
                    )
                selected_local_chain = choose_best_local_chain(local_chain_scores)

            round_committed_events = selected_local_chain["events"] if selected_local_chain else []
            if round_committed_events:
                segment_committed.extend(round_committed_events)

            round_anchor_reached = bool(selected_local_chain and selected_local_chain["anchor_reached"])
            local_chain_rounds.append(
                {
                    "round_index": round_index,
                    "fixed_prefix_used": fixed_prefix_for_round,
                    "tree": search_result["tree"],
                    "layer_summaries": search_result["layer_summaries"],
                    "completed_paths": search_result["completed_paths"],
                    "incomplete_paths": search_result["incomplete_paths"],
                    "terminal_chain_scores": local_chain_scores,
                    "terminal_chain_score_raw_response": local_chain_score_raw_response,
                    "selected_local_chain": selected_local_chain,
                    "round_committed": round_committed_events,
                    "anchor_reached_in_round": round_anchor_reached,
                }
            )
            progress_logger.log_local_chain_finish(
                stage=stage_name,
                anchor_index=anchor_index,
                total_anchors=len(seed_events),
                round_index=round_index,
                committed_events=round_committed_events,
                anchor_reached=round_anchor_reached,
            )

            if round_anchor_reached:
                anchor_reached_by_search = True
                break

            if not round_committed_events:
                break

        if not anchor_reached_by_search:
            if not segment_committed or segment_committed[-1] != target_anchor:
                segment_committed.append(target_anchor)
                force_appended_anchor = True

        final_event_line.extend(segment_committed)

        # v6：局部链选好后，提取本段提交事件的叙事套路，及时入库
        segment_motifs = abstract_micro_motifs_once(
            events=segment_committed,
            source_label=f"段{anchor_index + 1}",
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            progress_logger=progress_logger,
            stage_name=stage_name,
            anchor_index=anchor_index,
            total_anchors=len(seed_events),
        )
        trope_bank.extend(segment_motifs)
        progress_logger.info(f"段 {anchor_index + 1} 套路已入库：{len(segment_motifs)} 条，bank 累计 {len(trope_bank)} 条")

        segment_results.append(
            {
                "anchor_index": anchor_index,
                "target_anchor": target_anchor,
                "fixed_prefix_used": segment_prefix,
                "local_chain_rounds": local_chain_rounds,
                "segment_committed": segment_committed,
                "anchor_reached_by_search": anchor_reached_by_search,
                "force_appended_anchor": force_appended_anchor,
                "total_local_rounds": len(local_chain_rounds),
            }
        )
        progress_logger.log_segment_finish(stage_name, anchor_index, len(seed_events), segment_committed)

        if resolved_output_path:
            partial_result = assemble_generation_result(
                outline_path=outline_path,
                stage_name=stage_name,
                title=title,
                branch_factor=branch_factor,
                keep_top_k=keep_top_k,
                local_max_depth=local_max_depth,
                max_events_to_anchor=max_events_to_anchor,
                max_local_chains_per_anchor=max_local_chains_per_anchor,
                parallel_workers=parallel_workers,
                model_name=model_name,
                temperature=temperature,
                generated_at=generated_at,
                stage_payload=stage_payload,
                previous_stage_events=previous_stage_events,
                final_event_line=final_event_line,
                segment_results=segment_results,
                trope_bank=trope_bank,
            )
            save_json(resolved_output_path, partial_result)
            progress_logger.log_write(resolved_output_path, len(segment_results), len(seed_events))

    return assemble_generation_result(
        outline_path=outline_path,
        stage_name=stage_name,
        title=title,
        branch_factor=branch_factor,
        keep_top_k=keep_top_k,
        local_max_depth=local_max_depth,
        max_events_to_anchor=max_events_to_anchor,
        max_local_chains_per_anchor=max_local_chains_per_anchor,
        parallel_workers=parallel_workers,
        model_name=model_name,
        temperature=temperature,
        generated_at=generated_at,
        stage_payload=stage_payload,
        previous_stage_events=previous_stage_events,
        final_event_line=final_event_line,
        segment_results=segment_results,
        trope_bank=trope_bank,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对指定 stage 执行局部树搜索式事件线生成实验 v6")
    parser.add_argument("--outline-path", default=str(DEFAULT_OUTLINE_PATH), help="story outline JSON 路径")
    parser.add_argument("--stage", default="发展", help="要扩写的 stage 名称，默认'发展'")
    parser.add_argument("--branch-factor", type=int, default=4, help="每个节点生成多少个下一事件候选")
    parser.add_argument("--keep-top-k", type=int, default=2, help="每层保留多少个候选继续扩展")
    parser.add_argument("--local-max-depth", type=int, default=4, help="单轮局部链搜索的最大 depth，默认 4")
    parser.add_argument(
        "--max-events-to-anchor",
        type=int,
        default=12,
        help="同一锚点段允许累计提交的最大事件数，作为续链安全上限",
    )
    parser.add_argument(
        "--max-local-chains-per-anchor",
        type=int,
        default=4,
        help="同一锚点段最多开启多少轮局部链，防止无限续链",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=8,
        help="同层叶子节点候选生成的最大并行 worker 数",
    )
    parser.add_argument("--model-name", default="gemini-2.5-pro", help="模型名")
    parser.add_argument("--temperature", type=float, default=0.9, help="候选生成采样温度")
    parser.add_argument("--max-new-tokens", type=int, default=32768, help="最大生成 token 数")
    parser.add_argument("--output", help="输出 JSON 路径；不传则自动落到 output/stage_eventline_local_tree_v6")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve() if args.output else build_output_path(DEFAULT_OUTPUT_DIR, args.stage)
    result = generate_stage_eventline_with_local_tree_v6(
        outline_path=args.outline_path,
        stage_name=args.stage,
        branch_factor=args.branch_factor,
        keep_top_k=args.keep_top_k,
        local_max_depth=args.local_max_depth,
        max_events_to_anchor=args.max_events_to_anchor,
        max_local_chains_per_anchor=args.max_local_chains_per_anchor,
        parallel_workers=args.parallel_workers,
        model_name=args.model_name,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        output_path=output_path,
    )
    save_json(output_path, result)
    print(f"已完成 `{args.stage}` stage 的局部树搜索实验 v6")
    print(f"最终事件数：{len(result['final_event_line'])}")
    print(f"结果已写入：{output_path}")


if __name__ == "__main__":
    main()
