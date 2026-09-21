from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluations.common.project_llm_local import query_json

from evaluations.common.io import save_json

DEFAULT_OUTLINE_PATH = PROJECT_ROOT / "output/60ep_0810_1946/03_story_outline/outline.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output/stage_eventline_local_tree_v2"

STEP_SCORING_FEWSHOT = """
Few-shot 示例 1：
前序已发生情节：
- 女主刚靠撒泼和舆论表演夺回将军府控制权。
- 接下来剧情需要自然转入第一笔讨债。
当前分支：
- 团队准备去找礼部尚书追债。
同一父节点下的候选下一事件：
候选 1：女主继续在尚书府门口哭嚎骂街，逼对方拿钱息事宁人。
候选 2：团队先摸清尚书最爱名画、最怕赝品丑闻，再借一场假拍卖逼他露怯。
评分参考：
- 候选 1：logic = 7, drive = 7, novelty = 3, appeal = 5
- 说明：能推动讨债，但机制仍是“公开闹场施压”，只是换个对象重复旧套路。
- 候选 2：logic = 8, drive = 8, novelty = 8, appeal = 8
- 说明：既顺着讨债目标推进，又从硬闹切换到设局智取，较明显跳脱前序套路。

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

SCORING_FEWSHOT = """
Few-shot 示例 1：
前序已发生情节：
- 女主刚靠当街撒泼赶走恶邻，团队也第一次露出“扮猪吃老虎”的底牌。
- 男主已经拿出欠条，接下来必须自然转入讨债主线。
候选链 A：
- 女主带人直接去礼部尚书府门口哭丧讨债。
- 尚书嫌丢人，甩出银票打发他们。
评分参考：
- coherence = 7
- novelty = 3
- anchor_fit = 8
- 说明：逻辑基本成立，但推进几乎仍是“公开闹场→权贵嫌烦给钱”的旧套路，和前序“撒泼驱敌”在机制上过于相似，缺乏新变化。

Few-shot 示例 2：
前序已发生情节：
- 女主刚靠厨房争夺战和舆论表演夺回将军府控制权。
- 团队准备向第一位权贵讨债。
候选链 B：
- 团队先调查出尚书酷爱名画且最怕赝品丑闻。
- 女主设局办一场赝品拍卖会，让尚书在抢画、护脸面和还债之间自乱阵脚。
评分参考：
- coherence = 8
- novelty = 8
- anchor_fit = 8
- 说明：链条顺着人物目标推进，且从“撒泼式硬闹”切换到“投其所好+设局智取”，相对前序情节有明确的推进机制变化，属于跳脱既有套路但仍合理。

Few-shot 示例 3：
前序已发生情节：
- 男女主刚通过讨债追回家产，继母准备反扑。
- 宁国公作为后续更大反派尚未正式浮出水面。
候选链 C：
- 继母直接当众说出宁国公就是最终真凶。
- 皇帝当场下旨抄家宁国公。
评分参考：
- coherence = 4
- novelty = 2
- anchor_fit = 3
- 说明：虽然表面上有大反转，但它粗暴消耗了后续悬念，且让真凶暴露与朝堂清算来得过快，破坏后续锚点节奏，不算有效的新颖，只是跳步。
""".strip()


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


def collect_previous_stage_events(outline: dict[str, Any], target_stage_name: str) -> list[str]:
    framework = outline.get("framework", [])
    if not isinstance(framework, list):
        raise ValueError("outline.framework 必须是数组")

    previous_events: list[str] = []
    for stage in framework:
        if not isinstance(stage, dict):
            continue
        current_name = str(stage.get("stage", "")).strip()
        if current_name == target_stage_name:
            return previous_events
        previous_events.extend(
            str(item.get("event", "")).strip()
            for item in stage.get("event_list", [])
            if isinstance(item, dict) and str(item.get("event", "")).strip()
        )
    raise ValueError(f"未找到 stage：{target_stage_name}")


class NodeIdFactory:
    def __init__(self) -> None:
        self._next = 1

    def new_id(self) -> str:
        value = self._next
        self._next += 1
        return f"n{value:04d}"


class ProgressLogger:
    def __init__(self) -> None:
        self._call_index = 0

    def log(self, stage: str, anchor_index: int, total_anchors: int, phase: str, detail: str) -> None:
        self._call_index += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(
            f"[{timestamp}] LLM调用#{self._call_index} | stage={stage} | 段={anchor_index + 1}/{total_anchors} | {phase} | {detail}",
            flush=True,
        )

    def log_segment_start(self, stage: str, anchor_index: int, total_anchors: int, target_anchor: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(
            f"[{timestamp}] 开始处理锚点段 {anchor_index + 1}/{total_anchors} | stage={stage} | target={target_anchor}",
            flush=True,
        )

    def log_segment_finish(self, stage: str, anchor_index: int, total_anchors: int, committed_events: list[str]) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(
            f"[{timestamp}] 完成锚点段 {anchor_index + 1}/{total_anchors} | stage={stage} | 提交事件数={len(committed_events)}",
            flush=True,
        )

    def log_write(self, output_path: Path, completed_segments: int, total_segments: int) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(
            f"[{timestamp}] 已落盘 | progress={completed_segments}/{total_segments} | file={output_path}",
            flush=True,
        )


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

    prompt = f"""你现在扮演短剧剧情规划器，要为某个 stage 做“局部树搜索”的下一步扩展。

你的任务不是一次性写完整阶段，而是：
- 基于当前已经固定的前序剧情；
- 只提出“下一步可能发生的事件”候选；
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
1. 只输出接下来的 {branch_factor} 个候选“下一事件”。
2. 每个候选必须是一句话，不写对白，不写镜头，不写分集。
3. 候选之间不要只是同义改写，要在推进方式上拉开差异。
4. 允许候选是桥接、受挫、反打、关系推进、线索浮现、代价暴露中的任意一种。
5. `progress_to_anchor` 取 0-100，表示该事件发生后距离目标锚点有多近。
6. `anchor_reached=true` 只在该事件已经可以视作抵达当前目标锚点时使用。
7. 尽量不要粗暴复用前序剧情已经用过的推进机制；如果前序已经靠“当众撒泼/硬闹/公开打脸”解决问题，这一步优先尝试新的推进办法。
8. 不要为了显得新就跳步，不要提前把后续锚点的核心悬念直接消耗掉。
9. 当前这条分支最多还允许再写 {remaining_budget} 个事件，务必控制在预算内自然抵达目标锚点。
10. 只输出 JSON，不要解释。

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
                "anchor_reached": sanitize_bool(item.get("anchor_reached"), default=False),
                "reason": str(item.get("reason", "")).strip(),
            }
        )
        if len(normalized) >= branch_factor:
            break
    if not normalized:
        raise ValueError("模型没有返回可用候选")
    return normalized


def build_candidate_step_score_messages(
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
    candidates: list[dict[str, Any]],
) -> list[dict[str, str]]:
    candidate_lines = []
    for index, candidate in enumerate(candidates, start=1):
        candidate_lines.append(f"候选 {index}：")
        candidate_lines.append(f"- event: {candidate['event']}")
        candidate_lines.append(f"- progress_to_anchor: {candidate['progress_to_anchor']}")
        candidate_lines.append(f"- anchor_reached: {candidate['anchor_reached']}")
        candidate_lines.append(f"- generation_reason: {candidate.get('reason', '')}")
    candidates_text = "\n".join(candidate_lines)

    schema = {
        "scores": [
            {
                "candidate_index": 1,
                "logic": 8,
                "drive": 8,
                "novelty": 7,
                "appeal": 7,
                "reason": "两三句话说明这个下一事件为什么适合或不适合放在这里，尤其解释它是否顺逻辑、能推动情节，以及 novelty 和吸引力高低。",
            }
        ]
    }

    prompt = f"""你现在是短剧剧情树搜索中的独立小评委，负责给“同一父节点下的多个下一事件候选”分别打分，并决定哪些候选更值得继续扩展。

注意：
- 你不是生成者，不要补写剧情，不要改写候选。
- 你评估的是“这个下一事件放在当前位置是否合适”，而不是整条长链。
- novelty 不是猎奇程度，而是：相对于 fixed_prefix + current_segment 已经出现过的推进套路，这个候选是否真的有机制变化，而不是换皮重复。
- appeal 指短剧观看层面的吸引力：它是否有冲突、钩子、反差、悬念、爽点或情绪牵引，能让观众愿意继续看下一步。
- 如果一个候选虽然热闹，但破坏人物动机、强行跳步、提前消费 future_anchors 的核心悬念，logic、novelty、appeal 都不能高。
- 请分别给每个候选打分，不要只选一个。

{STEP_SCORING_FEWSHOT}

待评估的故事信息：
- 标题：{title}
- 故事背景：{background}
- 卖点：{highlights}
- 摘要：{summary}
- 当前 stage：{stage_name}
- 本 stage 的粗粒度锚点：{seed_events}
- 已固定的完整前缀事件（包含前序 stage 与本 stage 已完成部分）：{fixed_prefix}
- 当前这条分支已经走到的事件链：{current_segment}
- 当前要逼近的目标锚点：{target_anchor}
- 后续锚点（必须避免提前消耗其核心悬念）：{future_anchors}

同一父节点下待评估的候选下一事件：
{candidates_text}

评分维度：
1. `logic`：1-10，评价这个下一事件放在这里是否符合人物动机、因果关系和事件逻辑。
2. `drive`：1-10，评价这个下一事件是否真正推动剧情继续朝目标锚点前进，而不是原地热闹。
3. `novelty`：1-10，评价它相对于 fixed_prefix + current_segment 的固有套路，是否有真实的新推进机制。
4. `appeal`：1-10，评价它作为短剧下一拍是否足够抓人，是否有钩子、反差、悬念、爽点或情绪牵引。
5. `reason`：2-3 句话，必须明确解释它为什么顺或不顺、为什么能或不能推进情节，以及 novelty / appeal 高低的原因。

只输出 JSON，不要解释。
JSON 格式：{schema}
"""
    return [{"role": "user", "content": prompt}]


def parse_candidate_step_scores(payload: Any, expected_count: int) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if not isinstance(payload, dict):
        raise ValueError("候选步评分返回不是 JSON 对象")
    raw_scores = payload.get("scores", [])
    if not isinstance(raw_scores, list):
        raise ValueError("scores 必须是数组")

    score_map: dict[int, dict[str, Any]] = {}
    for item in raw_scores:
        if not isinstance(item, dict):
            continue
        candidate_index = sanitize_int(item.get("candidate_index"), default=-1, minimum=-1, maximum=10_000)
        if candidate_index < 1 or candidate_index > expected_count:
            continue
        score_map[candidate_index - 1] = {
            "logic": sanitize_int(item.get("logic"), default=7, minimum=1, maximum=10),
            "drive": sanitize_int(item.get("drive"), default=7, minimum=1, maximum=10),
            "novelty": sanitize_int(item.get("novelty"), default=6, minimum=1, maximum=10),
            "appeal": sanitize_int(item.get("appeal"), default=6, minimum=1, maximum=10),
            "reason": str(item.get("reason", "")).strip(),
        }

    normalized: list[dict[str, Any]] = []
    for index in range(expected_count):
        normalized.append(
            score_map.get(
                index,
                {
                    "logic": 7,
                    "drive": 7,
                    "novelty": 6,
                    "appeal": 6,
                    "reason": "评分模型未返回该候选分数，使用默认值。",
                },
            )
        )
    return normalized


def compute_candidate_expansion_score(score_payload: dict[str, Any]) -> float:
    logic = float(score_payload["logic"])
    drive = float(score_payload["drive"])
    novelty = float(score_payload["novelty"])
    appeal = float(score_payload["appeal"])
    return round(0.34 * logic + 0.28 * drive + 0.20 * novelty + 0.18 * appeal, 4)


def score_candidates_for_expansion_once(
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
    candidates: list[dict[str, Any]],
    model_name: str,
    max_new_tokens: int,
    progress_logger: ProgressLogger,
    anchor_index: int,
    total_anchors: int,
    depth: int,
) -> tuple[list[dict[str, Any]], str]:
    messages = build_candidate_step_score_messages(
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
        candidates=candidates,
    )
    progress_logger.log(
        stage=stage_name,
        anchor_index=anchor_index,
        total_anchors=total_anchors,
        phase="候选评分",
        detail=f"depth={depth} | parent_path_len={len(current_segment)} | candidates={len(candidates)}",
    )
    parsed, raw_response = query_json(
        messages,
        model_name=model_name,
        temperature=0.2,
        max_new_tokens=max_new_tokens,
    )
    scores = parse_candidate_step_scores(parsed, len(candidates))
    return scores, raw_response


def choose_top_candidates_for_expansion(scored_candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    ranked = sorted(
        scored_candidates,
        key=lambda item: (
            item["expansion_score"],
            item["logic"],
            item["drive"],
            item["novelty"],
            item["appeal"],
            item["progress_to_anchor"],
            1 if item["anchor_reached"] else 0,
        ),
        reverse=True,
    )
    return ranked[: max(1, top_k)]


def build_batch_chain_score_messages(
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
    completed_chains: list[list[str]],
) -> list[dict[str, str]]:
    chain_lines = []
    for index, chain in enumerate(completed_chains, start=1):
        chain_lines.append(f"候选链 {index}：")
        for event in chain:
            chain_lines.append(f"- {event}")
    chains_text = "\n".join(chain_lines)

    schema = {
        "scores": [
            {
                "chain_index": 1,
                "coherence": 8,
                "novelty": 7,
                "anchor_fit": 8,
                "reason": "两三句话说明评分理由，重点解释 novelty 是否真正跳脱了前序情节固有套路",
            }
        ]
    }

    prompt = f"""你现在是短剧剧情树搜索中的独立评委，负责给“多条完整候选链”分别打分。

注意：
- 你不是生成者，不要帮它润色或补写事件。
- 你必须只评估每条候选链本身，并分别给出分数。
- novelty 的含义不是猎奇程度，而是：这条链相对于“前序情节已经反复出现的推进套路”是否有真正跳脱。
- 如果这条链只是换皮重复前序情节的老办法（例如前面已经多次靠当众撒泼、公开羞辱、硬闹打脸来推进，这里只是换个对象继续这么做），那么 novelty 必须偏低。
- 如果这条链虽然看起来热闹，但破坏人物动机、跳过铺垫、提前消耗未来锚点的核心悬念，也不能算高 novelty。
- 所有候选链都已经抵达当前目标锚点，请比较它们各自到达方式是否自然、是否真正跳脱前序套路。

{SCORING_FEWSHOT}

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

待评估候选链如下：
{chains_text}

评分维度：
1. `coherence`：1-10，评价这整条链的逻辑、人物动机、因果衔接是否自然。
2. `novelty`：1-10，评价这整条链相对于前序情节固有推进套路是否有真实变化；要和 fixed_prefix 对比，而不是只看表面热闹。
3. `anchor_fit`：1-10，评价这整条链是否自然逼近并抵达当前目标锚点，同时没有粗暴跳步。
4. `reason`：2-3 句话，必须明确说明 novelty 为什么高或低，是否跳脱了前序套路，以及是否有提前消费后续锚点的问题。

只输出 JSON，不要解释。
JSON 格式：{schema}
"""
    return [{"role": "user", "content": prompt}]


def parse_batch_chain_scores(payload: Any, expected_count: int) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if not isinstance(payload, dict):
        raise ValueError("批量链路评分返回不是 JSON 对象")
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
            "anchor_fit": sanitize_int(item.get("anchor_fit"), default=7, minimum=1, maximum=10),
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
                    "anchor_fit": 7,
                    "reason": "评分模型未返回该链分数，使用默认值。",
                },
            )
        )
    return normalized


def compute_chain_score(score_payload: dict[str, Any]) -> float:
    coherence = float(score_payload["coherence"])
    novelty = float(score_payload["novelty"])
    anchor_fit = float(score_payload["anchor_fit"])
    return round(0.42 * coherence + 0.38 * novelty + 0.20 * anchor_fit, 4)


def score_completed_chains_once(
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
    completed_chains: list[list[str]],
    model_name: str,
    max_new_tokens: int,
    progress_logger: ProgressLogger,
    anchor_index: int,
    total_anchors: int,
) -> tuple[list[dict[str, Any]], str]:
    messages = build_batch_chain_score_messages(
        title=title,
        background=background,
        highlights=highlights,
        summary=summary,
        stage_name=stage_name,
        seed_events=seed_events,
        fixed_prefix=fixed_prefix,
        target_anchor=target_anchor,
        future_anchors=future_anchors,
        completed_chains=completed_chains,
    )
    progress_logger.log(
        stage=stage_name,
        anchor_index=anchor_index,
        total_anchors=total_anchors,
        phase="整链评分",
        detail=f"completed_chains={len(completed_chains)}",
    )
    parsed, raw_response = query_json(
        messages,
        model_name=model_name,
        temperature=0.2,
        max_new_tokens=max_new_tokens,
    )
    scores = parse_batch_chain_scores(parsed, len(completed_chains))
    return scores, raw_response


def create_node_payload(
    *,
    node_id: str,
    event: str,
    depth: int,
    progress_to_anchor: int,
    anchor_reached: bool,
    reason: str,
    selected_for_expansion: bool,
    logic: int | None = None,
    drive: int | None = None,
    novelty: int | None = None,
    appeal: int | None = None,
    evaluation_reason: str = "",
    expansion_score: float | None = None,
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "event": event,
        "depth": depth,
        "progress_to_anchor": progress_to_anchor,
        "anchor_reached": anchor_reached,
        "generation_reason": reason,
        "selected_for_expansion": selected_for_expansion,
        "logic": logic,
        "drive": drive,
        "novelty": novelty,
        "appeal": appeal,
        "evaluation_reason": evaluation_reason,
        "expansion_score": expansion_score,
        "children": [],
    }


def build_tree_until_anchor(
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
    max_events_to_anchor: int,
    model_name: str,
    temperature: float,
    max_new_tokens: int,
    progress_logger: ProgressLogger,
    anchor_index: int,
    total_anchors: int,
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

    for depth in range(1, max_events_to_anchor + 1):
        if not active_states:
            break

        next_active_states: list[dict[str, Any]] = []
        layer_info = {"depth": depth, "expanded_paths": []}

        for state in active_states:
            path = state["path"]
            parent_node = state["tree_node"]
            current_segment = [item["event"] for item in path]
            remaining_budget = max_events_to_anchor - len(current_segment)
            if remaining_budget <= 0:
                incomplete_paths.append(
                    {
                        "events": current_segment,
                        "terminal_reason": "budget_exhausted_before_generation",
                    }
                )
                continue

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
                remaining_budget=remaining_budget,
            )
            progress_logger.log(
                stage=stage_name,
                anchor_index=anchor_index,
                total_anchors=total_anchors,
                phase="候选生成",
                detail=f"depth={depth} | parent_path_len={len(current_segment)} | remaining_budget={remaining_budget}",
            )
            parsed, raw_generation_response = query_json(
                messages,
                model_name=model_name,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            candidates = parse_candidates(parsed, branch_factor)

            step_scores, raw_step_scoring_response = score_candidates_for_expansion_once(
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
                candidates=candidates,
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                progress_logger=progress_logger,
                anchor_index=anchor_index,
                total_anchors=total_anchors,
                depth=depth,
            )

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
                "generation_raw_response": raw_generation_response,
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
                "terminal_reason": "max_events_to_anchor_reached_without_anchor",
            }
        )

    return {
        "tree": root,
        "layer_summaries": layer_summaries,
        "completed_paths": completed_paths,
        "incomplete_paths": incomplete_paths,
    }


def choose_best_scored_chain(scored_chains: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not scored_chains:
        return None
    ranked = sorted(
        scored_chains,
        key=lambda item: (
            item["score"],
            item["novelty"],
            item["coherence"],
            item["anchor_fit"],
        ),
        reverse=True,
    )
    return ranked[0]


def assemble_generation_result(
    *,
    outline_path: str | Path,
    stage_name: str,
    title: str,
    branch_factor: int,
    keep_top_k: int,
    max_events_to_anchor: int,
    model_name: str,
    temperature: float,
    generated_at: str,
    stage_payload: dict[str, Any],
    previous_stage_events: list[str],
    final_event_line: list[str],
    segment_results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "meta": {
            "mode": "local_tree_search_experiment_v2",
            "outline_path": str(Path(outline_path).expanduser().resolve()),
            "stage": stage_name,
            "title": title,
            "branch_factor": branch_factor,
            "keep_top_k": keep_top_k,
            "max_events_to_anchor": max_events_to_anchor,
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
        "segments": segment_results,
    }


def generate_stage_eventline_with_local_tree_v2(
    *,
    outline_path: str | Path,
    stage_name: str,
    branch_factor: int,
    keep_top_k: int,
    max_events_to_anchor: int,
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

    for anchor_index, target_anchor in enumerate(seed_events):
        progress_logger.log_segment_start(stage_name, anchor_index, len(seed_events), target_anchor)
        fixed_prefix = previous_stage_events + final_event_line
        future_anchors = seed_events[anchor_index + 1 :]
        search_result = build_tree_until_anchor(
            node_factory=node_factory,
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
            keep_top_k=keep_top_k,
            max_events_to_anchor=max_events_to_anchor,
            model_name=model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            progress_logger=progress_logger,
            anchor_index=anchor_index,
            total_anchors=len(seed_events),
        )

        completed_event_chains = [item["events"] for item in search_result["completed_paths"]]
        scored_chains: list[dict[str, Any]] = []
        scoring_raw_response = ""
        selected_chain: dict[str, Any] | None = None
        force_appended_anchor = False

        if completed_event_chains:
            scores, scoring_raw_response = score_completed_chains_once(
                title=title,
                background=background,
                highlights=highlights,
                summary=summary,
                stage_name=stage_name,
                seed_events=seed_events,
                fixed_prefix=fixed_prefix,
                target_anchor=target_anchor,
                future_anchors=future_anchors,
                completed_chains=completed_event_chains,
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                progress_logger=progress_logger,
                anchor_index=anchor_index,
                total_anchors=len(seed_events),
            )
            for index, chain in enumerate(completed_event_chains):
                score_payload = scores[index]
                scored_chains.append(
                    {
                        "chain_index": index + 1,
                        "events": chain,
                        "coherence": score_payload["coherence"],
                        "novelty": score_payload["novelty"],
                        "anchor_fit": score_payload["anchor_fit"],
                        "reason": score_payload["reason"],
                        "score": compute_chain_score(score_payload),
                    }
                )
            selected_chain = choose_best_scored_chain(scored_chains)

        if selected_chain:
            final_event_line.extend(selected_chain["events"])
        else:
            final_event_line.append(target_anchor)
            force_appended_anchor = True

        committed_events = selected_chain["events"] if selected_chain else [target_anchor]
        segment_results.append(
            {
                "anchor_index": anchor_index,
                "target_anchor": target_anchor,
                "fixed_prefix_used": fixed_prefix,
                "tree": search_result["tree"],
                "layer_summaries": search_result["layer_summaries"],
                "completed_paths": search_result["completed_paths"],
                "incomplete_paths": search_result["incomplete_paths"],
                "completed_chain_scores": scored_chains,
                "completed_chain_score_raw_response": scoring_raw_response,
                "selected_chain": selected_chain,
                "segment_committed": committed_events,
                "anchor_reached_by_search": bool(selected_chain),
                "force_appended_anchor": force_appended_anchor,
            }
        )
        progress_logger.log_segment_finish(stage_name, anchor_index, len(seed_events), committed_events)

        if resolved_output_path:
            partial_result = assemble_generation_result(
                outline_path=outline_path,
                stage_name=stage_name,
                title=title,
                branch_factor=branch_factor,
                keep_top_k=keep_top_k,
                max_events_to_anchor=max_events_to_anchor,
                model_name=model_name,
                temperature=temperature,
                generated_at=generated_at,
                stage_payload=stage_payload,
                previous_stage_events=previous_stage_events,
                final_event_line=final_event_line,
                segment_results=segment_results,
            )
            save_json(resolved_output_path, partial_result)
            progress_logger.log_write(resolved_output_path, len(segment_results), len(seed_events))

    return assemble_generation_result(
        outline_path=outline_path,
        stage_name=stage_name,
        title=title,
        branch_factor=branch_factor,
        keep_top_k=keep_top_k,
        max_events_to_anchor=max_events_to_anchor,
        model_name=model_name,
        temperature=temperature,
        generated_at=generated_at,
        stage_payload=stage_payload,
        previous_stage_events=previous_stage_events,
        final_event_line=final_event_line,
        segment_results=segment_results,
    )


def build_output_path(output_dir: str | Path, stage_name: str) -> Path:
    safe_stage = "".join(ch if ch.isalnum() else "_" for ch in stage_name).strip("_") or "stage"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_dir).expanduser().resolve() / f"{safe_stage}_local_tree_v2_{timestamp}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对指定 stage 执行局部树搜索式事件线生成实验 v2")
    parser.add_argument("--outline-path", default=str(DEFAULT_OUTLINE_PATH), help="story outline JSON 路径")
    parser.add_argument("--stage", default="发展", help="要扩写的 stage 名称，默认‘发展’")
    parser.add_argument("--branch-factor", type=int, default=4, help="每个节点生成多少个下一事件候选")
    parser.add_argument("--keep-top-k", type=int, default=2, help="每层保留多少个候选继续扩展")
    parser.add_argument("--max-events-to-anchor", type=int, default=6, help="每个锚点段最多允许展开多少个事件，作为安全上限")
    parser.add_argument("--model-name", default="gemini-2.5-pro", help="模型名")
    parser.add_argument("--temperature", type=float, default=0.9, help="候选生成采样温度")
    parser.add_argument("--max-new-tokens", type=int, default=8192, help="最大生成 token 数")
    parser.add_argument("--output", help="输出 JSON 路径；不传则自动落到 output/stage_eventline_local_tree_v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve() if args.output else build_output_path(DEFAULT_OUTPUT_DIR, args.stage)
    result = generate_stage_eventline_with_local_tree_v2(
        outline_path=args.outline_path,
        stage_name=args.stage,
        branch_factor=args.branch_factor,
        keep_top_k=args.keep_top_k,
        max_events_to_anchor=args.max_events_to_anchor,
        model_name=args.model_name,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        output_path=output_path,
    )
    save_json(output_path, result)
    print(f"已完成 `{args.stage}` stage 的局部树搜索实验 v2")
    print(f"最终事件数：{len(result['final_event_line'])}")
    print(f"结果已写入：{output_path}")


if __name__ == "__main__":
    main()
