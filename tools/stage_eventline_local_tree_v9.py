from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluations.common.project_llm_local import query_json

from evaluations.common.io import save_json


try:
    # 业务链路以 package 方式导入本模块。
    from tools.stage_eventline_local_tree_v2 import (
        DEFAULT_OUTLINE_PATH,
        NodeIdFactory,
        choose_top_candidates_for_expansion,
        collect_previous_stage_events,
        create_node_payload,
        extract_stage_payload,
        load_outline,
        parse_candidate_step_scores,
        parse_candidates,
        sanitize_int,
    )
except ImportError:
    # 保留 `python tools/stage_eventline_local_tree_v9.py` 独立运行方式。
    from stage_eventline_local_tree_v2 import (
        DEFAULT_OUTLINE_PATH,
        NodeIdFactory,
        choose_top_candidates_for_expansion,
        collect_previous_stage_events,
        create_node_payload,
        extract_stage_payload,
        load_outline,
        parse_candidate_step_scores,
        parse_candidates,
        sanitize_int,
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


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output/stage_eventline_local_tree_v9"

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


def write_llm_failure_dump(
    *,
    stage_name: str,
    anchor_index: int,
    total_anchors: int,
    phase: str,
    attempt: int,
    error: Exception,
    raw_response: str | None,
) -> Path:
    failure_dir = DEFAULT_OUTPUT_DIR / "llm_failure_dumps"
    failure_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M%S_%f")
    safe_stage = stage_name.replace("/", "_").replace(" ", "_")
    safe_phase = phase.replace("/", "_").replace(" ", "_")
    dump_path = failure_dir / (
        f"{safe_stage}_seg{anchor_index + 1}of{total_anchors}_{safe_phase}_attempt{attempt}_{timestamp}.txt"
    )
    content = [
        f"stage={stage_name}",
        f"anchor_index={anchor_index}",
        f"total_anchors={total_anchors}",
        f"phase={phase}",
        f"attempt={attempt}",
        f"error_type={type(error).__name__}",
        f"error_message={error}",
        "",
        "===RAW_RESPONSE_BEGIN===",
        raw_response or "<EMPTY_RAW_RESPONSE>",
        "===RAW_RESPONSE_END===",
        "",
    ]
    dump_path.write_text("\n".join(content), encoding="utf-8")
    return dump_path


def call_llm_with_schema_retry(
    *,
    phase: str,
    stage_name: str,
    anchor_index: int,
    total_anchors: int,
    progress_logger: ProgressLogger,
    runner: Callable[[], tuple[Any, str]],
    parser: Callable[[Any, str], Any],
    max_attempts: int = 3,
) -> tuple[Any, str]:
    last_error: Exception | None = None
    last_dump_path: Path | None = None
    for attempt in range(1, max_attempts + 1):
        raw_response = None
        try:
            parsed, raw_response = runner()
            result = parser(parsed, raw_response)
            if attempt > 1:
                progress_logger.info(
                    f"{phase} 重试成功 | stage={stage_name} | 段={anchor_index + 1}/{total_anchors} | attempt={attempt}"
                )
            return result, raw_response
        except Exception as error:
            last_error = error
            last_dump_path = write_llm_failure_dump(
                stage_name=stage_name,
                anchor_index=anchor_index,
                total_anchors=total_anchors,
                phase=phase,
                attempt=attempt,
                error=error,
                raw_response=raw_response,
            )
            progress_logger.info(
                f"{phase} 返回解析失败 | stage={stage_name} | 段={anchor_index + 1}/{total_anchors} | attempt={attempt}/{max_attempts} | dump={last_dump_path} | error={error}"
            )
            if attempt < max_attempts:
                time.sleep(float(attempt))
    assert last_error is not None
    raise RuntimeError(
        f"{phase} 连续 {max_attempts} 次返回解析失败，最后一次原始返回已保存到 {last_dump_path}"
    ) from last_error


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
    def run_query() -> tuple[Any, str]:
        return query_json(
            [{"role": "user", "content": prompt}],
            model_name=model_name,
            temperature=0.2,
            max_new_tokens=max_new_tokens,
        )

    def parse_motifs(payload: Any, _raw_response: str) -> list[dict[str, str]]:
        if isinstance(payload, list):
            payload = payload[0] if payload else {}
        motifs: list[dict[str, str]] = []
        if isinstance(payload, dict):
            for item in payload.get("motifs", []):
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

    motifs, _raw = call_llm_with_schema_retry(
        phase="套路提取",
        stage_name=stage_name,
        anchor_index=anchor_index,
        total_anchors=total_anchors,
        progress_logger=progress_logger,
        runner=run_query,
        parser=parse_motifs,
    )
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
    trope_bank: list[dict[str, str]],
) -> list[dict[str, str]]:
    candidate_lines = []
    for index, candidate in enumerate(candidates, start=1):
        candidate_lines.append(f"候选 {index}：")
        candidate_lines.append(f"- event: {candidate['event']}")
        candidate_lines.append(f"- progress_to_anchor: {candidate['progress_to_anchor']}")
        candidate_lines.append(f"- anchor_reached: {candidate['anchor_reached']}")
        candidate_lines.append(f"- generation_reason: {candidate.get('reason', '')}")
    candidates_text = "\n".join(candidate_lines)
    trope_bank_text = format_trope_bank(trope_bank)

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

    prompt = f"""你现在是短剧剧情树搜索中的独立小评委，负责给"同一父节点下的多个下一事件候选"分别打分，并决定哪些候选更值得继续扩展。

注意：
- 你不是生成者，不要补写剧情，不要改写候选。
- 你评估的是"这个下一事件放在当前位置是否合适"，而不是整条长链。
- novelty 不是猎奇程度，而是：相对于"已用叙事套路 bank"里已记录的因果套路，这个候选是否真的有机制变化，而不是换皮重复。
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
- 已用过的叙事套路 bank（起因->手段->结果->反应，供 novelty 去重）：
{trope_bank_text}
- 当前要逼近的目标锚点：{target_anchor}
- 后续锚点（必须避免提前消耗其核心悬念）：{future_anchors}

同一父节点下待评估的候选下一事件：
{candidates_text}

评分维度：
1. `logic`：1-10，评价这个下一事件放在这里是否符合人物动机、因果关系和事件逻辑。
2. `drive`：1-10，评价这个下一事件是否真正推动剧情继续朝目标锚点前进，而不是原地热闹。
3. `novelty`：1-10，评价它相对"已用叙事套路 bank"中记录的因果链（起因->手段->结果->反应），是否引入了新的因果推进模式，而非复用已固定的完整前缀事件中的叙事套路（包含前序 stage 与本 stage 已完成部分）；若本候选的核心套路与 bank 中某条高度雷同或相似度较高（换皮复用），novelty 必须给低分；若本候选所属的事件链内部存在叙事同质化、套路化的问题，也应适度下调 novelty 得分。
4. `appeal`：1-10，评价它作为短剧下一拍是否足够抓人，是否有钩子、反差、悬念、爽点或情绪牵引。
5. `reason`：2-3 句话，必须明确解释它为什么顺或不顺、为什么能或不能推进情节，以及 novelty / appeal 高低的原因。

只输出 JSON，不要解释。
JSON 格式：{schema}
"""
    return [{"role": "user", "content": prompt}]


def compute_candidate_expansion_score(score_payload: dict[str, Any]) -> float:
    logic = float(score_payload["logic"])
    drive = float(score_payload["drive"])
    novelty = float(score_payload["novelty"])
    appeal = float(score_payload["appeal"])
    return round(0.30 * logic + 0.20 * drive + 0.30 * novelty + 0.20 * appeal, 4)


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
    trope_bank: list[dict[str, str]],
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
        trope_bank=trope_bank,
    )
    progress_logger.log(
        stage=stage_name,
        anchor_index=anchor_index,
        total_anchors=total_anchors,
        phase="候选评分",
        detail=f"depth={depth} | parent_path_len={len(current_segment)} | candidates={len(candidates)}",
    )
    def run_query() -> tuple[Any, str]:
        return query_json(
            messages,
            model_name=model_name,
            temperature=0.2,
            max_new_tokens=max_new_tokens,
        )

    return call_llm_with_schema_retry(
        phase="候选评分",
        stage_name=stage_name,
        anchor_index=anchor_index,
        total_anchors=total_anchors,
        progress_logger=progress_logger,
        runner=run_query,
        parser=lambda payload, _raw_response: parse_candidate_step_scores(payload, len(candidates)),
    )


def build_local_chain_score_messages(
    *, title: str, background: str, highlights: str, summary: str,
    stage_name: str, seed_events: list[str], fixed_prefix: list[str],
    target_anchor: str, future_anchors: list[str], chain: dict[str, Any],
    trope_bank: list[dict[str, str]], anchor_remaining_budget: int,
) -> list[dict[str, str]]:
    remaining_budget = anchor_remaining_budget - len(chain['events'])
    if remaining_budget < 0:
        raise ValueError('候选局部链超过当前锚点剩余事件预算')
    schema = {
        'consistency_review': '引用前序事件与当前链中的具体事件，说明状态、因果、衔接是否冲突；有冲突须逐项指出，无冲突也要说明核对依据。',
        'novelty_review': '与 bank 中相关套路比较，说明机制变化或重复之处。',
        'budget_review': '说明当前链是否实际到达目标；未到时还缺哪些必要事件，能否在剩余额度内自然完成。',
        'coherence': 80, 'novelty': 70, 'budget_fit': 80,
        'reason': '简要总结这条链的优缺点，与三个维度评分保持一致。',
    }
    prompt = f"""你是短剧剧情局部树搜索的独立评委。局部事件链是目前设计的、接在已确定事件之后的一串后续事件，目标是抵达下一个粗事件锚点。本次只审查一条这样的局部候选事件链，分别给出三个维度的分数。

请认真核对“已确定的完整前序事件”与“当前局部链”，先给出有具体事件依据的审查结论，再打分。

故事基本信息：
- 标题：{title}
- 背景：{background}
- 卖点：{highlights}
- 摘要：{summary}
- 当前阶段：{stage_name}
- 当前阶段粗锚点：{json.dumps(seed_events, ensure_ascii=False)}

已确定的完整前序事件列表（前序阶段全部事件＋当前阶段已提交事件）：
{json.dumps(fixed_prefix, ensure_ascii=False, indent=2)}

当前待评局部链（只评价这一条；以下事件接在上述前序事件之后）：
{json.dumps(chain['events'], ensure_ascii=False, indent=2)}

目标及真实事件预算：
- 当前目标锚点：{target_anchor}
- 到达目标锚点的剩余事件预算：{remaining_budget} 条事件（已扣除当前局部链）。
- 生成者自报是否到达：{chain['anchor_reached']}；自报进度：{chain['final_progress_to_anchor']}；停止原因：{chain['terminal_reason']}。请结合实际事件判断目标完成程度。
- 当前阶段后续锚点（供评估剧情衔接与推进顺序）：{json.dumps(future_anchors, ensure_ascii=False)}

已用叙事套路 bank（起因→手段→结果→反应）：
{format_trope_bank(trope_bank)}

评分维度（各0—100分，满分100分）：
1. coherence，权重40%：评估当前局部链与全部前序事件是否存在矛盾，情节衔接是否自然合理，并检查链内一致性。
   检查事项包括但不限于：当前局部链条与前序事件的人物身份、人物或世界状态与关系是否连续或存在情节逻辑矛盾，人物行为是否符合已有动机和掌握的信息，事件的起因、行动与结果是否相互支持，时间地点的变化是否有合理过渡，以及前序事件的结果和已有设定是否得到一致承接。
   结合具体事件说明矛盾或衔接缺口，并考虑其严重程度与影响范围。
   0—39分：存在严重矛盾或因果断裂，影响核心情节成立。
   40—69分：基本情节可理解，但有明显状态冲突、动机跳跃或衔接缺口。
   70—89分：整体一致、衔接合理，仅有少量可完善之处。
   90—100分：与前序及链内事件高度一致，人物行动和情节推进自然充分。

2. novelty，权重30%：重点检查当前局部链的因果推进机制，是否与此前 trope bank 中的某一条或多条机制相似。
   对照“起因→手段→结果→反应”，指出最接近的已有套路，说明核心机制的相似之处与实质差异；若无明显相似项，也需给出依据。同时考虑本链内部的机制重复。
   0—39分：与已有机制高度相似，主要是更换人物、对象或道具。
   40—69分：有局部变化，但核心因果推进方式仍接近已有套路。
   70—89分：存在明确的机制差异，形成较新的推进方式。
   90—100分：因果推进机制有显著创新，与已有套路区别鲜明且融入故事自然。

3. budget_fit，权重30%：评估当前局部链推进到的位置，能否在给定剩余事件预算内自然抵达目标锚点。
   综合考虑目标已完成的程度、尚缺的关键推进、过渡难度与剩余预算，按完成差距和后续可行性连续评分。
   0—39分：距离目标较远，仍缺多个关键步骤，预算内完成的可行性较低。
   40—69分：已完成部分关键推进，但仍有明显缺口，预算较紧或需要较大压缩。
   70—89分：已接近目标，主要推进充分，剩余工作较少或有清楚可行的完成路径。
   90—100分：已自然完整抵达目标，或剩余预算足以支持清晰、充分的后续推进。

程序按0.40×coherence＋0.30×novelty＋0.30×budget_fit计算总分。请按以下格式输出一个JSON对象。
JSON格式：{json.dumps(schema, ensure_ascii=False)}
"""
    return [{'role': 'user', 'content': prompt}]


def parse_local_chain_score(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError('单链评分必须是JSON对象')
    result = {}
    for key in ('coherence', 'novelty', 'budget_fit'):
        value = payload.get(key)
        if type(value) not in (int, float) or not 0 <= value <= 100:
            raise ValueError(f'{key}必须是0—100的数字')
        result[key] = value
    for key in ('consistency_review', 'novelty_review', 'budget_review', 'reason'):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{key}必须提供非空审查依据')
        result[key] = value.strip()
    return result


def compute_local_chain_score(score_payload: dict[str, Any]) -> float:
    return round(0.40 * score_payload['coherence'] +
                 0.30 * score_payload['novelty'] +
                 0.30 * score_payload['budget_fit'], 4)


def score_local_chain_once(
    *, title, background, highlights, summary, stage_name, seed_events,
    fixed_prefix, target_anchor, future_anchors, chain, trope_bank,
    anchor_remaining_budget, model_name, max_new_tokens, progress_logger,
    anchor_index, total_anchors, round_index, chain_index, chain_count,
):
    messages = build_local_chain_score_messages(
        title=title, background=background, highlights=highlights, summary=summary,
        stage_name=stage_name, seed_events=seed_events, fixed_prefix=fixed_prefix,
        target_anchor=target_anchor, future_anchors=future_anchors, chain=chain,
        trope_bank=trope_bank, anchor_remaining_budget=anchor_remaining_budget)
    progress_logger.log(
        stage=stage_name, anchor_index=anchor_index, total_anchors=total_anchors,
        phase='局部链独立评分',
        detail=f'round={round_index} | chain={chain_index}/{chain_count} | remaining_after_chain={anchor_remaining_budget-len(chain["events"])}')

    def run_query():
        return query_json(messages, model_name=model_name, temperature=0.2,
                          max_new_tokens=max_new_tokens)

    return call_llm_with_schema_retry(
        phase=f'局部链独立评分_chain_{chain_index}', stage_name=stage_name,
        anchor_index=anchor_index, total_anchors=total_anchors,
        progress_logger=progress_logger, runner=run_query,
        parser=lambda payload, _raw: parse_local_chain_score(payload))


def score_local_chains_in_parallel(
    *, title, background, highlights, summary, stage_name, seed_events,
    fixed_prefix, target_anchor, future_anchors, terminal_chains, trope_bank,
    anchor_remaining_budget, model_name, max_new_tokens, parallel_workers,
    progress_logger, anchor_index, total_anchors, round_index,
):
    if not terminal_chains:
        return [], '', []
    scores = [None] * len(terminal_chains)
    records = [None] * len(terminal_chains)
    with ThreadPoolExecutor(max_workers=max(1, min(parallel_workers, len(terminal_chains)))) as executor:
        futures = {
            executor.submit(
                score_local_chain_once,
                title=title, background=background, highlights=highlights, summary=summary,
                stage_name=stage_name, seed_events=seed_events, fixed_prefix=fixed_prefix,
                target_anchor=target_anchor, future_anchors=future_anchors,
                chain=chain, trope_bank=trope_bank, anchor_remaining_budget=anchor_remaining_budget,
                model_name=model_name, max_new_tokens=max_new_tokens,
                progress_logger=progress_logger, anchor_index=anchor_index,
                total_anchors=total_anchors, round_index=round_index,
                chain_index=index+1, chain_count=len(terminal_chains)): index
            for index, chain in enumerate(terminal_chains)
        }
        for future in as_completed(futures):
            index = futures[future]
            score, raw = future.result()
            scores[index] = score
            records[index] = {
                'chain_index': index+1, 'candidate_count': 1,
                'anchor_remaining_budget_before_chain': anchor_remaining_budget,
                'candidate_event_count': len(terminal_chains[index]['events']),
                'remaining_budget_after_chain': anchor_remaining_budget-len(terminal_chains[index]['events']),
                'evaluation': score, 'raw_response': raw,
            }
    raw_response = '\n\n'.join(f'=== chain {r["chain_index"]}/{len(records)} ===\n{r["raw_response"]}' for r in records)
    return scores, raw_response, records


def choose_best_local_chain(scored_chains: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not scored_chains:
        return None
    # 总分相同才依次比较一致性、预算可行性、新颖性；完全相同保留原顺序。
    return max(scored_chains, key=lambda item: (
        item['score'], item['coherence'], item['budget_fit'], item['novelty']))


def build_output_path(output_dir: str | Path, stage_name: str) -> Path:
    safe_stage = "".join(ch if ch.isalnum() else "_" for ch in stage_name).strip("_") or "stage"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_dir).expanduser().resolve() / f"{safe_stage}_local_tree_v8_{timestamp}.json"


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
    def run_query() -> tuple[Any, str]:
        return query_json(
            messages,
            model_name=model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    candidates, raw_generation_response = call_llm_with_schema_retry(
        phase="候选生成",
        stage_name=stage_name,
        anchor_index=anchor_index,
        total_anchors=total_anchors,
        progress_logger=progress_logger,
        runner=run_query,
        parser=lambda payload, _raw_response: parse_candidates(payload, branch_factor),
    )
    return {
        "position": position,
        "state": state,
        "current_segment": current_segment,
        "remaining_budget": remaining_budget,
        "candidates": candidates,
        "raw_generation_response": raw_generation_response,
        "skipped": False,
    }


def should_trigger_mutation(
    *,
    prepared_result: dict[str, Any],
    stage_name: str,
    anchor_index: int,
    round_index: int,
    depth: int,
    mutation_probability: float,
) -> tuple[bool, float]:
    if mutation_probability <= 0:
        return False, 1.0
    if mutation_probability >= 1:
        return True, 0.0

    current_segment = prepared_result.get("current_segment", [])
    fingerprint = "||".join(
        [
            stage_name,
            str(anchor_index),
            str(round_index),
            str(depth),
            str(prepared_result.get("position", 0)),
            *current_segment,
        ]
    )
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
    roll = int(digest[:12], 16) / float(16**12 - 1)
    return roll < mutation_probability, roll


def build_mutation_messages(
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
    best_candidate: dict[str, Any],
    scored_candidates: list[dict[str, Any]],
) -> list[dict[str, str]]:
    schema = {
        "candidates": [
            {
                "event": "一句话概括突变后的下一步事件",
                "progress_to_anchor": 72,
                "anchor_reached": False,
                "reason": "一句话说明它相对原候选改变了什么因果推进机制",
            }
        ]
    }

    sibling_lines: list[str] = []
    for index, candidate in enumerate(scored_candidates, start=1):
        sibling_lines.append(f"候选 {index}：")
        sibling_lines.append(f"- event: {candidate['event']}")
        sibling_lines.append(f"- progress_to_anchor: {candidate['progress_to_anchor']}")
        sibling_lines.append(f"- anchor_reached: {candidate['anchor_reached']}")
        sibling_lines.append(f"- logic: {candidate['logic']}")
        sibling_lines.append(f"- drive: {candidate['drive']}")
        sibling_lines.append(f"- novelty: {candidate['novelty']}")
        sibling_lines.append(f"- appeal: {candidate['appeal']}")
        sibling_lines.append(f"- expansion_score: {candidate['expansion_score']}")
        sibling_lines.append(f"- reason: {candidate['evaluation_reason']}")
    sibling_text = "\n".join(sibling_lines)

    prompt = f"""你现在是短剧剧情树搜索里的“候选进化器”。

你的任务：
- 基于当前父节点下评分最高的那个下一事件候选，做一次“突变”；
- 突变不是同义改写，而是要适度改变这个下一事件与前后剧情的因果关系、触发条件、出手方式、借势对象、博弈结构或推进机制；
- 但突变后仍然必须保持人物动机成立、因果逻辑顺、并继续朝当前目标锚点推进；
- 这个突变候选会作为额外保留分支，因此必须和原高分候选以及同层其他候选在推进机制上有清晰差异。

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
- 后续锚点（必须避免提前消耗其核心悬念）：{future_anchors}

当前父节点下已评分的原始候选：
{sibling_text}

其中当前最高分候选是：
- event: {best_candidate['event']}
- progress_to_anchor: {best_candidate['progress_to_anchor']}
- anchor_reached: {best_candidate['anchor_reached']}
- generation_reason: {best_candidate.get('reason', '')}
- evaluation_reason: {best_candidate['evaluation_reason']}
- expansion_score: {best_candidate['expansion_score']}

突变要求：
1. 只输出 1 个突变后的“下一事件”候选。
2. 必须还是“下一步事件”，不能一口气写成两三步连招。
3. 不能只是换个说法复述最高分候选，也不能与同层其他候选实质重复。
4. 优先改变“怎么推进到目标锚点”的机制，例如从硬闯改成设局、从正面对抗改成借势、从被动挨打改成反诱导等。
5. 可以改变局部因果触发，但不能破坏人物动机，不能强行跳步，不能提前消耗后续锚点悬念。
6. 每个候选仍只用一句话，不写对白，不写镜头，不写分集。
7. `progress_to_anchor` 取 0-100，表示该事件发生后距离目标锚点有多近。
8. `anchor_reached=true` 只在该事件已经可以视作抵达当前目标锚点时使用。
9. 只输出 JSON，不要解释。

JSON 格式：{schema}
"""
    return [{"role": "user", "content": prompt}]


def mutate_top_candidate_for_state_once(
    *,
    prepared_result: dict[str, Any],
    title: str,
    background: str,
    highlights: str,
    summary: str,
    stage_name: str,
    seed_events: list[str],
    fixed_prefix: list[str],
    target_anchor: str,
    future_anchors: list[str],
    trope_bank: list[dict[str, str]],
    model_name: str,
    temperature: float,
    max_new_tokens: int,
    progress_logger: ProgressLogger,
    anchor_index: int,
    total_anchors: int,
    depth: int,
) -> dict[str, Any]:
    scored_candidates = prepared_result.get("scored_candidates", [])
    if not scored_candidates:
        return {
            "position": prepared_result["position"],
            "mutation_candidate": None,
            "raw_mutation_response": "",
            "raw_mutation_score_response": "",
        }

    best_candidate = choose_top_candidates_for_expansion(scored_candidates, 1)[0]
    current_segment = prepared_result["current_segment"]
    messages = build_mutation_messages(
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
        best_candidate=best_candidate,
        scored_candidates=scored_candidates,
    )
    progress_logger.log(
        stage=stage_name,
        anchor_index=anchor_index,
        total_anchors=total_anchors,
        phase="候选突变",
        detail=f"depth={depth} | parent_path_len={len(current_segment)} | base={best_candidate['event']}",
    )
    def run_query() -> tuple[Any, str]:
        return query_json(
            messages,
            model_name=model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    mutation_candidates, raw_mutation_response = call_llm_with_schema_retry(
        phase="候选突变",
        stage_name=stage_name,
        anchor_index=anchor_index,
        total_anchors=total_anchors,
        progress_logger=progress_logger,
        runner=run_query,
        parser=lambda payload, _raw_response: parse_candidates(payload, 1),
    )
    mutation_candidate = mutation_candidates[0] if mutation_candidates else None
    if not mutation_candidate:
        return {
            "position": prepared_result["position"],
            "mutation_candidate": None,
            "raw_mutation_response": raw_mutation_response,
            "raw_mutation_score_response": "",
        }

    existing_events = {candidate["event"] for candidate in scored_candidates}
    if mutation_candidate["event"] in existing_events:
        progress_logger.log(
            stage=stage_name,
            anchor_index=anchor_index,
            total_anchors=total_anchors,
            phase="候选突变跳过",
            detail="突变结果与原始候选重复，跳过该额外分支",
        )
        return {
            "position": prepared_result["position"],
            "mutation_candidate": None,
            "raw_mutation_response": raw_mutation_response,
            "raw_mutation_score_response": "",
        }

    progress_logger.log(
        stage=stage_name,
        anchor_index=anchor_index,
        total_anchors=total_anchors,
        phase="突变评分",
        detail=f"depth={depth} | event={mutation_candidate['event']}",
    )
    mutation_scores, raw_mutation_score_response = score_candidates_for_expansion_once(
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
        candidates=[mutation_candidate],
        trope_bank=trope_bank,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        progress_logger=progress_logger,
        anchor_index=anchor_index,
        total_anchors=total_anchors,
        depth=depth,
    )
    score_payload = mutation_scores[0]
    return {
        "position": prepared_result["position"],
        "mutation_candidate": {
            **mutation_candidate,
            "logic": score_payload["logic"],
            "drive": score_payload["drive"],
            "novelty": score_payload["novelty"],
            "appeal": score_payload["appeal"],
            "evaluation_reason": score_payload["reason"],
            "expansion_score": compute_candidate_expansion_score(score_payload),
            "candidate_source": "mutation",
            "mutated_from_event": best_candidate["event"],
        },
        "raw_mutation_response": raw_mutation_response,
        "raw_mutation_score_response": raw_mutation_score_response,
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
    trope_bank: list[dict[str, str]],
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
        trope_bank=trope_bank,
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
    mutation_probability: float,
    progress_logger: ProgressLogger,
    anchor_index: int,
    total_anchors: int,
    round_index: int,
    anchor_remaining_budget: int,
    trope_bank: list[dict[str, str]],
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
                        trope_bank=trope_bank,
                    ): result["position"]
                    for result in scoring_inputs
                }
                for future in as_completed(future_to_position):
                    scored_result = future.result()
                    scoring_results_by_position[scored_result["position"]] = scored_result

        prepared_results: list[dict[str, Any]] = []
        for result in generation_results:
            if result["skipped"]:
                prepared_results.append(result)
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
                        "candidate_source": "original",
                        "mutated_from_event": "",
                    }
                )
            prepared_results.append(
                {
                    **result,
                    "scored_candidates": scored_candidates,
                    "raw_step_scoring_response": raw_step_scoring_response,
                }
            )

        mutation_results_by_position: dict[int, dict[str, Any]] = {}
        mutation_inputs: list[dict[str, Any]] = []
        for result in prepared_results:
            if result.get("skipped") or not result.get("scored_candidates"):
                continue
            mutation_selected, mutation_roll = should_trigger_mutation(
                prepared_result=result,
                stage_name=stage_name,
                anchor_index=anchor_index,
                round_index=round_index,
                depth=depth,
                mutation_probability=mutation_probability,
            )
            result["mutation_roll"] = mutation_roll
            result["mutation_selected"] = mutation_selected
            if mutation_selected:
                mutation_inputs.append(result)

        if mutation_inputs:
            mutation_workers = max(1, min(parallel_workers, len(mutation_inputs)))
            with ThreadPoolExecutor(max_workers=mutation_workers) as executor:
                future_to_position = {
                    executor.submit(
                        mutate_top_candidate_for_state_once,
                        prepared_result=result,
                        title=title,
                        background=background,
                        highlights=highlights,
                        summary=summary,
                        stage_name=stage_name,
                        seed_events=seed_events,
                        fixed_prefix=fixed_prefix,
                        target_anchor=target_anchor,
                        future_anchors=future_anchors,
                        trope_bank=trope_bank,
                        model_name=model_name,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        progress_logger=progress_logger,
                        anchor_index=anchor_index,
                        total_anchors=total_anchors,
                        depth=depth,
                    ): result["position"]
                    for result in mutation_inputs
                }
                for future in as_completed(future_to_position):
                    mutation_result = future.result()
                    mutation_results_by_position[mutation_result["position"]] = mutation_result

        next_active_states: list[dict[str, Any]] = []

        for result in prepared_results:
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

            scored_candidates = list(result["scored_candidates"])
            raw_step_scoring_response = result["raw_step_scoring_response"]
            mutation_result = mutation_results_by_position.get(result["position"], {})
            mutated_candidate = mutation_result.get("mutation_candidate")
            raw_mutation_response = mutation_result.get("raw_mutation_response", "")
            raw_mutation_score_response = mutation_result.get("raw_mutation_score_response", "")

            selected_original_candidates = choose_top_candidates_for_expansion(scored_candidates, keep_top_k)
            selected_candidates = list(selected_original_candidates)
            if mutated_candidate:
                selected_candidates.append(mutated_candidate)

            selected_node_keys = {
                (candidate["event"], candidate.get("candidate_source", "original")) for candidate in selected_candidates
            }

            expansion_record = {
                "parent_path": current_segment,
                "generation_raw_response": result["raw_generation_response"],
                "step_score_raw_response": raw_step_scoring_response,
                "mutation_raw_response": raw_mutation_response,
                "mutation_step_score_raw_response": raw_mutation_score_response,
                "generated_candidates": [],
            }

            all_candidates_for_tree = list(scored_candidates)
            if mutated_candidate:
                all_candidates_for_tree.append(mutated_candidate)

            for candidate in all_candidates_for_tree:
                node = create_node_payload(
                    node_id=node_factory.new_id(),
                    event=candidate["event"],
                    depth=depth,
                    progress_to_anchor=candidate["progress_to_anchor"],
                    anchor_reached=candidate["anchor_reached"],
                    reason=candidate.get("reason", ""),
                    selected_for_expansion=(candidate["event"], candidate.get("candidate_source", "original")) in selected_node_keys,
                    logic=candidate["logic"],
                    drive=candidate["drive"],
                    novelty=candidate["novelty"],
                    appeal=candidate["appeal"],
                    evaluation_reason=candidate["evaluation_reason"],
                    expansion_score=candidate["expansion_score"],
                )
                node["candidate_source"] = candidate.get("candidate_source", "original")
                node["mutated_from_event"] = candidate.get("mutated_from_event", "")
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
                    "candidate_source": node["candidate_source"],
                    "mutated_from_event": node["mutated_from_event"],
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
    mutation_probability: float,
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
            "mode": "local_tree_search_experiment_v9",
            "local_chain_judge": "individual_consistency_novelty_budget_v2_100",
            "local_chain_score_weights": {"coherence": 0.4, "novelty": 0.3, "budget_fit": 0.3},
            "outline_path": str(Path(outline_path).expanduser().resolve()),
            "stage": stage_name,
            "title": title,
            "branch_factor": branch_factor,
            "keep_top_k": keep_top_k,
            "local_max_depth": local_max_depth,
            "max_events_to_anchor": max_events_to_anchor,
            "max_local_chains_per_anchor": max_local_chains_per_anchor,
            "parallel_workers": parallel_workers,
            "mutation_probability": mutation_probability,
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


def generate_stage_eventline_with_local_tree_v9(
    *,
    outline_path: str | Path,
    stage_name: str,
    branch_factor: int,
    keep_top_k: int,
    local_max_depth: int,
    max_events_to_anchor: int,
    max_local_chains_per_anchor: int,
    parallel_workers: int,
    mutation_probability: float = 0.25,
    model_name: str,
    temperature: float,
    max_new_tokens: int,
    output_path: str | Path | None = None,
    progress_logger: ProgressLogger | None = None,
    initial_trope_bank: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    mutation_probability = max(0.0, min(1.0, mutation_probability))
    outline = load_outline(outline_path)
    stage_payload = extract_stage_payload(outline, stage_name)
    seed_events = stage_payload["seed_events"]
    previous_stage_events = collect_previous_stage_events(outline, stage_name)

    title = str(outline.get("title", "")).strip()
    background = str(outline.get("background", "")).strip()
    highlights = str(outline.get("highlights", "")).strip()
    summary = str(outline.get("summary", "")).strip()
    generated_at = datetime.now().isoformat(timespec="seconds")
    mutation_probability = max(0.0, min(1.0, mutation_probability))
    resolved_output_path = Path(output_path).expanduser().resolve() if output_path else None

    progress_logger = progress_logger or ProgressLogger()
    node_factory = NodeIdFactory()
    final_event_line: list[str] = []
    segment_results: list[dict[str, Any]] = []

    # v9：初始化叙事套路 bank
    trope_bank: list[dict[str, str]] = []
    if initial_trope_bank:
        # 复用已有 bank（前序 stage 已优化、bank 已建好），不再重新对历史阶段提取套路
        trope_bank = list(initial_trope_bank)
        progress_logger.info(f"复用已有叙事套路 bank：{len(trope_bank)} 条，跳过历史阶段建库")
    elif previous_stage_events:
        # 前序 stage 尚未优化、无 bank，需要先对历史阶段建立套路
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
                mutation_probability=mutation_probability,
                progress_logger=progress_logger,
                anchor_index=anchor_index,
                total_anchors=len(seed_events),
                round_index=round_index,
                anchor_remaining_budget=remaining_total_budget,
                trope_bank=trope_bank,
            )

            terminal_chains = collect_terminal_chains(search_result)
            local_chain_scores: list[dict[str, Any]] = []
            local_chain_score_raw_response = ""
            local_chain_score_records: list[dict[str, Any]] = []
            selected_local_chain: dict[str, Any] | None = None

            if terminal_chains:
                scores, local_chain_score_raw_response, local_chain_score_records = score_local_chains_in_parallel(
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
                    anchor_remaining_budget=remaining_total_budget,
                    model_name=model_name,
                    max_new_tokens=max_new_tokens,
                    parallel_workers=parallel_workers,
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
                            **score_payload,
                            "remaining_budget_after_chain": remaining_total_budget - len(chain["events"]),
                            "score": compute_local_chain_score(score_payload),
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
                    "anchor_remaining_budget_before_round": remaining_total_budget,
                    "fixed_prefix_used": fixed_prefix_for_round,
                    "tree": search_result["tree"],
                    "layer_summaries": search_result["layer_summaries"],
                    "completed_paths": search_result["completed_paths"],
                    "incomplete_paths": search_result["incomplete_paths"],
                    "terminal_chain_scores": local_chain_scores,
                    "terminal_chain_score_raw_response": local_chain_score_raw_response,
                    "terminal_chain_score_records": local_chain_score_records,
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

        # v9：局部链选好后，提取本段提交事件的叙事套路，及时入库
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
                mutation_probability=mutation_probability,
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
        mutation_probability=mutation_probability,
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
    parser = argparse.ArgumentParser(description="对指定 stage 执行局部树搜索式事件线生成实验 v9")
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
        default=32,
        help="候选生成与评分的最大并行 worker 数",
    )
    parser.add_argument("--model-name", default="gemini-2.5-pro", help="模型名")
    parser.add_argument(
        "--mutation-probability",
        type=float,
        default=0.25,
        help="每个父节点触发突变分支的概率，默认 0.25",
    )
    parser.add_argument("--temperature", type=float, default=0.9, help="候选生成采样温度")
    parser.add_argument("--max-new-tokens", type=int, default=32768, help="最大生成 token 数")
    parser.add_argument("--output", help="输出 JSON 路径；不传则自动落到 output/stage_eventline_local_tree_v9")
    parser.add_argument(
        "--initial-trope-bank",
        help="已有叙事套路 bank 的 JSON 文件路径（复用前序 stage 已建好的 bank，跳过历史阶段建库）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve() if args.output else build_output_path(DEFAULT_OUTPUT_DIR, args.stage)

    # 读取已有的叙事套路 bank（若提供），复用前序 stage 已建好的 bank
    initial_trope_bank: list[dict[str, str]] | None = None
    if args.initial_trope_bank:
        import json
        with open(args.initial_trope_bank, "r", encoding="utf-8") as f:
            bank_data = json.load(f)
        if isinstance(bank_data, dict):
            bank_data = bank_data.get("trope_bank", [])
        if isinstance(bank_data, list):
            initial_trope_bank = bank_data
        print(f"已载入初始叙事套路 bank：{len(initial_trope_bank)} 条")

    result = generate_stage_eventline_with_local_tree_v9(
        outline_path=args.outline_path,
        stage_name=args.stage,
        branch_factor=args.branch_factor,
        keep_top_k=args.keep_top_k,
        local_max_depth=args.local_max_depth,
        max_events_to_anchor=args.max_events_to_anchor,
        max_local_chains_per_anchor=args.max_local_chains_per_anchor,
        parallel_workers=args.parallel_workers,
        mutation_probability=args.mutation_probability,
        model_name=args.model_name,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        output_path=output_path,
        initial_trope_bank=initial_trope_bank,
    )
    save_json(output_path, result)
    print(f"已完成 `{args.stage}` stage 的局部树搜索实验 v9")
    print(f"最终事件数：{len(result['final_event_line'])}")
    print(f"结果已写入：{output_path}")


if __name__ == "__main__":
    main()
