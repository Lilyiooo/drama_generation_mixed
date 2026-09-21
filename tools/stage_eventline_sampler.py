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

try:
    from evaluations.common.project_llm_trpc import query_json
except ImportError:
    from evaluations.common.project_llm import query_json

from evaluations.common.io import save_json


DEFAULT_OUTLINE_PATH = PROJECT_ROOT / "output/60ep_0810_1946/03_story_outline/outline.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output/stage_eventline_sampler"


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


def build_messages(
    title: str,
    background: str,
    highlights: str,
    summary: str,
    stage_name: str,
    seed_events: list[str],
    candidate_index: int,
    target_event_count: int,
    previous_candidates: list[dict[str, Any]],
) -> list[dict[str, str]]:
    json_schema = {
        "candidate_id": f"{stage_name}-candidate-{candidate_index:02d}",
        "core_idea": "这一版事件线的核心变化方向，用一句话概括",
        "event_line": [
            "事件1：一句话概括",
            "事件2：一句话概括",
            "事件3：一句话概括",
        ],
    }

    prior_summaries = [
        {
            "candidate_id": item.get("candidate_id", ""),
            "core_idea": item.get("core_idea", ""),
            "event_line_head": item.get("event_line", [])[:4],
        }
        for item in previous_candidates
    ]

    prompt = f"""你是短剧策划编辑，现在要基于已有的大纲，为其中一个阶段扩写出一条更完整的事件线。

任务目标：
1. 只处理 `{stage_name}` 这个 stage。
2. 你输出的是一条“更完整的阶段事件线”，不是整部剧，不写分集，不写场景。
3. 每个节点只允许一句话概括事件内容，不要展开成段落。
4. 事件链要比原始大纲更细，但仍保持中观粒度。
5. 必须保留原 stage 的核心方向和卖点，不能改掉题材与主线。
6. 允许补充桥接事件、反噬事件、关系推进事件、线索浮现事件。
7. 重点追求：更完整、更顺滑、不要只是在原事件上换个说法。
8. 输出事件数尽量控制在 {target_event_count} 个左右，可上下浮动 2 个。
9. 事件顺序必须形成清晰推进关系，最终能自然收束到该 stage 的末尾目标。
10. 只输出 JSON 对象，不要额外解释。

创作约束：
- 每个事件必须是一句话。
- 不要写“第X集”。
- 不要写镜头、对白、分场格式。
- 不要写过细执行细节。
- 不要把事件写成同义改写的重复项。
- 要兼顾事业推进、冲突升级、关系变化、旧案线索浮现。

故事标题：{title}
故事背景：{background}
整体卖点：{highlights}
整体摘要：{summary}

当前 stage：{stage_name}
原始粗粒度事件：{seed_events}

你这次要生成的是第 {candidate_index} 条候选事件线。
已有候选摘要（用于避免重复）：{prior_summaries}

请尽量在“推进机制”上和已有候选拉开差异，例如：
- 从先内宅后讨债，改成先借讨债反制内宅；
- 从持续强攻，改成试探、受挫、借势反打；
- 从单线推进，改成事业线与旧案线并行渗透；
- 从纯爽推进，改成阶段胜利伴随代价。

JSON 输出格式示例：{json_schema}
"""
    return [{"role": "user", "content": prompt}]


def normalize_candidate(payload: Any, candidate_index: int) -> dict[str, Any]:
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if not isinstance(payload, dict):
        raise ValueError("模型返回的 JSON 不是对象")

    raw_events = payload.get("event_line", [])
    if not isinstance(raw_events, list):
        raise ValueError("event_line 必须是数组")

    event_line = [str(item).strip() for item in raw_events if str(item).strip()]
    if not event_line:
        raise ValueError("event_line 不能为空")

    candidate_id = str(payload.get("candidate_id", "")).strip() or f"candidate-{candidate_index:02d}"
    core_idea = str(payload.get("core_idea", "")).strip() or "未提供核心思路"

    return {
        "candidate_id": candidate_id,
        "core_idea": core_idea,
        "event_count": len(event_line),
        "event_line": event_line,
    }


def generate_stage_eventlines(
    outline_path: str | Path,
    stage_name: str,
    sample_count: int,
    target_event_count: int,
    model_name: str,
    temperature: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    outline = load_outline(outline_path)
    stage_payload = extract_stage_payload(outline, stage_name)

    title = str(outline.get("title", "")).strip()
    background = str(outline.get("background", "")).strip()
    highlights = str(outline.get("highlights", "")).strip()
    summary = str(outline.get("summary", "")).strip()

    candidates: list[dict[str, Any]] = []
    for candidate_index in range(1, sample_count + 1):
        messages = build_messages(
            title=title,
            background=background,
            highlights=highlights,
            summary=summary,
            stage_name=stage_name,
            seed_events=stage_payload["seed_events"],
            candidate_index=candidate_index,
            target_event_count=target_event_count,
            previous_candidates=candidates,
        )
        parsed, raw_response = query_json(
            messages,
            model_name=model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        candidate = normalize_candidate(parsed, candidate_index)
        candidate["raw_response"] = raw_response
        candidates.append(candidate)

    return {
        "meta": {
            "outline_path": str(Path(outline_path).expanduser().resolve()),
            "stage": stage_name,
            "title": title,
            "sample_count": sample_count,
            "target_event_count": target_event_count,
            "model_name": model_name,
            "temperature": temperature,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "source_stage": stage_payload,
        "candidates": candidates,
    }


def build_output_path(output_dir: str | Path, stage_name: str) -> Path:
    safe_stage = "".join(ch if ch.isalnum() else "_" for ch in stage_name).strip("_") or "stage"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_dir).expanduser().resolve() / f"{safe_stage}_eventlines_{timestamp}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="读取 outline.json 中指定 stage，多次调用模型生成更完整的阶段事件线")
    parser.add_argument("--outline-path", default=str(DEFAULT_OUTLINE_PATH), help="story outline JSON 路径")
    parser.add_argument("--stage", default="发展", help="要扩写的 stage 名称，默认‘发展’")
    parser.add_argument("--sample-count", type=int, default=5, help="生成多少条候选事件线")
    parser.add_argument("--target-event-count", type=int, default=10, help="每条候选事件线的目标事件数")
    parser.add_argument("--model-name", default="gemini-2.5-pro", help="模型名")
    parser.add_argument("--temperature", type=float, default=0.9, help="采样温度")
    parser.add_argument("--max-new-tokens", type=int, default=8192, help="最大生成 token 数")
    parser.add_argument("--output", help="输出 JSON 路径；不传则自动落到 output/stage_eventline_sampler")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_stage_eventlines(
        outline_path=args.outline_path,
        stage_name=args.stage,
        sample_count=args.sample_count,
        target_event_count=args.target_event_count,
        model_name=args.model_name,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    output_path = Path(args.output).expanduser().resolve() if args.output else build_output_path(DEFAULT_OUTPUT_DIR, args.stage)
    save_json(output_path, result)
    print(f"已生成 {len(result['candidates'])} 条 `{args.stage}` stage 候选事件线")
    print(f"结果已写入：{output_path}")


if __name__ == "__main__":
    main()
