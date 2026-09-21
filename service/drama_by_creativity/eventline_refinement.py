"""把故事总纲中的粗粒度 beats 用 v9 局部树搜索扩写为详细事件线。"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from drama_local.runtime import logger

from tools.stage_eventline_local_tree_v9 import generate_stage_eventline_with_local_tree_v9


@dataclass(frozen=True)
class EventlineRefinementConfig:
    """与参考产物 stage_eventline_pipeline_v9_max4 对齐的默认参数。"""

    stages: tuple[str, ...] = ("开端", "发展", "高潮", "结局")
    branch_factor: int = 4
    keep_top_k: int = 2
    local_max_depth: int = 4
    max_events_to_anchor: int = 4
    max_local_chains_per_anchor: int = 4
    parallel_workers: int = 32
    mutation_probability: float = 0.25
    model_name: str = "qwen-local"
    temperature: float = 0.9
    max_new_tokens: int = 32768


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def eventline_refinement_enabled() -> bool:
    """默认开启；排障或复跑旧链路时可通过环境变量显式关闭。"""

    return _env_bool("DRAMA_EVENTLINE_REFINEMENT_ENABLED", True)


def parse_refinement_stages(value: str) -> tuple[str, ...]:
    """支持逗号（中英文）或空格分隔，保留调用者指定的顺序。"""
    stages = tuple(part for part in re.split(r"[,，\s]+", value.strip()) if part)
    if not stages:
        raise ValueError("细化阶段不能为空；关闭细化请设置 DRAMA_EVENTLINE_REFINEMENT_ENABLED=0")
    if len(stages) != len(set(stages)):
        raise ValueError(f"细化阶段不能重复：{list(stages)}")
    return stages


def _load_config_from_env() -> EventlineRefinementConfig:
    stages = parse_refinement_stages(os.environ.get("DRAMA_EVENTLINE_STAGES", "开端,发展,高潮,结局"))
    return EventlineRefinementConfig(
        stages=stages,
        branch_factor=int(os.environ.get("DRAMA_EVENTLINE_BRANCH_FACTOR", "4")),
        keep_top_k=int(os.environ.get("DRAMA_EVENTLINE_KEEP_TOP_K", "2")),
        local_max_depth=int(os.environ.get("DRAMA_EVENTLINE_LOCAL_MAX_DEPTH", "4")),
        max_events_to_anchor=int(os.environ.get("DRAMA_EVENTLINE_MAX_EVENTS_TO_ANCHOR", "4")),
        max_local_chains_per_anchor=int(
            os.environ.get("DRAMA_EVENTLINE_MAX_LOCAL_CHAINS_PER_ANCHOR", "4")
        ),
        parallel_workers=int(os.environ.get("DRAMA_EVENTLINE_PARALLEL_WORKERS", "32")),
        mutation_probability=float(os.environ.get("DRAMA_EVENTLINE_MUTATION_PROBABILITY", "0.25")),
        model_name=os.environ.get("DRAMA_LLM_MODEL", "qwen-local"),
        temperature=float(os.environ.get("DRAMA_EVENTLINE_TEMPERATURE", "0.9")),
        max_new_tokens=int(os.environ.get("DRAMA_EVENTLINE_MAX_NEW_TOKENS", "32768")),
    )


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=str)


def _safe_stage_name(stage: str) -> str:
    return re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", stage).strip("_") or "stage"


def _replace_stage_events(outline: dict[str, Any], stage_name: str, events: list[str]) -> None:
    for stage in outline.get("framework", []):
        if isinstance(stage, dict) and str(stage.get("stage", "")).strip() == stage_name:
            stage["event_list"] = [{"event": event} for event in events]
            return
    raise ValueError(f"故事总纲中未找到待细化阶段：{stage_name}")


def _available_stage_names(outline: dict[str, Any]) -> list[str]:
    return [
        str(stage.get("stage", "")).strip()
        for stage in outline.get("framework", [])
        if isinstance(stage, dict) and str(stage.get("stage", "")).strip()
    ]


def _refine_story_outline_sync(
    outline: dict[str, Any],
    config: EventlineRefinementConfig,
    artifact_dir: Path,
) -> dict[str, Any]:
    refined = copy.deepcopy(outline)
    available_stages = _available_stage_names(refined)
    stages = list(config.stages)
    if not stages or len(stages) != len(set(stages)):
        raise ValueError(f"细化阶段须非空且不能重复：{stages}")
    missing = [stage for stage in stages if stage not in available_stages]
    if missing:
        raise ValueError(f"故事总纲不含待细化阶段 {missing}；实际阶段为 {available_stages}")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    _save_json(artifact_dir / "outline_before_refinement.json", refined)

    trope_bank: list[dict[str, str]] | None = None
    stage_records: list[dict[str, Any]] = []
    for stage_index, stage_name in enumerate(stages, start=1):
        safe_stage = _safe_stage_name(stage_name)
        stage_input = artifact_dir / f"outline_before_{stage_index:02d}_{safe_stage}.json"
        stage_result = artifact_dir / f"v9_result_{stage_index:02d}_{safe_stage}.json"
        stage_after = artifact_dir / f"outline_after_{stage_index:02d}_{safe_stage}.json"
        _save_json(stage_input, refined)

        result = generate_stage_eventline_with_local_tree_v9(
            outline_path=stage_input,
            stage_name=stage_name,
            branch_factor=config.branch_factor,
            keep_top_k=config.keep_top_k,
            local_max_depth=config.local_max_depth,
            max_events_to_anchor=config.max_events_to_anchor,
            max_local_chains_per_anchor=config.max_local_chains_per_anchor,
            parallel_workers=config.parallel_workers,
            mutation_probability=config.mutation_probability,
            model_name=config.model_name,
            temperature=config.temperature,
            max_new_tokens=config.max_new_tokens,
            output_path=stage_result,
            initial_trope_bank=trope_bank,
        )
        final_event_line = result.get("final_event_line", [])
        if not isinstance(final_event_line, list) or not final_event_line:
            raise RuntimeError(f"v9 未能为阶段 {stage_name} 生成有效 final_event_line")
        normalized_events = [str(event).strip() for event in final_event_line if str(event).strip()]
        if not normalized_events:
            raise RuntimeError(f"v9 为阶段 {stage_name} 返回了空事件线")

        _replace_stage_events(refined, stage_name, normalized_events)
        _save_json(stage_after, refined)
        result_bank = result.get("trope_bank", [])
        if isinstance(result_bank, list) and result_bank:
            trope_bank = result_bank
        bank_path = artifact_dir / f"trope_bank_after_{stage_index:02d}_{safe_stage}.json"
        _save_json(bank_path, trope_bank or [])
        stage_records.append(
            {
                "stage": stage_name,
                "event_count": len(normalized_events),
                "input": str(stage_input),
                "result": str(stage_result),
                "output": str(stage_after),
                "trope_bank_size": len(trope_bank or []),
                "trope_bank_path": str(bank_path),
            }
        )

    _save_json(artifact_dir / "outline_refined_final.json", refined)
    _save_json(
        artifact_dir / "manifest.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "algorithm": "stage_eventline_local_tree_v9",
            "config": asdict(config),
            "stages": stage_records,
        },
    )
    return refined


async def _run_refinement_in_worker(
    outline: dict[str, Any],
    config: EventlineRefinementConfig,
    artifact_dir: Path,
) -> dict[str, Any]:
    """兼容 drama 运行环境的线程桥接。

    该环境的默认 asyncio executor 不能可靠唤醒等待方，因此不使用
    ``asyncio.to_thread``。v9 内部同步客户端需要在独立线程里创建自己的事件循环。
    """

    result_holder: list[dict[str, Any]] = []
    error_holder: list[BaseException] = []
    finished = threading.Event()

    def worker() -> None:
        try:
            result_holder.append(_refine_story_outline_sync(outline, config, artifact_dir))
        except BaseException as error:  # noqa: BLE001 - 必须把工作线程异常送回主协程
            error_holder.append(error)
        finally:
            finished.set()

    threading.Thread(target=worker, name="drama-eventline-v9", daemon=True).start()
    while not finished.is_set():
        await asyncio.sleep(0.1)
    if error_holder:
        raise error_holder[0]
    if not result_holder:
        raise RuntimeError("故事事件线 v9 工作线程结束但未返回结果")
    return result_holder[0]


async def refine_story_outline_eventlines(ctx, outline: dict[str, Any]) -> dict[str, Any]:
    """在线流程适配器：在线程中运行同步 v9，避免阻塞当前 asyncio 事件循环。"""

    if not eventline_refinement_enabled():
        logger.info_context(ctx, "故事事件线 v9 细化已通过环境变量关闭，沿用原始故事总纲。")
        return outline
    if not isinstance(outline, dict):
        raise TypeError(f"故事总纲必须是 JSON 对象，实际为 {type(outline).__name__}")

    config = _load_config_from_env()
    output_root = os.environ.get("DRAMA_OUTPUT_DIR")
    if output_root:
        artifact_dir = Path(output_root).expanduser().resolve() / "03_story_outline" / "eventline_refinement_v9"
        logger.info_context(
            ctx,
            f"开始执行故事事件线 v9 细化：stages={list(config.stages)}, "
            f"max_events={config.max_events_to_anchor}, max_chains={config.max_local_chains_per_anchor}；"
            f"产物目录={artifact_dir}",
        )
        return await _run_refinement_in_worker(outline, config, artifact_dir)

    # RPC 未配置实时输出目录时仍执行细化，但只使用临时文件承接 v9 的文件接口。
    with tempfile.TemporaryDirectory(prefix="drama_eventline_v9_") as temp_dir:
        logger.info_context(ctx, "开始执行故事事件线 v9 细化（当前未配置 DRAMA_OUTPUT_DIR，使用临时产物目录）。")
        return await _run_refinement_in_worker(
            outline,
            config,
            Path(temp_dir),
        )
