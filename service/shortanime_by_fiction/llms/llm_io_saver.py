"""LLM 输入输出持久化模块

职责单一：将每次真实调用（非 cache hit）的 LLM 输入输出保存到 JSONL 文件。
对外只暴露 save_llm_io 一个函数，内部异常静默处理，绝不影响主流程。

存储路径：<project_root>/.cache/llm_io_logs/{stage}_{prompt_name}/YYYY-MM-DD.jsonl
每行一条 JSON 记录。
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def _get_log_dir() -> Path:
    """与 LLMCache 保持一致，从 llms/ 目录向上找到 .cache，再定位到 llm_io_logs"""
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    log_dir = Path(base) / "../.cache" / "llm_io_logs"
    log_dir = log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def save_llm_io(
    *,
    request_key: str,
    model_name: str,
    stage: str,
    prompt_name: str,
    messages: list,
    think: Optional[str],
    answer: str,
    extra: Optional[dict] = None,
) -> None:
    """保存一次 LLM 真实调用的输入输出（cache hit 请勿调用此函数）。

    Args:
        request_key:  调用标识，如 "stage_prompt_name_debug_info"
        model_name:   模型名称
        stage:        任务阶段
        prompt_name:  prompt 名称
        messages:     发给 LLM 的完整 messages 列表
        think:        模型的 reasoning_content（可为空）
        answer:       模型的最终回复
        extra:        附加信息（如 story_info/generate_input），仅用于日志
    """
    try:
        input_str = json.dumps(messages, ensure_ascii=False)
        think_str = think or ""
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "request_key": request_key,
            "model_name": model_name,
            "stage": stage,
            "prompt_name": prompt_name,
            "extra": extra or {},
            "input_len": len(input_str),
            "think_len": len(think_str),
            "output_len": len(answer),
            "input": messages,
            "think": think_str,
            "output": answer,
        }

        log_dir = _get_log_dir() / f"{stage}_{prompt_name}"
        log_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = log_dir / f"{date_str}_{run_type}.jsonl"

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    except Exception:
        # 静默失败：日志保存出错不应中断业务主流程
        pass
