from __future__ import annotations

import importlib.util
import json
import re
import time
from pathlib import Path
from typing import Any

try:
    import json_repair
except Exception:  # noqa: BLE001
    json_repair = None


def _load_query_llm():
    module_path = Path(__file__).resolve().parents[2] / "service/shortdrama_by_fiction/tools/llm_services.py"
    spec = importlib.util.spec_from_file_location("drama_eval_llm_services", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载项目 LLM 客户端模块：{module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.query_llm


_query_llm = None


def get_query_llm():
    global _query_llm
    if _query_llm is None:
        _query_llm = _load_query_llm()
    return _query_llm


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.S)


def _extract_json_candidate(text: str) -> str:
    raw = text.strip()
    if not raw:
        raise ValueError("LLM 返回为空，无法解析 JSON")
    fenced = _JSON_FENCE.search(raw)
    if fenced:
        return fenced.group(1)
    start_object = raw.find("{")
    end_object = raw.rfind("}")
    start_array = raw.find("[")
    end_array = raw.rfind("]")
    candidates = []
    if start_object != -1 and end_object != -1 and end_object > start_object:
        candidates.append(raw[start_object : end_object + 1])
    if start_array != -1 and end_array != -1 and end_array > start_array:
        candidates.append(raw[start_array : end_array + 1])
    if not candidates:
        return raw
    return max(candidates, key=len)


def parse_json_response(text: str) -> Any:
    candidate = _extract_json_candidate(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        if json_repair is not None:
            return json_repair.loads(candidate)
        raise


def query_json(
    messages: list[dict[str, str]],
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.2,
    max_new_tokens: int = 8192,
    retries: int = 3,
    retry_sleep_seconds: float = 1.0,
) -> tuple[Any, str]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = get_query_llm()(
                messages,
                model_name=model_name,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                enable_stream=False,
                return_think=False,
            )
            parsed = parse_json_response(str(response))
            return parsed, str(response)
        except Exception as error:  # noqa: BLE001
            last_error = error
            if attempt < retries - 1:
                time.sleep(retry_sleep_seconds * (attempt + 1))
    raise RuntimeError(f"LLM JSON 调用失败：{last_error}") from last_error
