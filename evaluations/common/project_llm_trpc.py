"""TRPC proxy LLM adapter – same interface as project_llm.query_json but uses
the internal trpc_mllm proxy (llm_proxy_sdk.LLM) instead of direct HTTP VenusClient.

Requires: drama conda env (provides llm_proxy_sdk, trpc_mllm_mllm, trpc_naming_polaris)
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

try:
    import json_repair
except Exception:
    json_repair = None

from jinja2 import Template
from llm_proxy_sdk import LLM
from trpc.context import Context

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


def _build_llm(model_name: str, max_new_tokens: int, temperature: float) -> LLM:
    params: dict[str, Any] = {
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 50,
        "max_tokens": max_new_tokens,
    }
    return LLM(model=model_name, template=Template("{{text}}"), **params)


def query_json(
    messages: list[dict[str, str]],
    model_name: str = "venus_model://876:gemini-2.5-pro",
    temperature: float = 0.2,
    max_new_tokens: int = 8192,
    retries: int = 3,
    retry_sleep_seconds: float = 1.0,
) -> tuple[Any, str]:
    if not model_name.startswith("venus_model://"):
        model_name = f"venus_model://876:{model_name}"

    text = messages[-1]["content"] if messages else ""
    if len(messages) > 1:
        text = "\n\n".join(m["content"] for m in messages if m["content"].strip())

    ctx = Context()
    llm = _build_llm(model_name, max_new_tokens, temperature)

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            rsp = asyncio.run(llm.request(ctx, {"text": text}, f"eval_{int(time.time() * 1000)}"))
            response = str(rsp.response)
            parsed = parse_json_response(response)
            return parsed, response
        except Exception as error:
            last_error = error
            if attempt < retries - 1:
                time.sleep(retry_sleep_seconds * (attempt + 1))
    raise RuntimeError(f"LLM JSON 调用失败：{last_error}") from last_error
