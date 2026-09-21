"""vLLM Chat Completions adapter for the creativity pipeline.

Preserves the existing LLM(...).request(ctx, params, request_id).response API.
"""
import asyncio
import os
import json
import time
from pathlib import Path
from drama_local.diagnostics import save_event
from types import SimpleNamespace

import requests


class LLM:
    def __init__(self, *, model, system_prompt, temperature, top_p, top_k,
                 max_length, max_new_tokens, template):
        self.model = os.getenv("DRAMA_LLM_MODEL") or model
        self.url = os.getenv("DRAMA_LLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/") + "/chat/completions"
        self.system_prompt = system_prompt
        self.template = template
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        # max_length belonged to the old SDK; vLLM manages context length.
        limit = os.getenv("DRAMA_LLM_MAX_TOKENS")
        self.max_tokens = int(limit) if limit is not None else max_new_tokens
        if self.max_tokens is not None and self.max_tokens < 0:
            raise ValueError("DRAMA_LLM_MAX_TOKENS must be >= 0")
        self.timeout = float(os.getenv("DRAMA_LLM_TIMEOUT", "0"))
        if self.timeout < 0:
            raise ValueError("DRAMA_LLM_TIMEOUT must be >= 0")
        self.thinking = os.getenv("DRAMA_LLM_THINKING", "0").lower() in ("1", "true", "yes", "on")

    async def request(self, ctx, params, request_id):
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.template.render(**params)})
        # Keep blocking requests off the event loop used by the pipeline.
        metadata = {"request_id": request_id, "context": dict(getattr(ctx, "fields", {})),
                    "episode_number": next((params[k] for k in ("EpisodeNumber", "EpisodeIndex", "EpisodeIdx") if k in params), None),
                    "parameters": params}
        return await asyncio.to_thread(self.request_messages, messages, metadata)

    def request_messages(self, messages, metadata=None):
        """Synchronous entry for the eventline worker threads."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "chat_template_kwargs": {"enable_thinking": self.thinking},
            "stream": False,
        }
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        self.last_output = None
        self.last_error = None
        trace = save_event("model_calls", url=self.url, request=payload, metadata=metadata or {})
        self.last_trace_dir = trace
        if trace:
            Path(trace, "prompt.txt").write_text("\n\n".join(
                f"[{message.get('role', '')}]\n{message.get('content', '')}" for message in messages), encoding="utf-8")
        start = time.monotonic()
        error = None
        try:
            result = self._post(payload, trace)
            status = "success"
            return result
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            self.last_error = error
            raise
        finally:
            if trace:
                Path(trace, "status.json").write_text(json.dumps({
                    "status": status, "elapsed_seconds": time.monotonic() - start,
                    "error": error,
                }, ensure_ascii=False, indent=2), encoding="utf-8")

    def _post(self, payload, trace=None):
        headers = {}
        api_key = os.getenv("DRAMA_LLM_API_KEY")
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        with requests.Session() as session:
            session.trust_env = False
            response = session.post(self.url, json=payload, headers=headers,
                                    timeout=(5, self.timeout or None))
        # Save HTTP body before checking status or decoding, including malformed responses.
        if trace:
            Path(trace, "http_response.txt").write_text(response.text, encoding="utf-8")
        if not response.ok:
            raise RuntimeError(f"Local LLM HTTP {response.status_code}: {response.text[:2000]}")
        data = response.json()
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Local LLM response missing choices[0].message.content") from exc
        self.last_output = content
        if trace:
            Path(trace, "output.txt").write_text(
                content if isinstance(content, str) else json.dumps(content, ensure_ascii=False), encoding="utf-8")
        if choice.get("finish_reason") == "length":
            raise ValueError("Local LLM output truncated; increase DRAMA_LLM_MAX_TOKENS or shorten the input")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Local LLM returned empty content")
        return SimpleNamespace(response=content, usage=data.get("usage", {}))
