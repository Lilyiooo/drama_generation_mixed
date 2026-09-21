"""Local vLLM JSON interface used by v9 eventline refinement."""
import importlib.util
import os
from pathlib import Path
import time
from drama_local.diagnostics import save_event

from evaluations.common.project_llm import parse_json_response

# Load the transport without importing the RPC service package.
_path = Path(__file__).resolve().parents[2] / "service/drama_by_creativity/local_llm.py"
_spec = importlib.util.spec_from_file_location("eventline_local_llm", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def query_json(messages, model_name="qwen-local", temperature=0.2,
               max_new_tokens=8192, retries=3, retry_sleep_seconds=1.0):
    client = _module.LLM(
        model=os.getenv("DRAMA_LLM_MODEL") or "qwen-local",
        system_prompt="", temperature=temperature, top_p=0.95, top_k=50,
        max_length=None, max_new_tokens=max_new_tokens, template=None,
    )
    last_error = None
    for attempt in range(retries):
        try:
            text = client.request_messages(messages).response
            return parse_json_response(text), text
        except Exception as exc:
            save_event("eventline_json_retry", attempt=attempt+1, reason=str(exc),
                       source=getattr(client, "last_trace_dir", None),
                       next_action="regenerate" if attempt+1 < retries else "raise")
            last_error = exc
            if attempt < retries - 1:
                time.sleep(retry_sleep_seconds * (attempt + 1))
    raise RuntimeError(f"Local LLM JSON request failed: {last_error}") from last_error
