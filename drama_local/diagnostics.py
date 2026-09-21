"""Per-process diagnostic artifacts. Unique directories keep concurrent calls separate."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import uuid

RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]


def save_event(kind, *, text=None, **metadata):
    root = os.environ.get('DRAMA_OUTPUT_DIR')
    if not root:
        return None
    directory = Path(root) / 'diagnostics' / RUN_ID / kind / (
        str(time.time_ns()) + '_' + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    record = {'time_utc': datetime.now(timezone.utc).isoformat(), 'kind': kind, **metadata}
    (directory / 'record.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    if text is not None:
        (directory / 'output.txt').write_text(text if isinstance(text,str) else json.dumps(text,ensure_ascii=False,indent=2,default=str), encoding='utf-8')
    return str(directory)


def record_skip(episode_number, reason, service, *, parsed_output=None, partial_script=''):
    """Always retain the actual last output, even if JSON parsing returned {}."""
    raw = getattr(service, 'last_output', None)
    trace = getattr(service, 'last_trace_dir', None)
    if raw is None and trace:
        for name in ('output.txt', 'http_response.txt'):
            path = Path(trace) / name
            if path.exists():
                raw = path.read_text(encoding='utf-8')
                break
    return save_event('skipped_episodes', episode_number=episode_number, reason=reason,
                      last_call=trace, last_error=getattr(service, 'last_error', None),
                      parsed_output=parsed_output, partial_script=partial_script,
                      text=raw if raw is not None else 'No model text returned; see last_error and last_call.')
