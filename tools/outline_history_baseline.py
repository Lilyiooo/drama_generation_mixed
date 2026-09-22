#!/usr/bin/env python3
"""Three-run direct outline-to-script baseline with complete prior-script context."""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone

import requests


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(os.environ.get("SCRIPT_ROOT", str(REPOSITORY_ROOT))).expanduser().resolve()
SOURCE_ROOT = Path(
    os.environ.get("BASELINE_SOURCE_ROOT", str(SCRIPT_ROOT / "output/full_scene_gate_ab_v1"))
).expanduser().resolve()
ROOT = Path(
    os.environ.get(
        "BASELINE_OUTPUT_ROOT",
        str(SCRIPT_ROOT / "output/outline_direct_all_history_baseline_v1"),
    )
).expanduser().resolve()
EVALUATOR_ROOT = Path(
    os.environ.get(
        "DRAMA_EVALUATOR_ROOT",
        str(REPOSITORY_ROOT / "evaluation/drama_evaluator_logic_quality"),
    )
).expanduser().resolve()
MODEL_PATH = Path(
    os.environ.get("GENERATION_MODEL_PATH", str(REPOSITORY_ROOT / "models/Qwen3.6-27B"))
).expanduser().resolve()
RUNS = ("R01", "R02", "R03")
SOURCE_ARMS = ("control_hybrid", "candidate_gate")
EPISODES = 60
MODEL = "Qwen3.6-27B"
MAX_CONTEXT_TOKENS = 131072
MAX_OUTPUT_TOKENS = 8192
RUN_SEEDS = {"R01": 360100, "R02": 360200, "R03": 360300}

SYSTEM_PROMPT = (
    "你是一位资深中文短番编剧。你的唯一任务是直接依据给定的世界观、人物设定、"
    "故事总纲、本集分集大纲以及此前所有已生成剧集原文，创作本集完整剧本。"
    "此前剧集是不可篡改的已发生事实。只输出最终剧本正文，不解释，不输出Markdown。"
)

PROMPT_VERSION = "outline-direct-all-history-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_text(text: str) -> str:
    return sha_bytes(text.encode("utf-8"))


def canonical_sha(value) -> str:
    return sha_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def extract_string_assignment(path: Path, variable: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == variable for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise ValueError(f"Cannot find string assignment {variable} in {path}")


def source_asset_paths() -> list[Path]:
    return [SOURCE_ROOT / arm / run / "generation_assets.json" for arm in SOURCE_ARMS for run in RUNS]


def validate_assets(assets: dict) -> None:
    for key in ("world_view", "role_info", "story_outline"):
        if not isinstance(assets.get(key), str) or not assets[key].strip():
            raise ValueError(f"Invalid generation asset: {key}")
    outlines = assets.get("episode_outlines")
    if not isinstance(outlines, list) or len(outlines) != EPISODES:
        raise ValueError(f"Expected {EPISODES} episode outlines")
    for index, row in enumerate(outlines):
        if not isinstance(row, dict) or row.get("episode_id") != index:
            raise ValueError(f"Invalid outline at episode {index + 1}")
        if not isinstance(row.get("content"), str) or not row["content"].strip():
            raise ValueError(f"Empty outline at episode {index + 1}")


def prepare() -> None:
    paths = source_asset_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source assets: " + ", ".join(missing))
    blobs = [path.read_bytes() for path in paths]
    hashes = [sha_bytes(blob) for blob in blobs]
    if len(set(hashes)) != 1:
        raise ValueError("The six source trajectories do not use identical generation assets")
    assets = json.loads(blobs[0])
    validate_assets(assets)

    format_source = SCRIPT_ROOT / "service/drama_by_creativity/prompts/script_format.py"
    script_format = extract_string_assignment(format_source, "SCRIPT_FORMAT_PROMPT")
    protocol = {
        "protocol_version": PROMPT_VERSION,
        "description": "Direct episode-outline to script; every episode receives all prior baseline scripts verbatim.",
        "source_experiment": str(SOURCE_ROOT),
        "canonical_source": str(paths[-3]),
        "source_asset_sha256": hashes[0],
        "verified_identical_sources": [str(path) for path in paths],
        "episodes": EPISODES,
        "runs": list(RUNS),
        "run_seeds": RUN_SEEDS,
        "model": MODEL,
        "generation": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 50,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "enable_thinking": False,
            "max_context_tokens": MAX_CONTEXT_TOKENS,
        },
        "inputs_each_episode": [
            "world_view",
            "role_info",
            "story_outline",
            "current_episode_outline",
            "all_previous_baseline_episode_scripts_verbatim",
        ],
        "excluded": [
            "scene_outline_agent",
            "narrative_memory",
            "state_lifecycle_memory",
            "obligation_memory",
            "hybrid_retrieval",
            "strategy_cards",
            "state_gate",
            "plot_library_retrieval",
            "post_generation_rewrite_or_polish",
        ],
        "system_prompt_sha256": sha_text(SYSTEM_PROMPT),
        "script_format_source": str(format_source),
        "script_format_source_sha256": sha_bytes(format_source.read_bytes()),
        "frozen_script_format_sha256": sha_text(script_format),
    }
    manifest = ROOT / "protocol.json"
    if manifest.exists() and read_json(manifest) != protocol:
        raise ValueError(f"Protocol changed; use a new experiment root instead of overwriting {ROOT}")

    ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(ROOT / "frozen_generation_assets.json", assets)
    atomic_text(ROOT / "frozen_script_format.txt", script_format)
    atomic_json(manifest, protocol)
    for run in RUNS:
        for directory in ("05_drama", "request_metadata", "attempts"):
            (ROOT / run / directory).mkdir(parents=True, exist_ok=True)
    print(f"prepared={ROOT}")
    print(f"asset_sha256={hashes[0]} episodes={len(assets['episode_outlines'])}")


def require_prepared() -> tuple[dict, dict, str]:
    protocol_path = ROOT / "protocol.json"
    assets_path = ROOT / "frozen_generation_assets.json"
    format_path = ROOT / "frozen_script_format.txt"
    if not (protocol_path.is_file() and assets_path.is_file() and format_path.is_file()):
        raise RuntimeError("Experiment is not prepared; run prepare first")
    protocol = read_json(protocol_path)
    assets = read_json(assets_path)
    script_format = format_path.read_text(encoding="utf-8")
    validate_assets(assets)
    if sha_text(script_format) != protocol["frozen_script_format_sha256"]:
        raise ValueError("Frozen script format hash mismatch")
    if sha_bytes(source_asset_paths()[-3].read_bytes()) != protocol["source_asset_sha256"]:
        raise ValueError("Canonical source assets changed after baseline preparation")
    return protocol, assets, script_format


def episode_path(run: str, episode: int) -> Path:
    return ROOT / run / "05_drama" / f"episode_{episode:02d}.json"


def metadata_path(run: str, episode: int) -> Path:
    return ROOT / run / "request_metadata" / f"episode_{episode:02d}.json"


def read_episode(run: str, episode: int) -> str:
    path = episode_path(run, episode)
    row = read_json(path)
    if row.get("episode_id") != episode - 1:
        raise ValueError(f"episode_id mismatch: {path}")
    content = row.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"Empty episode: {path}")
    return content


def render_prompt(assets: dict, script_format: str, episode: int, history: list[str]) -> str:
    history_text = "暂无（这是第1集）。"
    if history:
        history_text = "\n\n".join(
            f"<episode index=\"{index}\">\n{text}\n</episode>"
            for index, text in enumerate(history, 1)
        )
    outline = assets["episode_outlines"][episode - 1]["content"].strip()
    frozen_format = script_format.replace("{{EpisodeIndex}}", str(episode)).strip()
    return f"""请直接创作第{episode}集完整剧本。

## 基线方法约束
1. 这是“分集大纲直接生成剧本”基线，只进行这一次写作调用；不要先生成场次大纲、记忆、分析、检查表或修改稿。
2. 下方“此前全部剧集原文”按顺序完整提供。它们都是本故事已经发生的事实；本集不得改变人物已知信息、关系、伤势、物品归属、资源、时空、能力边界和未完伏笔。
3. 严格落实本集大纲，不提前消费后续情节，不重复此前已经完成的核心事件。
4. 主线清晰、节奏迅猛、开场快速进入冲突；每集结尾设置自然钩子。
5. 本集写1至3场，总字数控制在1000至1400个中文字符左右；台词占70%以上。
6. 只输出最终中文剧本正文，不要分析、前言、总结、代码围栏或Markdown标题。

## 世界观
{assets['world_view'].strip()}

## 主要人物设定
{assets['role_info'].strip()}

## 故事总纲
{assets['story_outline'].strip()}

## 本集分集大纲
{outline}

## 此前全部剧集原文（第1集至第{episode - 1}集，逐字提供，不是摘要）
{history_text}

## 输出格式
{frozen_format}

再次确认：现在只输出第{episode}集的最终剧本正文。"""


class TokenCounter:
    def __init__(self):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_PATH), trust_remote_code=True, local_files_only=True
        )
        self.lock = threading.Lock()

    def count(self, messages: list[dict]) -> int:
        with self.lock:
            tokens = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
            )
        return len(tokens)


def verify_server(base_url: str, model: str) -> None:
    with requests.Session() as session:
        session.trust_env = False
        response = session.get(base_url.rstrip("/") + "/models", timeout=10)
    response.raise_for_status()
    names = [row.get("id") for row in response.json().get("data", [])]
    if model not in names:
        raise RuntimeError(f"{base_url} serves {names}, not {model}")


def post_completion(base_url: str, payload: dict, run: str, episode: int) -> tuple[str, dict, int]:
    headers = {}
    api_key = os.environ.get("DRAMA_LLM_API_KEY")
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    last_error = None
    for attempt in range(1, 4):
        started = time.monotonic()
        trace = {
            "run": run,
            "episode": episode,
            "attempt": attempt,
            "started_at": utc_now(),
            "payload_sha256": canonical_sha(payload),
        }
        try:
            with requests.Session() as session:
                session.trust_env = False
                response = session.post(
                    base_url.rstrip("/") + "/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=(10, 1200),
                )
            trace["http_status"] = response.status_code
            response.raise_for_status()
            data = response.json()
            choice = data["choices"][0]
            content = choice["message"]["content"]
            if choice.get("finish_reason") == "length":
                raise ValueError("Model output was truncated")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Model returned empty script")
            trace.update(
                status="success",
                elapsed_seconds=time.monotonic() - started,
                finish_reason=choice.get("finish_reason"),
                usage=data.get("usage", {}),
                response_sha256=sha_text(content),
                response_characters=len(content),
            )
            atomic_json(ROOT / run / "attempts" / f"episode_{episode:02d}_attempt_{attempt}.json", trace)
            return content, data.get("usage", {}), attempt
        except Exception as error:
            last_error = error
            trace.update(
                status="error",
                elapsed_seconds=time.monotonic() - started,
                error=f"{type(error).__name__}: {error}",
            )
            atomic_json(ROOT / run / "attempts" / f"episode_{episode:02d}_attempt_{attempt}.json", trace)
            if attempt < 3:
                time.sleep(5 * attempt)
    raise RuntimeError(f"{run}/E{episode:02d} failed after 3 attempts: {last_error}")


def verify_existing_prefix(run: str, assets: dict, script_format: str) -> list[str]:
    history: list[str] = []
    found_gap = False
    for episode in range(1, EPISODES + 1):
        output = episode_path(run, episode)
        metadata = metadata_path(run, episode)
        if not output.exists() and not metadata.exists():
            found_gap = True
            continue
        if found_gap:
            raise ValueError(f"Non-contiguous output in {run}: episode {episode} exists after a gap")
        if not (output.is_file() and metadata.is_file()):
            raise ValueError(f"Incomplete output/metadata pair for {run}/E{episode:02d}")
        content = read_episode(run, episode)
        record = read_json(metadata)
        prompt = render_prompt(assets, script_format, episode, history)
        expected_history_hashes = [sha_text(text) for text in history]
        if record.get("prompt_sha256") != sha_text(prompt):
            raise ValueError(f"Prompt hash mismatch for existing {run}/E{episode:02d}")
        if record.get("history_episode_sha256") != expected_history_hashes:
            raise ValueError(f"History provenance mismatch for existing {run}/E{episode:02d}")
        if record.get("response_sha256") != sha_text(content):
            raise ValueError(f"Response hash mismatch for existing {run}/E{episode:02d}")
        history.append(content)
    return history


def generate_run(run: str, limit: int | None, base_url: str, counter: TokenCounter) -> dict:
    _, assets, script_format = require_prepared()
    lock_path = ROOT / run / ".generate.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"{run} is already generating") from error
        history = verify_existing_prefix(run, assets, script_format)
        start_episode = len(history) + 1
        stop_episode = EPISODES if limit is None else min(EPISODES, len(history) + limit)
        if start_episode > stop_episode:
            return {"run": run, "generated": 0, "complete": len(history) == EPISODES}
        generated = 0
        for episode in range(start_episode, stop_episode + 1):
            prompt = render_prompt(assets, script_format, episode, history)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            prompt_tokens = counter.count(messages)
            if prompt_tokens + MAX_OUTPUT_TOKENS > MAX_CONTEXT_TOKENS:
                raise ValueError(
                    f"{run}/E{episode:02d} needs {prompt_tokens}+{MAX_OUTPUT_TOKENS} tokens; "
                    "refusing to truncate the required full history"
                )
            seed = RUN_SEEDS[run] + episode
            payload = {
                "model": MODEL,
                "messages": messages,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 50,
                "max_tokens": MAX_OUTPUT_TOKENS,
                "seed": seed,
                "chat_template_kwargs": {"enable_thinking": False},
                "stream": False,
            }
            content, usage, attempts = post_completion(base_url, payload, run, episode)
            output = {"episode_id": episode - 1, "content": content}
            atomic_json(episode_path(run, episode), output)
            history_hashes = [sha_text(text) for text in history]
            record = {
                "protocol_version": PROMPT_VERSION,
                "run": run,
                "episode": episode,
                "created_at": utc_now(),
                "model": MODEL,
                "seed": seed,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 50,
                "max_tokens": MAX_OUTPUT_TOKENS,
                "enable_thinking": False,
                "history_episode_count": len(history),
                "history_episode_numbers": list(range(1, episode)),
                "history_episode_sha256": history_hashes,
                "history_characters": sum(len(text) for text in history),
                "history_bundle_sha256": canonical_sha(history_hashes),
                "prompt_characters": len(SYSTEM_PROMPT) + len(prompt),
                "prompt_tokens": prompt_tokens,
                "prompt_sha256": sha_text(prompt),
                "messages_sha256": canonical_sha(messages),
                "response_sha256": sha_text(content),
                "response_characters": len(content),
                "usage": usage,
                "attempts": attempts,
                "base_url": base_url,
            }
            atomic_json(metadata_path(run, episode), record)
            history.append(content)
            generated += 1
            print(
                f"completed={run}/E{episode:02d} history={episode - 1} "
                f"prompt_tokens={prompt_tokens} output_chars={len(content)}",
                flush=True,
            )
        return {"run": run, "generated": generated, "complete": len(history) == EPISODES}


def run_generation(runs: list[str], workers: int, limit: int | None, execute_api: bool) -> None:
    prepare()
    if not execute_api:
        print("Dry run only. Add --execute-api to call Qwen3.6-27B.")
        show_status()
        return
    base_url = os.environ.get("DRAMA_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    verify_server(base_url, MODEL)
    # transformers uses lazy module imports. Loading once on the main thread avoids
    # an import race when three trajectory workers start simultaneously.
    counter = TokenCounter()
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(runs))) as executor:
        futures = {
            executor.submit(generate_run, run, limit, base_url, counter): run for run in runs
        }
        for future in concurrent.futures.as_completed(futures):
            run = futures[future]
            try:
                print(json.dumps(future.result(), ensure_ascii=False), flush=True)
            except Exception as error:
                failures.append((run, error))
                print(json.dumps({"run": run, "error": str(error)}, ensure_ascii=False), flush=True)
    show_status()
    if failures:
        raise RuntimeError(f"{len(failures)} run(s) failed; rerun the same command to resume")


def run_state(run: str) -> dict:
    completed = 0
    valid_history = False
    error = None
    try:
        if (ROOT / "protocol.json").is_file():
            _, assets, script_format = require_prepared()
            completed = len(verify_existing_prefix(run, assets, script_format))
            valid_history = True
        else:
            completed = sum(episode_path(run, episode).is_file() for episode in range(1, EPISODES + 1))
    except Exception as exc:
        error = str(exc)
    score_path = ROOT / run / "drama_evaluations_logic_quality_qwen38_v24/full_60_episodes/scores.json"
    return {
        "run": run,
        "completed_episodes": completed,
        "complete": completed == EPISODES and valid_history,
        "history_audit": valid_history,
        "scored": score_path.is_file(),
        "error": error,
    }


def show_status() -> None:
    states = [run_state(run) for run in RUNS]
    for state in states:
        suffix = f" error={state['error']}" if state["error"] else ""
        print(
            f"{state['run']}: episodes={state['completed_episodes']}/{EPISODES} "
            f"history_audit={state['history_audit']} scored={state['scored']}{suffix}"
        )


def export_run(run: str) -> Path:
    state = run_state(run)
    if not state["complete"]:
        raise ValueError(f"Finish generation first: {run} ({state['completed_episodes']}/{EPISODES})")
    target = ROOT / run / "drama_evaluator_logic_quality_input/full_60_episodes.txt"
    subprocess.run(
        [
            sys.executable,
            str(EVALUATOR_ROOT / "tools/export_pipeline_script.py"),
            str(ROOT / run),
            str(target),
        ],
        check=True,
    )
    return target


def evaluate_run(run: str, execute_api: bool) -> dict:
    script = export_run(run)
    output = ROOT / run / "drama_evaluations_logic_quality_qwen38_v24/full_60_episodes"
    command = [
        sys.executable,
        str(EVALUATOR_ROOT / "evaluate_multi_agent.py"),
        str(script),
        "--output-dir",
        str(output),
        "--model",
        "Qwen3.8-27B",
        "--context-mode",
        "direct",
        "--direct-char-limit",
        "300000",
        "--chunk-chars",
        "100000",
        "--review-workers",
        "2",
        "--audit-workers",
        "2",
        "--arbitration-workers",
        "2",
        "--max-output-tokens",
        "24576",
        "--timeout",
        "1200",
        "--retries",
        "3",
        "--parse-retries",
        "3",
    ]
    if not execute_api:
        command.append("--dry-run")
    log = ROOT / "logs" / f"evaluate_{run}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=EVALUATOR_ROOT, stdout=handle, stderr=subprocess.STDOUT)
    return {"run": run, "returncode": completed.returncode, "log": str(log)}


def run_evaluation(runs: list[str], workers: int, execute_api: bool) -> None:
    if execute_api:
        base_url = os.environ.get("DRAMA_EVAL_BASE_URL", "http://127.0.0.1:8001/v1")
        verify_server(base_url, "Qwen3.8-27B")
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(runs))) as executor:
        futures = {executor.submit(evaluate_run, run, execute_api): run for run in runs}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if result["returncode"]:
                failures.append(result)
    show_status()
    if failures:
        raise RuntimeError("Some evaluations failed; successful outputs are retained; rerun to resume")
    if execute_api:
        write_report()


def score_pair(path: Path) -> tuple[float, float]:
    payload = read_json(path)["scores"]
    return (
        float(payload["剧本逻辑总分"]["final_score"]),
        float(payload["剧本质量最终总分"]["final_score"]),
    )


def write_report() -> None:
    rows = []
    for run in RUNS:
        baseline_path = ROOT / run / "drama_evaluations_logic_quality_qwen38_v24/full_60_episodes/scores.json"
        candidate_path = SOURCE_ROOT / "candidate_gate" / run / "drama_evaluations_logic_quality_qwen38_v24/full_60_episodes/scores.json"
        control_path = SOURCE_ROOT / "control_hybrid" / run / "drama_evaluations_logic_quality_qwen38_v24/full_60_episodes/scores.json"
        if not baseline_path.is_file():
            raise ValueError(f"Missing baseline scores: {baseline_path}")
        baseline = score_pair(baseline_path)
        candidate = score_pair(candidate_path) if candidate_path.is_file() else (None, None)
        control = score_pair(control_path) if control_path.is_file() else (None, None)
        rows.append(
            {
                "run": run,
                "baseline": {"logic": baseline[0], "quality": baseline[1]},
                "candidate_gate": {"logic": candidate[0], "quality": candidate[1]},
                "control_hybrid": {"logic": control[0], "quality": control[1]},
                "candidate_minus_baseline": {
                    "logic": None if candidate[0] is None else round(candidate[0] - baseline[0], 4),
                    "quality": None if candidate[1] is None else round(candidate[1] - baseline[1], 4),
                },
            }
        )
    baseline_logic = [row["baseline"]["logic"] for row in rows]
    baseline_quality = [row["baseline"]["quality"] for row in rows]
    deltas_logic = [row["candidate_minus_baseline"]["logic"] for row in rows]
    deltas_quality = [row["candidate_minus_baseline"]["quality"] for row in rows]
    report = {
        "experiment": str(ROOT),
        "generated_at": utc_now(),
        "comparison": "candidate_gate minus direct-outline/all-history baseline, paired by run",
        "runs": rows,
        "summary": {
            "baseline_logic_mean": statistics.mean(baseline_logic),
            "baseline_logic_sample_std": statistics.stdev(baseline_logic),
            "baseline_quality_mean": statistics.mean(baseline_quality),
            "baseline_quality_sample_std": statistics.stdev(baseline_quality),
            "candidate_minus_baseline_logic_mean": statistics.mean(deltas_logic),
            "candidate_minus_baseline_logic_sample_std": statistics.stdev(deltas_logic),
            "candidate_minus_baseline_quality_mean": statistics.mean(deltas_quality),
            "candidate_minus_baseline_quality_sample_std": statistics.stdev(deltas_quality),
        },
    }
    target = ROOT / "reports/baseline_vs_candidate_gate_v24.json"
    atomic_json(target, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report={target}")


def selected_runs(values: list[str] | None) -> list[str]:
    runs = values or list(RUNS)
    invalid = sorted(set(runs) - set(RUNS))
    if invalid:
        raise ValueError(f"Invalid runs: {invalid}")
    return list(dict.fromkeys(runs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("status")
    subparsers.add_parser("report")
    for action in ("generate", "evaluate"):
        child = subparsers.add_parser(action)
        child.add_argument("--run", action="append", choices=RUNS)
        child.add_argument("--workers", type=int, default=3)
        child.add_argument("--execute-api", action="store_true")
        if action == "generate":
            child.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "status":
        show_status()
    elif args.action == "report":
        write_report()
    elif args.action == "generate":
        if args.workers < 1 or args.workers > 3:
            parser.error("--workers must be between 1 and 3")
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        run_generation(selected_runs(args.run), args.workers, args.limit, args.execute_api)
    elif args.action == "evaluate":
        if args.workers < 1 or args.workers > 3:
            parser.error("--workers must be between 1 and 3")
        run_evaluation(selected_runs(args.run), args.workers, args.execute_api)


if __name__ == "__main__":
    main()
