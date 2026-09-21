"""Run the fixed-outline ScriptPipeline experiment with optional NMF hybrid memory."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import requests


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from drama_local import models as pb
from drama_local.runtime import Context, LocalStore, ScriptContentFile, atomic_json
from service.drama_by_creativity.generate_episode_script import generate_episode_script
from service.drama_by_creativity.narrative_memory import NarrativeMemory
from service.drama_by_creativity.state_lifecycle_memory import StateLifecycleMemory, TENCENT_ROOT, assets_from_request, preserve_generation_assets, digest


DEFAULT_SOURCE = REPO / "output/full_pipeline_qwen36_1run"
DEFAULT_ROOT = REPO / "output/state_lifecycle_hybrid_v2"
ARMS = ("hybrid", "native_context")
EPISODES = 60
SOURCE_FILES = (
    "pipeline_result.json", "episode_outlines.json",
    "01_script_proposal/01_world_view.json", "01_script_proposal/02_role_setting.json",
    "04_future_map/future_map.json", "04_future_map/episode_contributions.json",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(source, root):
    marker = root / "experiment.json"
    if marker.exists():
        load_inputs(root)
        return
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Refusing to prepare inside a nonempty unregistered experiment: {root}")
    source_data = {name: read_json(source / name) for name in SOURCE_FILES}
    pipeline = source_data["pipeline_result.json"]
    story_outline = pipeline["story_outline"]["story_outline"]
    episode_outline = pipeline["episode_outline"]["episode_outline"]
    episodes = episode_outline["seasons"][0]["episodes"]
    if [item["episode_id"] for item in episodes] != list(range(EPISODES)):
        raise ValueError("Source must contain exactly 60 ordered episode outlines")
    generated_bible = pipeline["script_proposal"]["story_outline"]
    world = "\n\n".join(story_outline.get("world_building", [])) or "\n\n".join(generated_bible["world_building"])
    roles = "\n\n".join(story_outline.get("role_info", [])) or "\n\n".join(generated_bible["role_info"])
    if not isinstance(world, str) or world.strip() in {"", "暂无"} or not roles:
        raise ValueError("Source has no usable world view or role settings")
    story_outline["world_building"] = [world]
    story_outline["role_info"] = [roles]
    inputs = {
        "story_outline": story_outline,
        "episode_outline": episode_outline,
        "episode_outlines": source_data["episode_outlines.json"],
        "future_map": source_data["04_future_map/future_map.json"],
        "episode_contributions": source_data["04_future_map/episode_contributions.json"],
    }
    manifest = {
        "version": 2, "source": str(source), "episodes": EPISODES, "runs": 1,
        "arms": list(ARMS), "default_arm": "hybrid", "generator_model": "Qwen3.6-27B",
        "evaluator_model": "Qwen3.8-27B", "inputs_sha256": digest(inputs),
        "world_view_characters": len(world), "role_info_characters": len(story_outline["role_info"][0]),
        "initial_state": "Qwen3.6 extracts opening facts from actual generated assets, with exact source quotes; future outline events are not history",
        "asset_origin": "saved story_outline/script_proposal response from the source generation run",
        "intervention": "native memory + NMF lifecycle state with <=6000-character hybrid retrieval",
        "control": "native_context uses the same restored story bible but no added NMF state",
        "extraction": {"temperature": 0.0, "max_output_tokens": [4096, 8192, 8192], "max_attempts": 3},
        "generation": {"temperature": 1.0, "top_p": 1, "top_k": 50, "enable_thinking": False},
        "source_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in SOURCE_FILES},
        "implementation_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__),
                         REPO / "service/drama_by_creativity/generate_episode_script.py",
                         REPO / "service/drama_by_creativity/state_lifecycle_memory.py",
                         REPO / "service/drama_by_creativity/prompts/script.py",
                         REPO / "service/drama_by_creativity/model_configs.yaml",
                         REPO / "service/drama_by_creativity/generate_script_proposal.py",
                         REPO / "service/drama_by_creativity/generate_story_outline_and_role.py",
                         REPO / "service/drama_by_creativity/tests/run_test.py",
                         TENCENT_ROOT / "local_pipeline/gated_engine.py",
                         TENCENT_ROOT / "local_pipeline/state_retrieval.py")
        },
    }
    for arm in ARMS:
        directory = root / arm
        atomic_json(directory / "04_future_map/future_map.json", inputs["future_map"])
        atomic_json(directory / "04_future_map/episode_contributions.json", inputs["episode_contributions"])
        atomic_json(directory / "episode_outlines.json", inputs["episode_outlines"])
    atomic_json(root / "inputs.json", inputs)
    atomic_json(marker, manifest)


def load_inputs(root):
    manifest = read_json(root / "experiment.json")
    if manifest.get("version") != 2:
        raise ValueError("This experiment requires version 2 with state initialization from generated assets")
    inputs = read_json(root / "inputs.json")
    if manifest["inputs_sha256"] != digest(inputs):
        raise ValueError("Frozen experiment inputs changed")
    return inputs


def configure(directory, arm):
    os.environ.update({
        "DRAMA_OUTPUT_DIR": str(directory), "DRAMA_MEMORY_DIR": str(directory / "narrative_memory"),
        "DRAMA_LLM_MODEL": "Qwen3.6-27B", "DRAMA_LLM_THINKING": "0",
        "DRAMA_DISABLE_PLOT_RETRIEVAL": "1", "DRAMA_SUMMARY_MEMORY_BASELINE": "false",
        "DRAMA_ADJUST_WORD_COUNT": "false", "DRAMA_ADJUST_COHERENCE": "false",
        "DRAMA_STATE_LIFECYCLE_HYBRID": "1" if arm == "hybrid" else "0",
        "DRAMA_SUBTREE_MEMORY_LIMIT": "10", "DRAMA_LLM_MAX_TOKENS": "0",
    })
    os.environ.setdefault("DRAMA_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    os.environ.setdefault("DRAMA_LLM_TIMEOUT", "600")


def story_id(arm):
    return "state-hybrid-v2-" + arm


def scripts(directory):
    result = []
    for expected_id, path in enumerate(sorted((directory / "05_drama").glob("episode_*.json"))):
        item = read_json(path)
        if item.get("episode_id") != expected_id or not isinstance(item.get("content"), str) or not item["content"].strip():
            raise ValueError(f"Empty, duplicated, or nonconsecutive script: {path}")
        result.append(item["content"])
    if len(result) > EPISODES:
        raise ValueError("Too many generated episodes")
    return result


def progress(root, arm):
    return directory_progress(root / arm, arm, story_id(arm))


def directory_progress(directory, arm, identifier):
    texts = scripts(directory)
    state_path = directory / "narrative_memory" / identifier / "memory_state.json"
    native_count = int(read_json(state_path)["last_updated_episode"]) + 1 if state_path.exists() else 0
    if native_count > len(texts):
        raise ValueError("Native memory is ahead of saved scripts")
    lifecycle_count = None
    initialized = None
    if arm == "hybrid":
        memory = StateLifecycleMemory(identifier, directory)
        if len(memory.records) > len(texts):
            raise ValueError("Lifecycle memory is ahead of saved scripts")
        for index, record in enumerate(memory.records):
            if record["script_sha256"] != digest(texts[index]):
                raise ValueError(f"Episode {index + 1} changed after its state extraction")
        lifecycle_count = len(memory.records)
        initialized = memory.initialized
    completed = min(len(texts), native_count, lifecycle_count if lifecycle_count is not None else EPISODES)
    score_path = directory / "drama_evaluations_qwen38/full_60_episodes/scores.json"
    return {"arm": arm, "scripts": len(texts), "native_memory": native_count,
            "lifecycle_updates": lifecycle_count, "initialized": initialized, "complete": completed, "expected": EPISODES,
            "scored": score_path.exists()}


def verify_static_inputs(directory, inputs):
    for filename, key in (("future_map.json", "future_map"), ("episode_contributions.json", "episode_contributions")):
        if read_json(directory / "04_future_map" / filename) != inputs[key]:
            raise ValueError(f"Frozen planning artifact changed: {filename}")


def verify_server():
    with requests.Session() as session:
        session.trust_env = False
        headers = {"Authorization": "Bearer " + os.environ.get("DRAMA_LLM_API_KEY", "EMPTY")}
        response = session.get(os.environ["DRAMA_LLM_BASE_URL"].rstrip("/") + "/models", headers=headers, timeout=10)
        response.raise_for_status()
    models = {item["id"] for item in response.json().get("data", [])}
    if "Qwen3.6-27B" not in models:
        raise ValueError(f"Generation requires Qwen3.6-27B, available models: {sorted(models)}")


def make_request(inputs, arm, episode_id, identifier=None):
    story = pb.StoryOutline.from_dict(inputs["story_outline"])
    return pb.GenerateDramaByCreativityReq(
        story_info=pb.StoryInfo(story_id=identifier or story_id(arm), plot_type=pb.PlotType.SHORT_CARTOON),
        generate_input=pb.GenerateInputByCreativity(
            world_view=story.world_building[0], role_setting=story.role_info[0], topic="古代言情、轻喜",
            common=pb.GenerateInputCommon(season_nums=1, episode_nums=EPISODES,
                                         min_word_count_per_episode=1000, max_word_count_per_episode=1400)),
        generate_data=pb.GenerateDrama(
            story_outline=story, episode_outline=pb.EpisodeOutline.from_dict(inputs["episode_outline"]),
            select_range=[pb.GenerateDrama.SelectRange(season_id=0, episode_ids=[episode_id])]),
    )


async def generate(root, arm, limit):
    inputs = load_inputs(root)
    await generate_directory(inputs, root / arm, arm, story_id(arm), limit)


async def generate_directory(inputs, directory, arm, identifier, limit, *, scene_gate_factory=None):
    configure(directory, arm)
    verify_static_inputs(directory, inputs)
    verify_server()
    assets = assets_from_request(make_request(inputs, arm, 0, identifier))
    preserve_generation_assets(assets, directory)
    if arm == "hybrid":
        await StateLifecycleMemory(identifier).initialize(Context(), assets)
    episode_outlines = inputs["episode_outline"]["seasons"][0]["episodes"]
    for episode_id in range(limit):
        status = directory_progress(directory, arm, identifier)
        if episode_id < status["complete"]:
            continue
        if episode_id != status["complete"]:
            raise ValueError("Cannot skip incomplete episodes")
        ctx = Context(fields={"StoryID": identifier, "episode_number": episode_id + 1})
        path = directory / "05_drama" / f"episode_{episode_id + 1:02d}.json"
        outline = episode_outlines[episode_id]["content"]
        if path.exists():
            script = read_json(path)["content"]
            await ScriptContentFile(result=script).upload(ctx, LocalStore(), project_id=identifier, episode_id=episode_id)
            native = NarrativeMemory(identifier)
            if int(native.data["last_updated_episode"]) < episode_id:
                await native.update_from_episode(ctx, episode_id, outline, script)
            if arm == "hybrid":
                await StateLifecycleMemory(identifier).update_from_episode(ctx, episode_id, outline, script)
            print(f"recovered_existing_script=E{episode_id + 1:02d}", flush=True)
        else:
            if scene_gate_factory is None:
                await generate_episode_script(ctx, make_request(inputs, arm, episode_id, identifier), LocalStore())
            else:
                await generate_episode_script(ctx, make_request(inputs, arm, episode_id, identifier), LocalStore(),
                                              scene_gate_factory=scene_gate_factory)
        if directory_progress(directory, arm, identifier)["complete"] != episode_id + 1:
            raise RuntimeError(f"Episode {episode_id + 1} did not finish script and memory updates; rerun to recover")
        print(json.dumps(directory_progress(directory, arm, identifier), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "status", "generate", "evaluate"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--arm", choices=ARMS, default="hybrid")
    parser.add_argument("--limit", type=int, default=EPISODES, help="Generate through this episode, default 60")
    parser.add_argument("--execute-api", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not 1 <= args.limit <= EPISODES:
        parser.error("--limit must be between 1 and 60")
    if args.action == "prepare":
        prepare(args.source_dir.resolve(), root)
        print(f"Prepared {root}; source story bible restored; no API calls")
        return
    load_inputs(root)
    if args.action == "status":
        for arm in ARMS:
            print(json.dumps(progress(root, arm), ensure_ascii=False))
        return
    if args.action == "generate":
        if not args.execute_api:
            print(json.dumps(progress(root, args.arm), ensure_ascii=False))
            print("Dry-run; add --execute-api to generate")
            return
        with (root / args.arm / ".generation.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            asyncio.run(generate(root, args.arm, args.limit))
        return
    if progress(root, args.arm)["complete"] != EPISODES:
        raise ValueError("Finish all 60 scripts and memory updates before evaluation")
    command = [sys.executable, str(REPO / "tools/evaluate_with_drama_evaluator.py"), str(root / args.arm)]
    if args.execute_api:
        command.append("--execute-api")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
