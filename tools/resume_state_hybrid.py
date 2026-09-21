"""Resume only script writing in a saved 60-episode state-hybrid full-pipeline run."""

import argparse
import asyncio
import fcntl
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import state_hybrid_study as study


def saved_inputs(directory, identifier):
    assets = study.read_json(directory / "generation_assets.json")
    state_assets = directory / "state_lifecycle" / identifier / "story_assets.json"
    if not state_assets.exists() or study.read_json(state_assets) != assets:
        raise ValueError("Missing or mismatched original state assets; check --story-id and --run-dir")
    episodes = assets["episode_outlines"]
    if len(episodes) != study.EPISODES or any(
        item.get("episode_id") != index or item.get("season_id") != 0 or not item.get("content", "").strip()
        for index, item in enumerate(episodes)
    ):
        raise ValueError("Resume requires this run's 60 ordered, nonempty, single-season outlines")
    inputs = {
        "story_outline": {"world_building": [assets["world_view"]], "role_info": [assets["role_info"]],
                          "story_outline": [assets["story_outline"]]},
        "episode_outline": {"seasons": [{"season_id": 0, "episodes": episodes}]},
        "future_map": study.read_json(directory / "04_future_map/future_map.json"),
        "episode_contributions": study.read_json(directory / "04_future_map/episode_contributions.json"),
    }
    reconstructed = study.assets_from_request(study.make_request(inputs, "hybrid", 0, identifier))
    if reconstructed != assets:
        raise ValueError("Resumed request does not preserve the exact generated assets")
    return inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=REPO / "output/full_pipeline_state_hybrid_v2")
    parser.add_argument("--story-id", default="APID-test-001")
    parser.add_argument("--limit", type=int, default=study.EPISODES)
    parser.add_argument("--execute-api", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit <= study.EPISODES:
        parser.error("--limit must be between 1 and 60")
    if not args.story_id or Path(args.story_id).name != args.story_id or args.story_id in {".", ".."}:
        parser.error("--story-id must be a single directory name")
    directory = args.run_dir.resolve()
    inputs = saved_inputs(directory, args.story_id)
    print(json.dumps(study.directory_progress(directory, "hybrid", args.story_id), ensure_ascii=False), flush=True)
    if not args.execute_api:
        print("Read-only; saved world, roles and all 60 outlines verified; add --execute-api to resume script writing")
        return
    with (directory / ".generation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(study.generate_directory(inputs, directory, "hybrid", args.story_id, args.limit))


if __name__ == "__main__":
    main()
