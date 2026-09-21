from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import Scene

_SCENE_HEADING = re.compile(
    r"(?m)^\s*(?P<episode>\d+)\s*[-—]\s*(?P<scene>\d+)\s*[.、]\s*(?P<heading>[^\n]+)\s*$"
)
_EPISODE_NUMBER = re.compile(r"episode_(\d+)\.json$")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def split_script_content(content: str, source_file: str) -> list[Scene]:
    matches = list(_SCENE_HEADING.finditer(content))
    scenes: list[Scene] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        episode_id = int(match.group("episode"))
        scene_number = int(match.group("scene"))
        body = content[start:end].strip().strip("-").strip()
        scenes.append(
            Scene(
                scene_id=f"E{episode_id:03d}S{scene_number:03d}",
                episode_id=episode_id,
                scene_number=scene_number,
                heading=match.group("heading").strip(),
                text=body,
                source_file=source_file,
            )
        )
    return scenes


def load_final_scenes(run_dir: Path) -> list[Scene]:
    drama_dir = run_dir / "05_drama"
    scenes: list[Scene] = []
    for path in sorted(drama_dir.glob("episode_*.json")):
        payload = _read_json(path)
        content = payload.get("content", "") if isinstance(payload, dict) else ""
        parsed = split_script_content(content, str(path))
        if parsed:
            scenes.extend(parsed)
            continue
        match = _EPISODE_NUMBER.search(path.name)
        episode_id = int(match.group(1)) if match else int(payload.get("episode_id", 0)) + 1
        if content.strip():
            scenes.append(
                Scene(
                    scene_id=f"E{episode_id:03d}S001",
                    episode_id=episode_id,
                    scene_number=1,
                    heading="未识别场景头",
                    text=content.strip(),
                    source_file=str(path),
                )
            )
    return scenes


def load_outline_scenes(run_dir: Path) -> list[Scene]:
    outline_dir = run_dir / "04_episode_outline"
    scenes: list[Scene] = []
    for path in sorted(outline_dir.glob("*.json")):
        payload = _read_json(path)
        episodes = payload if isinstance(payload, list) else payload.get("episodes", [])
        for item in episodes:
            episode_id = int(item["episode_id"])
            fields = [
                item.get("core_plot", ""),
                item.get("highlights", ""),
                item.get("character_growth", ""),
                item.get("relationship_changes", ""),
                item.get("main_storyline_progression", ""),
                item.get("ending_hook", ""),
            ]
            scenes.append(
                Scene(
                    scene_id=f"E{episode_id:03d}S001",
                    episode_id=episode_id,
                    scene_number=1,
                    heading=item.get("title", f"第{episode_id}集"),
                    text="\n".join(value for value in fields if value),
                    source_file=str(path),
                    metadata=item,
                )
            )
    return scenes


def load_scenes(run_dir: str | Path, source: str = "auto") -> tuple[list[Scene], str]:
    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"评测目录不存在：{root}")
    if source not in {"auto", "drama", "outline"}:
        raise ValueError("source 必须是 auto、drama 或 outline")
    if source in {"auto", "drama"}:
        scenes = load_final_scenes(root)
        if scenes or source == "drama":
            return scenes, "drama"
    return load_outline_scenes(root), "outline"


def save_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
