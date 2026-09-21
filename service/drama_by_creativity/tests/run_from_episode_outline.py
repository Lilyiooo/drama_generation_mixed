"""从已有 04_episode_outline 直接续跑第 5 步剧本生成和叙事记忆。"""
import argparse
import asyncio
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from drama_local import runtime as context
from drama_local import models as pb

from service.drama_by_creativity.generate_episode_outline import transfer_outline_to_str
from service.drama_by_creativity.future_map import generate_future_map_artifacts
from service.drama_by_creativity.summary_memory import summary_baseline_enabled
from service.drama_by_creativity.regenerate_story_outline import transfer_outline_to_str as story_outline_to_str
from run_test import (
    DEFAULT_MAX_WORD_COUNT,
    DEFAULT_MIN_WORD_COUNT,
    DEFAULT_SEASON_NUMS,
    TEST_CORE_STORY,
    TEST_REFERENCE,
    TEST_ROLE_SETTING,
    TEST_TOPIC,
    create_creativity,
    get_plot_type,
)

DEFAULT_SOURCE_DIR = os.path.abspath(
    os.path.join(project_root, "output", "60ep_0810_1946")
)
REQUIRED_EPISODE_FIELDS = {
    "episode_id",
    "title",
    "core_plot",
    "roles",
    "highlights",
    "character_growth",
    "relationship_changes",
    "main_storyline_progression",
}


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_episode_outlines(source_dir: str) -> List[Dict[str, Any]]:
    outline_dir = os.path.join(source_dir, "04_episode_outline")
    paths = sorted(glob.glob(os.path.join(outline_dir, "*.json")))
    if not paths:
        raise FileNotFoundError(f"没有找到集大纲 JSON：{outline_dir}")

    episodes: List[Dict[str, Any]] = []
    for path in paths:
        payload = load_json(path)
        if not isinstance(payload, list):
            raise ValueError(f"集大纲文件顶层必须是数组：{path}")
        episodes.extend(payload)

    if not episodes:
        raise ValueError("集大纲为空")
    if any(not isinstance(episode, dict) for episode in episodes):
        raise ValueError("集大纲数组中存在非对象元素")

    episodes.sort(key=lambda episode: int(episode.get("episode_id", -1)))
    expected_ids = list(range(1, len(episodes) + 1))
    actual_ids = [int(episode.get("episode_id", -1)) for episode in episodes]
    if actual_ids != expected_ids:
        raise ValueError(f"集大纲 episode_id 必须从 1 连续递增，实际为：{actual_ids}")

    for episode in episodes:
        missing = REQUIRED_EPISODE_FIELDS - set(episode)
        if missing:
            raise ValueError(f"第 {episode['episode_id']} 集缺少字段：{sorted(missing)}")
        if not str(episode["title"]).strip() or not str(episode["core_plot"]).strip():
            raise ValueError(f"第 {episode['episode_id']} 集标题或核心情节为空")
        if not isinstance(episode["roles"], list):
            raise ValueError(f"第 {episode['episode_id']} 集 roles 必须是数组")
    return episodes


def build_episode_outline(episodes: List[Dict[str, Any]]) -> pb.EpisodeOutline:
    _, outline_strings = transfer_outline_to_str(episodes)
    pb_episodes = [
        pb.EpisodeOutline.Episode(
            season_id=0,
            episode_id=index,
            chapter_from=0,
            chapter_to=0,
            content=content,
        )
        for index, content in enumerate(outline_strings)
    ]
    return pb.EpisodeOutline(
        seasons=[pb.EpisodeOutline.Season(season_id=0, episodes=pb_episodes)]
    )


def build_story_outline(source_dir: str) -> pb.StoryOutline:
    story_path = os.path.join(source_dir, "03_story_outline", "outline.json")
    world_path = os.path.join(source_dir, "01_script_proposal", "01_world_view.json")
    roles_path = os.path.join(source_dir, "01_script_proposal", "02_role_setting.json")

    story_json = load_json(story_path)
    background_path = os.path.join(source_dir, "generation_background.json")
    if os.path.isfile(background_path):
        background = load_json(background_path)
        world_json = {"world_view": background.get("WorldView", "")}
        role_info = background.get("RoleInfo", "")
        roles_json = role_info
    else:
        world_json = load_json(world_path)
        roles_json = load_json(roles_path)
    if not isinstance(story_json, dict):
        raise ValueError(f"故事大纲必须是 JSON 对象：{story_path}")
    if not isinstance(world_json, dict) or not str(world_json.get("world_view", "")).strip():
        raise ValueError(f"世界观文件格式错误：{world_path}")
    if not isinstance(roles_json, (str, list)) or not roles_json or (isinstance(roles_json, str) and not roles_json.strip()):
        raise ValueError(f"角色设定必须是非空文字或 JSON 数组：{roles_path}")

    return pb.StoryOutline(
        story_outline=[story_outline_to_str(story_json)],
        role_info=[roles_json if isinstance(roles_json, str) else json.dumps(roles_json, ensure_ascii=False, indent=2)],
        world_building=[str(world_json["world_view"])],
    )


def resolve_episode_ids(args, episode_count: int) -> List[int]:
    memory_path = os.path.join(
        args.output_dir,
        "summary_memory" if summary_baseline_enabled() else "narrative_memory",
        args.story_id,
        "memory_state.json",
    )
    start_episode = args.start_episode
    if args.resume:
        if os.path.exists(memory_path):
            memory = load_json(memory_path)
            start_episode = int(memory.get("last_updated_episode", -1)) + 2
        else:
            start_episode = 1
    end_episode = args.end_episode or episode_count
    if not 1 <= start_episode <= episode_count:
        raise ValueError(f"start_episode 必须在 1～{episode_count}，实际为 {start_episode}")
    if not start_episode <= end_episode <= episode_count:
        raise ValueError(f"end_episode 必须在 {start_episode}～{episode_count}，实际为 {end_episode}")
    if start_episode > 1 and not os.path.exists(memory_path):
        raise ValueError(
            "不能从中间集启动一个空记忆：请从第1集开始，或使用已有相同 output-dir/story-id 的记忆并加 --resume。"
        )
    return list(range(start_episode - 1, end_episode))


def build_generate_input(episode_count: int, source_dir=None) -> pb.GenerateInputByCreativity:
    background_path = os.path.join(source_dir, "generation_background.json") if source_dir else ""
    background = load_json(background_path) if os.path.isfile(background_path) else {}
    return pb.GenerateInputByCreativity(
        core_story=background.get("StoryOutline", TEST_CORE_STORY),
        topic=background.get("Topic", TEST_TOPIC),
        world_view=background.get("WorldView", ""),
        role_setting=background.get("RoleInfo", TEST_ROLE_SETTING),
        reference=background.get("Reference", TEST_REFERENCE),
        common=pb.GenerateInputCommon(
            season_nums=DEFAULT_SEASON_NUMS,
            episode_nums=episode_count,
            min_word_count_per_episode=DEFAULT_MIN_WORD_COUNT,
            max_word_count_per_episode=DEFAULT_MAX_WORD_COUNT,
        ),
    )


def save_run_manifest(args, episode_ids: List[int], episode_count: int) -> None:
    from service.drama_by_creativity.script_postprocessing import word_count_adjustment_enabled, coherence_adjustment_enabled
    manifest = {
        "adjust_word_count": word_count_adjustment_enabled(),
        "adjust_coherence": coherence_adjustment_enabled(),
        "summary_memory_baseline": summary_baseline_enabled(),
        "source_dir": args.source_dir,
        "episode_outline_dir": os.path.join(args.source_dir, "04_episode_outline"),
        "output_dir": args.output_dir,
        "story_id": args.story_id,
        "plot_type": args.plot_type_name,
        "episode_count": episode_count,
        "selected_episode_numbers": [episode_id + 1 for episode_id in episode_ids],
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "resume_from_episode_outline.json"), "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


async def main() -> None:
    args = parse_args()
    if args.summary_memory_baseline:
        os.environ["DRAMA_SUMMARY_MEMORY_BASELINE"] = "1"
    args.source_dir = os.path.abspath(args.source_dir)
    args.output_dir = os.path.abspath(
        args.output_dir or f"{args.source_dir.rstrip(os.sep)}_memrun"
    )
    if args.output_dir == args.source_dir:
        raise ValueError("output-dir 不能与 source-dir 相同，避免覆盖原来的 05_drama。")

    args.plot_type_name = args.plot_type
    args.plot_type = get_plot_type(args.plot_type)
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["DRAMA_OUTPUT_DIR"] = args.output_dir

    episodes = load_episode_outlines(args.source_dir)
    story_outline = build_story_outline(args.source_dir)
    episode_outline = build_episode_outline(episodes)
    episode_ids = resolve_episode_ids(args, len(episodes))
    save_run_manifest(args, episode_ids, len(episodes))

    memory_dir = "summary_memory" if summary_baseline_enabled() else "narrative_memory"
    print(f"[输入] 已加载 {len(episodes)} 集大纲：{args.source_dir}/04_episode_outline")
    print(f"[范围] 第 {episode_ids[0] + 1}～{episode_ids[-1] + 1} 集")
    print(f"[输出] {args.output_dir}")
    print(f"[记忆] {args.output_dir}/{memory_dir}/{args.story_id}")
    print(f"[COS story_id] {args.story_id}")

    request = pb.GenerateDramaByCreativityReq(
        story_info=pb.StoryInfo(story_id=args.story_id, plot_type=args.plot_type),
        generate_input=build_generate_input(len(episodes), args.source_dir),
        generate_data=pb.GenerateDrama(
            story_outline=story_outline,
            episode_outline=episode_outline,
            prev_episode_content="",
            suggestion="",
            select_range=[
                pb.GenerateDrama.SelectRange(season_id=0, episode_ids=episode_ids)
            ],
        ),
    )

    ctx = context.Context()
    if not summary_baseline_enabled(request.generate_input):
        print("[Future Map] 开始根据完整集大纲生成倒推需求树与分集贡献映射")
        await generate_future_map_artifacts(ctx, episodes)

    creativity = create_creativity()
    start = time.time()
    response = await creativity.GenerateDrama(ctx, request)
    generated_count = 0
    if response and response.result and response.result.seasons:
        generated_count = len(response.result.seasons[0].episodes)
    print(f"[完成] 本次生成 {generated_count} 集，耗时 {time.time() - start:.2f}s")
    print(f"[剧本目录] {args.output_dir}/05_drama")
    print(f"[记忆目录] {args.output_dir}/{memory_dir}/{args.story_id}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="复用已有 04_episode_outline，直接生成最终剧本和 Event Graph/World State"
    )
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR, help="包含 01～04 阶段结果的源目录")
    parser.add_argument("--output-dir", help="新剧本和记忆输出目录；默认在源目录名后追加 _memrun")
    parser.add_argument("--story-id", default="APID-test-001-mem-v1", help="独立 COS 与记忆项目 ID")
    parser.add_argument(
        "--plot-type",
        default="SHORT_CARTOON",
        choices=["SHORT", "SHORT_CARTOON", "CARTOON"],
    )
    parser.add_argument("--start-episode", type=int, default=1, help="起始集，1-based")
    parser.add_argument("--end-episode", type=int, help="结束集，1-based；默认最后一集")
    parser.add_argument("--resume", action="store_true", help="从同一输出目录记忆中的下一集继续")
    parser.add_argument("--summary-memory-baseline", action="store_true", help="只使用此前各集梗概作为情节记忆，默认关闭")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main())
