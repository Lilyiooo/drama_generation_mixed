"""
从润色后的故事大纲继续生成后续流程（集大纲 + 剧本）。

这是一个「额外入口」：不修改原有剧本生成链路任何代码，只是从一份润色过的
故事大纲（outline_final_*.json）出发，接着生成 04 集大纲 和 05 剧本。

本目录（drama_operator_new）已改造：集大纲生成不再依赖卡点规划（plot_point），
只依赖故事大纲（story_outline）。因此本入口也不再需要 script_proposal。

原理：
- 通过设置独立的 DRAMA_OUTPUT_DIR 环境变量，把结果落到新目录，不覆盖原输出。
- 前3集剧本从本地存储 按 story_id 读取（默认 APID-test-001）。

用法：
    python run_from_refined_outline.py \
        --refined-outline /path/to/outline_final_0812_211602.json \
        --output-dir /path/to/new_output_dir \
        [--story-id APID-test-001] [--episode-nums 60] [--plot-type SHORT_CARTOON]
"""

import os
import sys
import json
import argparse
import asyncio
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from drama_local import runtime as context
from drama_local import models as pb
from drama_local.runtime import LocalStore

from service.drama_by_creativity import DramaByCreativity
from service.drama_by_creativity.generate_story_outline_and_role import transfer_outline_to_str

# 复用 run_test 里的测试常量与工具函数（不复制密钥，直接 import）
from run_test import (
    TEST_CORE_STORY,
    TEST_TOPIC,
    TEST_ROLE_SETTING,
    TEST_REFERENCE,
    DEFAULT_SEASON_NUMS,
    DEFAULT_MIN_WORD_COUNT,
    DEFAULT_MAX_WORD_COUNT,
    create_creativity,
    get_plot_type,
)

DEFAULT_STORY_ID = "APID-test-001"


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_story_outline(refined_outline_path: str) -> pb.StoryOutline:
    """读取润色后的 outline，转成 pb.StoryOutline。"""
    refined = load_json(refined_outline_path)
    outline_str = transfer_outline_to_str(refined)
    # 集大纲生成只消费 story_outline 字符串，role_info 可留空
    return pb.StoryOutline(story_outline=[outline_str], role_info=[], world_building=[])


def build_generate_input(args) -> pb.GenerateInputByCreativity:
    return pb.GenerateInputByCreativity(
        core_story=TEST_CORE_STORY,
        topic=TEST_TOPIC,
        world_view="",
        role_setting=TEST_ROLE_SETTING,
        reference=TEST_REFERENCE,
        common=pb.GenerateInputCommon(
            season_nums=DEFAULT_SEASON_NUMS,
            episode_nums=args.episode_nums,
            min_word_count_per_episode=DEFAULT_MIN_WORD_COUNT,
            max_word_count_per_episode=DEFAULT_MAX_WORD_COUNT,
        ),
    )


async def step_episode_outline(ctx, creativity, args, story_outline_pb):
    """第4步：集大纲生成（消费润色后的 story_outline，不依赖卡点规划）。"""
    request = pb.GenerateEpisodeOutlineByCreativityReq(
        story_info=pb.StoryInfo(story_id=args.story_id, plot_type=args.plot_type),
        generate_input=build_generate_input(args),
        generated_data=pb.GeneratedData(
            story_outline=story_outline_pb,
        ),
    )
    print("\n" + "=" * 70)
    print("  第4步：集大纲生成（基于润色后的故事大纲）")
    print("=" * 70)
    start = time.time()
    ret = await creativity.GenerateEpisodeOutline(ctx, request)
    print(f"[耗时] 集大纲生成: {time.time() - start:.2f}s")
    return ret


async def step_drama(ctx, creativity, args, story_outline_pb, episode_outline_rsp):
    """第5步：剧本撰写。"""
    all_episodes = episode_outline_rsp.episode_outline.seasons[0].episodes
    episode_ids = list(range(len(all_episodes)))

    request = pb.GenerateDramaByCreativityReq(
        story_info=pb.StoryInfo(story_id=args.story_id, plot_type=args.plot_type),
        generate_input=build_generate_input(args),
        generate_data=pb.GenerateDrama(
            story_outline=story_outline_pb,
            episode_outline=episode_outline_rsp.episode_outline,
            prev_episode_content="",
            suggestion="",
            select_range=[pb.GenerateDrama.SelectRange(season_id=0, episode_ids=episode_ids)],
        ),
    )
    print("\n" + "=" * 70)
    print(f"  第5步：剧本撰写（{len(episode_ids)}集）")
    print("=" * 70)
    start = time.time()
    ret = await creativity.GenerateDrama(ctx, request)
    print(f"[耗时] 剧本撰写: {time.time() - start:.2f}s")
    return ret


def parse_args():
    parser = argparse.ArgumentParser(description="从润色后的故事大纲继续生成集大纲和剧本（不依赖卡点规划）")
    parser.add_argument("--refined-outline", required=True, help="润色后的 outline JSON 路径")
    parser.add_argument("--output-dir", required=True, help="输出目录（不覆盖原输出）")
    parser.add_argument("--story-id", default=DEFAULT_STORY_ID, help="项目ID（用于从本地存储读取前3集剧本）")
    parser.add_argument("--episode-nums", type=int, default=60, help="总集数")
    parser.add_argument("--plot-type", type=str, default="SHORT_CARTOON",
                        choices=["SHORT", "SHORT_CARTOON", "CARTOON"])
    parser.add_argument("--only-episode-outline", action="store_true",
                        help="只生成集大纲（第4步），不生成剧本（第5步）")
    return parser.parse_args()


async def main():
    args = parse_args()
    args.plot_type = get_plot_type(args.plot_type)

    # 关键：设置独立输出目录，避免覆盖原输出
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.environ["DRAMA_OUTPUT_DIR"] = output_dir

    print(f"\n[配置] story_id={args.story_id}, episode_nums={args.episode_nums}, plot_type={args.plot_type}")
    print(f"[输出] DRAMA_OUTPUT_DIR={output_dir}")
    print(f"[润色大纲] {args.refined_outline}")

    # 1. 用润色 outline 构造 story_outline（不再重建 script_proposal）
    story_outline_pb = build_story_outline(args.refined_outline)
    print(f"✅ 已加载润色大纲（story_outline 字符串长度 {len(story_outline_pb.story_outline[0])}）")

    # 2. 初始化服务
    ctx = context.Context()
    creativity = create_creativity()

    # 3. 集大纲
    episode_outline_rsp = await step_episode_outline(ctx, creativity, args, story_outline_pb)

    if args.only_episode_outline:
        print("\n✅ 集大纲生成完成（已跳过剧本撰写）。")
        return

    # 4. 剧本
    await step_drama(ctx, creativity, args, story_outline_pb, episode_outline_rsp)

    print(f"\n✅ 全部完成！输出目录: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
