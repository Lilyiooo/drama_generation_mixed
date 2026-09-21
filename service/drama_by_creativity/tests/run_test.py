"""
创作链路测试脚本

创作链路说明：
1. 剧本规划（人设、世界观、卡点规划）  -> generate_script_proposal
2. 故事大纲生成                        -> GenerateStoryOutline
3. 前十集/全部集大纲生成               -> GenerateEpisodeOutline
4. 前十集/全部剧本撰写                 -> GenerateDrama
5. 重新生成单集大纲                    -> RegenerateEpisodeOutlineSingle

使用方式：
    # 从头跑到尾（全流程）
    python test_function.py --all

    # 单独测试某个环节
    python test_function.py --step script_proposal       # 1. 剧本规划
    python test_function.py --step story_outline         # 2. 故事大纲
    python test_function.py --step episode_outline       # 3. 集大纲生成
    python test_function.py --step drama                 # 4. 剧本撰写
    python test_function.py --step regenerate_single     # 5. 重新生成单集大纲

    # 从某个环节开始跑到尾

    # 指定项目ID（用于区分本地故事数据）
    python test_function.py --all --story_id APID-xxxx
"""

import sys
import os
import argparse
import asyncio
import json
import time
import logging
from pathlib import Path

# 设置项目根目录到Python路径（test文件夹在drama_by_creativity下，所以需要往上3级到drama_operator目录）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from drama_local import runtime as context
from drama_local import models as pb
from drama_local.runtime import LocalStore

from service.drama_by_creativity import DramaByCreativity


# ========== 本地存储 ==========


# ========== 测试数据 ==========
# 你可以根据需要修改以下测试数据

TEST_CORE_STORY = """男主是一个从战场上退役下来的将军，十六岁时英勇的将军父亲战死，首富的母亲病死，他被逼上战场，打了十年的仗，身无分文欠了一屁股债，还身中奇毒（因毒性太强，成了以毒攻毒的人形解药）。眼看无仗可打，要回到京城面对"装贤惠"的继母算计自己的婚事，伪善的亲戚朋友拿孝道压人，以及当年欺辱暗害趁火打劫的一帮死对头。男主深知自己不擅宅斗，手下都是一批直来直去的大老粗，发愁。

路过一个重镇，看到女主撒泼，追着自己的恶邻和亲戚输出，两波人顷刻间家破人亡的程度。男主大喜，算计女主，跟女主签订了雇佣关系，让女主嫁给自己做将军夫人，帮他对付自己的家人，以及欠他债的人他都要一一讨债。

女主是从战场死人堆里爬出来的，并且养了三个人，这三个人集齐了老弱病残幼，但其实各个是身怀绝技的扫地僧，有擅长下药+治病的，有擅长搜集情报的+坑蒙拐骗的，有能打架从未输过的，甚至几人的真实身份也都是龙王级别。四个人跟男主回了京城，本以为发财了，结果发现将军府破败，宅子都被人给占了，穷的只剩一堆欠条。"""

TEST_TOPIC = "古代言情、轻喜"

TEST_ROLE_SETTING = "男主是一个从战场上退役下来的将军，十六岁时英勇的将军父亲战死，首富的母亲病死，他被逼上战场，打了十年的仗，身无分文欠了一屁股债，还身中奇毒。女主是从战场死人堆里爬出来的，并且养了三个人，这三个人集齐了老弱病残幼，但其实各个是身怀绝技的扫地僧。"

TEST_REFERENCE = ""

# 默认配置
DEFAULT_EPISODE_NUMS = 60
DEFAULT_SEASON_NUMS = 1
DEFAULT_MIN_WORD_COUNT = 1000  # 每集字数下限
DEFAULT_MAX_WORD_COUNT = 1400  # 每集字数上限
DEFAULT_PLOT_TYPE = pb.PlotType.SHORT_CARTOON
DEFAULT_STORY_ID = "APID-test-001"


# ========== 工具函数 ==========

def create_cos():
    """创建当前输出目录的本地存储。"""
    return LocalStore()


def create_creativity():
    """创建DramaByCreativity实例"""
    cos = create_cos()
    return DramaByCreativity(cos)


def print_separator(title: str):
    """打印分隔线"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_result(result, label: str = "结果"):
    """打印结果"""
    print(f"\n--- {label} ---")
    print(str(result)[:3000])  # 限制输出长度，避免刷屏
    if len(str(result)) > 3000:
        print(f"\n... (结果过长，已截断，总长度: {len(str(result))} 字符)")
    print(f"--- {label} END ---\n")


# ========== 各环节测试函数 ==========

async def step_1_script_proposal(ctx, creativity, args) -> pb.GenerateScriptProposalRsp:
    """
    第1步：剧本规划（生成人设、世界观、卡点规划）
    
    输入：核心故事、题材、角色设定、参考
    输出：ScriptProposal（包含世界观、人设、卡点规划）
    """
    print_separator("第1步：剧本规划（人设、世界观、卡点规划）")
    
    request = pb.GenerateScriptProposalByCreativityReq(
        story_info=pb.StoryInfo(
            plot_type=args.plot_type
        ),
        generate_input=pb.GenerateInputByCreativity(
            core_story=TEST_CORE_STORY,
            topic=TEST_TOPIC,
            world_view="",
            role_setting=TEST_ROLE_SETTING,
            reference=TEST_REFERENCE,
            common=pb.GenerateInputCommon(
                season_nums=DEFAULT_SEASON_NUMS,
                episode_nums=args.episode_nums,
                min_word_count_per_episode=DEFAULT_MIN_WORD_COUNT,
                max_word_count_per_episode=DEFAULT_MAX_WORD_COUNT
            )
        )
    )
    
    start_time = time.time()
    ret = await creativity.GenerateScriptProposal(ctx, request)
    print(f"[耗时] 剧本规划: {time.time() - start_time:.2f}s")
    print_result(ret, "剧本规划结果")
    return ret


async def step_3_story_outline(ctx, creativity, args, script_proposal_rsp: pb.GenerateScriptProposalRsp = None) -> pb.GenerateStoryOutlineRsp:
    """
    第2步：故事大纲生成
    
    对于SHORT/SHORT_CARTOON类型：基于故事背景和整体剧本策划生成故事大纲（不生成人设）
    对于其他类型：生成故事大纲和人设
    
    输入：核心故事、题材、剧本规划
    输出：故事大纲 + 人设
    """
    print_separator("第2步：故事大纲生成")
    
    # 构造 generate_data（SHORT类型需要传入script_proposal）
    generate_data = None
    generated_world = ""
    generated_roles = TEST_ROLE_SETTING
    if script_proposal_rsp is not None:
        generate_data = pb.GenerateStoryOutline(
            script_proposal=script_proposal_rsp.script_proposal
        )
        generated_world = "\n\n".join(script_proposal_rsp.story_outline.world_building)
        generated_roles = "\n\n".join(script_proposal_rsp.story_outline.role_info) or TEST_ROLE_SETTING
    
    request = pb.GenerateStoryOutlineByCreativityReq(
        story_info=pb.StoryInfo(
            story_id=args.story_id,
            plot_type=args.plot_type
        ),
        generate_input=pb.GenerateInputByCreativity(
            core_story=TEST_CORE_STORY,
            topic=TEST_TOPIC,
            world_view=generated_world,
            role_setting=generated_roles,
            reference=TEST_REFERENCE,
            common=pb.GenerateInputCommon(
                season_nums=DEFAULT_SEASON_NUMS,
                episode_nums=args.episode_nums,
                min_word_count_per_episode=DEFAULT_MIN_WORD_COUNT,
                max_word_count_per_episode=DEFAULT_MAX_WORD_COUNT
            )
        ),
        generate_data=generate_data
    )
    
    start_time = time.time()
    ret = await creativity.GenerateStoryOutline(ctx, request)
    print(f"[耗时] 故事大纲生成: {time.time() - start_time:.2f}s")
    print_result(ret, "故事大纲结果")
    return ret


async def step_4_episode_outline(ctx, creativity, args,
                                  script_proposal_rsp: pb.GenerateScriptProposalRsp = None,
                                  story_outline_rsp: pb.GenerateStoryOutlineRsp = None) -> pb.GenerateEpisodeOutlineRsp:
    """
    第3步：集大纲生成（前十集/全部）
    
    输入：核心故事、题材、故事大纲、人设、剧本规划
    输出：所有集的大纲
    """
    print_separator("第3步：集大纲生成（前十集/全部）")
    
    # 从前面的结果中提取数据
    story_outline = pb.StoryOutline(story_outline=[""], role_info=[""], world_building=[""])
    script_proposal = pb.ScriptProposal()
    
    if story_outline_rsp is not None:
        story_outline = story_outline_rsp.story_outline
    elif script_proposal_rsp is not None:
        story_outline = script_proposal_rsp.story_outline
    
    if script_proposal_rsp is not None:
        script_proposal = script_proposal_rsp.script_proposal
    
    request = pb.GenerateEpisodeOutlineByCreativityReq(
        story_info=pb.StoryInfo(
            story_id=args.story_id,
            plot_type=args.plot_type
        ),
        generate_input=pb.GenerateInputByCreativity(
            core_story=TEST_CORE_STORY,
            topic=TEST_TOPIC,
            world_view="",
            role_setting=TEST_ROLE_SETTING,
            reference=TEST_REFERENCE,
            common=pb.GenerateInputCommon(
                season_nums=DEFAULT_SEASON_NUMS,
                episode_nums=args.episode_nums,
                min_word_count_per_episode=DEFAULT_MIN_WORD_COUNT,
                max_word_count_per_episode=DEFAULT_MAX_WORD_COUNT
            )
        ),
        generated_data=pb.GeneratedData(
            script_proposal=script_proposal,
            story_outline=story_outline
        )
    )
    
    start_time = time.time()
    ret = await creativity.GenerateEpisodeOutline(ctx, request)
    print(f"[耗时] 集大纲生成: {time.time() - start_time:.2f}s")
    print_result(ret, "集大纲结果")
    return ret


async def step_5_drama(ctx, creativity, args,
                       story_outline: pb.StoryOutline = None,
                       episode_outline_rsp: pb.GenerateEpisodeOutlineRsp = None,
                       episode_ids: list = None) -> pb.GenerateDramaRsp:
    """
    第4步：剧本撰写（前十集/全部）
    
    输入：故事大纲、人设、集大纲、要生成的集数范围
    输出：指定集数的剧本
    """
    print_separator("第4步：剧本撰写")
    
    if story_outline is None:
        story_outline = pb.StoryOutline(story_outline=[""], role_info=[""], world_building=[""])
    
    if episode_outline_rsp is None:
        print("[警告] 没有集大纲数据，无法生成剧本。请先运行第3步。")
        return None
    
    # 可通过 episode_ids 参数指定生成范围
    if episode_ids is None:
        # 从集大纲中获取所有集的ID
        all_episodes = episode_outline_rsp.episode_outline.seasons[0].episodes
        max_episodes = min(len(all_episodes), 10)  # 默认最多生成前10集
        episode_ids = list(range(max_episodes))
    
    request = pb.GenerateDramaByCreativityReq(
        story_info=pb.StoryInfo(
            story_id=args.story_id,
            plot_type=args.plot_type
        ),
        generate_input=pb.GenerateInputByCreativity(
            core_story=TEST_CORE_STORY,
            topic=TEST_TOPIC,
            world_view="",
            role_setting=TEST_ROLE_SETTING,
            reference=TEST_REFERENCE,
            common=pb.GenerateInputCommon(
                season_nums=DEFAULT_SEASON_NUMS,
                episode_nums=args.episode_nums,
                min_word_count_per_episode=DEFAULT_MIN_WORD_COUNT,
                max_word_count_per_episode=DEFAULT_MAX_WORD_COUNT
            )
        ),
        generate_data=pb.GenerateDrama(
            story_outline=story_outline,
            episode_outline=episode_outline_rsp.episode_outline,
            prev_episode_content="",
            suggestion="",
            select_range=[
                pb.GenerateDrama.SelectRange(
                    season_id=0,
                    episode_ids=episode_ids
                )
            ]
        )
    )
    
    start_time = time.time()
    ret = await creativity.GenerateDrama(ctx, request)
    print(f"[耗时] 剧本撰写({len(episode_ids)}集): {time.time() - start_time:.2f}s")
    print_result(ret, "剧本撰写结果")
    return ret


async def step_6_regenerate_single_episode_outline(ctx, creativity, args,
                                                     episode_outline_rsp: pb.GenerateEpisodeOutlineRsp = None,
                                                     target_episode_id: int = 2) -> pb.RegenerateEpisodeOutlineSingleRsp:
    """
    第5步：重新生成单集大纲
    
    输入：原始集大纲、修改意见、generated_data（包含所有集大纲用于提取前后集上下文）
    输出：重新生成的单集大纲
    
    Args:
        target_episode_id: 要重新生成的集ID（0-based），默认为2（即第3集）
    """
    print_separator(f"第5步：重新生成单集大纲（第{target_episode_id + 1}集）")
    
    if episode_outline_rsp is None:
        print("[警告] 没有集大纲数据，无法重新生成单集大纲。请先运行第3步。")
        print("[提示] 将使用模拟数据进行测试...")
        # 构造模拟的集大纲数据用于独立测试
        mock_episodes = []
        for i in range(10):
            ep = pb.EpisodeOutline.Episode(
                season_id=0,
                episode_id=i,
                chapter_from=i * 10 + 1,
                chapter_to=(i + 1) * 10,
                content=f"第{i+1}集大纲：这是模拟的第{i+1}集剧情内容。"
            )
            mock_episodes.append(ep)
        episode_outline = pb.EpisodeOutline(
            seasons=[pb.EpisodeOutline.Season(episodes=mock_episodes, season_id=0)]
        )
    else:
        episode_outline = episode_outline_rsp.episode_outline
    
    # 获取目标集的原始大纲内容
    target_episode_content = ""
    for season in episode_outline.seasons:
        for ep in season.episodes:
            if ep.episode_id == target_episode_id:
                target_episode_content = ep.content
                break
    
    if not target_episode_content:
        target_episode_content = f"第{target_episode_id + 1}集的原始大纲内容（模拟数据）"
    
    # 构造请求
    request = pb.RegenerateEpisodeOutlineSingleByCreativityReq(
        story_info=pb.StoryInfo(
            plot_type=args.plot_type
        ),
        generate_input=pb.GenerateInputByCreativity(
            core_story=TEST_CORE_STORY,
            topic=TEST_TOPIC,
            world_view="",
            role_setting=TEST_ROLE_SETTING,
            reference=TEST_REFERENCE,
            common=pb.GenerateInputCommon(
                season_nums=DEFAULT_SEASON_NUMS,
                episode_nums=args.episode_nums,
                min_word_count_per_episode=DEFAULT_MIN_WORD_COUNT,
                max_word_count_per_episode=DEFAULT_MAX_WORD_COUNT
            )
        ),
        episode_outline=target_episode_content,
        episode_id=target_episode_id,
        suggestion="请让这一集的冲突更加激烈，增加一个反转情节，让观众更有代入感",
        generated_data=pb.GeneratedData(
            episode_outline=episode_outline
        )
    )
    
    print(f"[参数] target_episode_id={target_episode_id}, suggestion='{request.suggestion}'")
    print(f"[参数] episode_outline长度={len(target_episode_content)}字符")
    print(f"[参数] generated_data中共有{sum(len(s.episodes) for s in episode_outline.seasons)}集大纲")
    
    start_time = time.time()
    ret = await creativity.RegenerateEpisodeOutlineSingle(ctx, request)
    print(f"[耗时] 重新生成单集大纲: {time.time() - start_time:.2f}s")
    print_result(ret, "重新生成单集大纲结果")
    
    # 验证返回结果非空
    if ret.episode_outline:
        print(f"✅ 重新生成成功，结果长度: {len(ret.episode_outline)} 字符")
    else:
        print("⚠️ 返回结果为空，请检查输入参数")
    
    return ret


# ========== 全流程运行 ==========

async def run_full_pipeline(ctx, creativity, args):
    """
    全流程运行：剧本规划 -> 故事大纲 -> 集大纲 -> 全部剧本 -> 重新生成单集大纲
    """
    print_separator("开始全流程测试")
    total_start = time.time()
    
    # 第1步：剧本规划
    script_proposal_rsp = await step_1_script_proposal(ctx, creativity, args)
    
    
    # 第2步：故事大纲
    story_outline_rsp = await step_3_story_outline(ctx, creativity, args, script_proposal_rsp)
    
    # 第3步：集大纲生成
    episode_outline_rsp = await step_4_episode_outline(ctx, creativity, args, script_proposal_rsp, story_outline_rsp)
    
    if getattr(args, 'outline_only', False):
        print_separator("逐集大纲生成、三轮审校及评分择优完成")
        print(f"[总耗时] {time.time() - total_start:.2f}s")
        return {"script_proposal": script_proposal_rsp,
                "story_outline": story_outline_rsp,
                "episode_outline": episode_outline_rsp}

    # 第4步：剧本撰写（生成全部集数）
    drama_rsp = await step_5_drama(ctx, creativity, args,
                                    story_outline=story_outline_rsp.story_outline,
                                    episode_outline_rsp=episode_outline_rsp,
                                    episode_ids=list(range(args.episode_nums)))
    
    # Do not report a partial generation as a completed run.
    actual_ids = [episode.episode_id for season in drama_rsp.result.seasons for episode in season.episodes]
    if sorted(actual_ids) != list(range(args.episode_nums)):
        raise RuntimeError(f"剧本生成不完整：期望 {args.episode_nums} 集，实际集 ID 为 {actual_ids}；已完成结果保留在输出目录。")

    # 第5步：重新生成单集大纲（测试第3集）
    regenerate_single_rsp = await step_6_regenerate_single_episode_outline(
        ctx, creativity, args, episode_outline_rsp=episode_outline_rsp, target_episode_id=2)
    
    print_separator("全流程测试完成")
    print(f"[总耗时] {time.time() - total_start:.2f}s")
    
    return {
        "script_proposal": script_proposal_rsp,
        "story_outline": story_outline_rsp,
        "episode_outline": episode_outline_rsp,
        "drama": drama_rsp,
        "regenerate_single": regenerate_single_rsp
    }


async def run_from_step(ctx, creativity, args, start_step: str):
    """
    从指定步骤开始运行到结束
    """
    steps = ["script_proposal", "story_outline", "episode_outline", "drama", "regenerate_single"]
    start_idx = steps.index(start_step)
    
    print_separator(f"从第{start_idx + 1}步 [{start_step}] 开始运行")
    total_start = time.time()
    
    script_proposal_rsp = None
    story_outline_rsp = None
    episode_outline_rsp = None
    
    for i in range(start_idx, len(steps)):
        step = steps[i]
        if step == "script_proposal":
            script_proposal_rsp = await step_1_script_proposal(ctx, creativity, args)
        elif step == "story_outline":
            story_outline_rsp = await step_3_story_outline(ctx, creativity, args, script_proposal_rsp)
        elif step == "episode_outline":
            episode_outline_rsp = await step_4_episode_outline(ctx, creativity, args, script_proposal_rsp, story_outline_rsp)
        elif step == "drama":
            story_outline = story_outline_rsp.story_outline if story_outline_rsp else None
            await step_5_drama(ctx, creativity, args,
                              story_outline=story_outline,
                              episode_outline_rsp=episode_outline_rsp,
                              episode_ids=list(range(min(10, args.episode_nums))))
        elif step == "regenerate_single":
            await step_6_regenerate_single_episode_outline(
                ctx, creativity, args, episode_outline_rsp=episode_outline_rsp)
    
    print_separator("运行完成")
    print(f"[总耗时] {time.time() - total_start:.2f}s")


# ========== 主入口 ==========

def parse_args():
    parser = argparse.ArgumentParser(description="创作链路测试脚本")
    
    # 运行模式
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="全流程运行")
    group.add_argument("--step", type=str, choices=[
        "script_proposal", "story_outline", "episode_outline", "drama", "regenerate_single"
    ], help="单独测试某个环节")
    group.add_argument("--from", type=str, dest="from_step", choices=[
        "script_proposal", "story_outline", "episode_outline", "drama", "regenerate_single"
    ], help="从某个环节开始跑到尾")
    
    parser.add_argument("--outline-only", action="store_true", help="与 --all 配合，从头生成到逐集大纲审校择优后停止")

    # 配置参数
    parser.add_argument("--story_id", type=str, default=DEFAULT_STORY_ID, help="项目ID（用于区分本地故事数据）")
    parser.add_argument("--episode_nums", type=int, default=DEFAULT_EPISODE_NUMS, help="总集数")
    parser.add_argument("--plot_type", type=str, default="SHORT_CARTOON",
                        choices=["SHORT", "SHORT_CARTOON", "CARTOON"],
                        help="剧集类型")
    parser.add_argument(
        "--disable_plot_retrieval",
        action="store_true",
        help="关闭全部桥段库检索，并从相关 Prompt 中移除桥段参考段落",
    )
    
    parser.add_argument("--summary-memory-baseline", action="store_true", help="只使用此前各集梗概作为情节记忆，默认关闭")
    from service.drama_by_creativity.eventline_refinement import parse_refinement_stages
    parser.add_argument(
        "--eventline-stages", nargs="+", metavar="STAGE",
        help="按传入顺序细化阶段，如 开端 发展 高潮 结局；也支持逗号分隔。默认沿用环境变量或开端、发展、高潮、结局",
    )
    args = parser.parse_args()
    if args.outline_only and not args.all:
        parser.error("--outline-only 必须与 --all 一起使用")
    if args.eventline_stages is not None:
        try:
            args.eventline_stages = parse_refinement_stages(" ".join(args.eventline_stages))
        except ValueError as error:
            parser.error(str(error))
    return args


def get_plot_type(plot_type_str: str):
    """将字符串转为PlotType枚举"""
    mapping = {
        "SHORT": pb.PlotType.SHORT,
        "SHORT_CARTOON": pb.PlotType.SHORT_CARTOON,
        "CARTOON": pb.PlotType.CARTOON,
    }
    return mapping.get(plot_type_str, pb.PlotType.SHORT_CARTOON)


async def main():
    args = parse_args()
    os.environ["DRAMA_OUTLINE_ONLY"] = "1" if args.outline_only else "0"
    if args.eventline_stages is not None:
        os.environ["DRAMA_EVENTLINE_STAGES"] = ",".join(args.eventline_stages)
        os.environ["DRAMA_EVENTLINE_REFINEMENT_ENABLED"] = "1"
    if args.summary_memory_baseline:
        os.environ["DRAMA_SUMMARY_MEMORY_BASELINE"] = "1"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ.setdefault("DRAMA_OUTPUT_DIR", str(Path(project_root) / "output" / time.strftime("local_%Y%m%d_%H%M%S")))
    # Local-only entry: no external trope retrieval.
    os.environ["DRAMA_DISABLE_PLOT_RETRIEVAL"] = "1"
    args.plot_type = get_plot_type(args.plot_type)
    if args.disable_plot_retrieval:
        os.environ["DRAMA_DISABLE_PLOT_RETRIEVAL"] = "1"
    
    # 初始化
    ctx = context.Context()
    creativity = create_creativity()
    
    print(
        f"\n[配置] story_id={args.story_id}, episode_nums={args.episode_nums}, "
        f"plot_type={args.plot_type}, disable_plot_retrieval={args.disable_plot_retrieval}"
    )
    
    if args.all:
        results = await run_full_pipeline(ctx, creativity, args)
        from drama_local.runtime import atomic_json
        atomic_json(Path(os.environ["DRAMA_OUTPUT_DIR"]) / "pipeline_result.json",
                    {name: value.to_dict() for name, value in results.items()})
    elif args.step:
        # 单独测试某个环节
        if args.step == "script_proposal":
            await step_1_script_proposal(ctx, creativity, args)
        elif args.step == "story_outline":
            await step_3_story_outline(ctx, creativity, args)
        elif args.step == "episode_outline":
            await step_4_episode_outline(ctx, creativity, args)
        elif args.step == "drama":
            print("[提示] 单独测试剧本撰写需要先有集大纲数据。")
            print("[提示] 建议使用 --from episode_outline 从集大纲开始运行。")
        elif args.step == "regenerate_single":
            # 单独测试重新生成单集大纲（使用模拟数据）
            await step_6_regenerate_single_episode_outline(ctx, creativity, args)
    elif args.from_step:
        await run_from_step(ctx, creativity, args, args.from_step)


if __name__ == "__main__":
    asyncio.run(main())
