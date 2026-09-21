#!/usr/bin/env python3
"""
DramaByFiction 接口测试入口。

用法:
    python -m service.drama_by_fiction.tests.run_test all          # 完整流程，全部调用
    python -m service.drama_by_fiction.tests.run_test 1            # 只测试第1个接口（从文件加载依赖）
    python -m service.drama_by_fiction.tests.run_test 1 3 5        # 测试第1、3、5个接口
    python -m service.drama_by_fiction.tests.run_test              # 默认：完整流程

接口编号:
    1 - GenerateScriptProposal    生成剧本策划
    2 - RegenerateScriptProposal  重新生成剧本策划
    3 - GeneratePointPlan         生成卡点规划
    4 - GenerateStoryOutline      生成故事大纲
    5 - RegenerateStoryOutline    重新生成故事大纲
    6 - GenerateEpisodeOutline    生成集大纲
    7 - GenerateEpisodeScenePlan  生成分场规划
    8 - GenerateDrama             生成剧本内容
    9 - GenerateSceneOutline      生成指定单场规划
   10 - GenerateSceneDrama        生成指定单场剧本
   11 - GenerateSceneOutlinePrompt 获取单场规划提示词
   12 - GenerateSceneDramaPrompt   获取单场剧本提示词
"""

import os
import sys
import time
import asyncio

from trpc import context, config
from trpc.plugin import plugin
from trpc_cos import Client as CosClient
from trpc_script_drama_operator import pb

from service.drama_by_fiction import DramaByFiction
from service.drama_by_fiction.tests.test_helper import (
    format_print,
    save_pb_json,
    load_pb_json,
    save_and_print,
)


def print_estimate_time(loop, drama, ctx, rpc_name, request):
    """打印预估时间"""
    from google.protobuf.any_pb2 import Any as AnyPb

    any_request = AnyPb()
    any_request.Pack(request)
    req = pb.GetEstimatedTimeReq(rpc_name=rpc_name, request=any_request)
    time_rsp = loop.run_until_complete(drama.GetEstimatedTime(ctx, req))
    format_print(time_rsp)


def run_and_time(loop, coro):
    """执行异步调用并返回 (结果, 耗时秒数)"""
    start = time.time()
    result = loop.run_until_complete(coro)
    elapsed = time.time() - start
    return result, elapsed


def main():
    # ========= 初始化 ========= #
    config.load_global_config("trpc_python.yaml", "utf-8")
    plugin.setup_master()

    cos = CosClient("trpc.script.drama_operator.CommonCos")
    drama = DramaByFiction(cos)
    ctx = context.Context()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # ========= 测试参数配置 ========= #
    story_id = "story0003"
    story_name = "青山"
    plot_type = pb.PlotType.CARTOON

    novel_id = "fiction-dqouSbnC"
    season_nums = 3
    episode_nums = 26
    duration = 20
    min_word_count_per_episode = 1000
    max_word_count_per_episode = 10000

    # 测试数据保存目录
    test_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), ".cache", "test_data", story_id
    )
    os.makedirs(test_data_dir, exist_ok=True)

    # ========= 公共请求参数 ========= #
    story_info = pb.StoryInfo(
        story_id=story_id,
        story_name=story_name,
        plot_type=plot_type,
    )
    common = pb.GenerateInputCommon(
        season_nums=season_nums,
        episode_nums=episode_nums,
        duration=duration,
        min_word_count_per_episode=min_word_count_per_episode,
        max_word_count_per_episode=max_word_count_per_episode,
        adaptation_approach="",
    )
    generate_input = pb.GenerateInputByFiction(
        novel_id=novel_id,
        common=common,
        suggestion="",
    )

    # 保存公共参数
    save_pb_json(test_data_dir, "00_story_info.json", story_info)
    save_pb_json(test_data_dir, "00_generate_input.json", generate_input)

    # ========= 接口测试选择 ========= #
    args = sys.argv[1:]
    if not args or "all" in args:
        run_steps = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
    else:
        run_steps = {int(x) for x in args}

    run_all = run_steps == {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

    elapsed_times = {}  # 记录每个接口的实际耗时

    # --------------------------------------------- #
    # 1. 生成剧本策划
    if 1 in run_steps:
        request = pb.GenerateScriptProposalByFictionReq(
            story_info=story_info,
            generate_input=generate_input,
        )
        print_estimate_time(loop, drama, ctx, "GenerateScriptProposal", request)
        script_proposals, elapsed = run_and_time(
            loop, drama.GenerateScriptProposal(ctx, request)
        )
        elapsed_times["1. GenerateScriptProposal"] = elapsed
        save_and_print(
            test_data_dir,
            f"1. GenerateScriptProposal（生成剧本策划）  ⏱ {elapsed:.1f}s",
            "01_generate_proposal_req.json",
            "01_generate_proposal_rsp.json",
            request,
            script_proposals,
        )

    # --------------------------------------------- #
    # 2. 重新生成剧本策划
    if 2 in run_steps:
        if not run_all or 1 not in run_steps:
            script_proposals = load_pb_json(
                test_data_dir,
                "01_generate_proposal_rsp.json",
                pb.GenerateScriptProposalRsp,
            )
        regenerate_data = pb.RegenerateScriptProposal(
            suggestion="增加主角的戏份，强化主线剧情"
        )
        request = pb.RegenerateScriptProposalByFictionReq(
            story_info=story_info,
            generate_input=generate_input,
            regenerate_data=regenerate_data,
        )
        print_estimate_time(loop, drama, ctx, "RegenerateScriptProposal", request)
        script_proposals, elapsed = run_and_time(
            loop, drama.RegenerateScriptProposal(ctx, request)
        )
        elapsed_times["2. RegenerateScriptProposal"] = elapsed
        save_and_print(
            test_data_dir,
            f"2. RegenerateScriptProposal（重新生成剧本策划）  ⏱ {elapsed:.1f}s",
            "02_regenerate_proposal_req.json",
            "02_regenerate_proposal_rsp.json",
            request,
            script_proposals,
        )

    # --------------------------------------------- #
    # 3. 生成卡点规划
    if 3 in run_steps:
        if not run_all or 1 not in run_steps:
            script_proposals = load_pb_json(
                test_data_dir,
                "01_generate_proposal_rsp.json",
                pb.GenerateScriptProposalRsp,
            )
        regenerate_data = pb.RegenerateScriptProposal(
            suggestion="增加主角的戏份，强化主线剧情",
            story_outline=script_proposals.story_outline,
        )
        request = pb.RegenerateScriptProposalByFictionReq(
            story_info=story_info,
            generate_input=generate_input,
            regenerate_data=regenerate_data,
        )
        print_estimate_time(loop, drama, ctx, "GeneratePointPlan", request)
        script_proposals, elapsed = run_and_time(
            loop, drama.GeneratePointPlan(ctx, request)
        )
        elapsed_times["3. GeneratePointPlan"] = elapsed
        save_and_print(
            test_data_dir,
            f"3. GeneratePointPlan（生成卡点规划）  ⏱ {elapsed:.1f}s",
            "03_generate_point_plan_req.json",
            "03_generate_point_plan_rsp.json",
            request,
            script_proposals,
        )

    # --------------------------------------------- #
    # 4. 生成故事大纲
    if 4 in run_steps:
        if not run_all or 3 not in run_steps:
            script_proposals = load_pb_json(
                test_data_dir,
                "03_generate_point_plan_rsp.json",
                pb.GenerateScriptProposalRsp,
            )
        generate_data = pb.GenerateStoryOutline(
            story_outline=script_proposals.story_outline,
            script_proposal=script_proposals.script_proposal,
        )
        request = pb.GenerateStoryOutlineByFictionReq(
            story_info=story_info,
            generate_input=generate_input,
            generate_data=generate_data,
        )
        print_estimate_time(loop, drama, ctx, "GenerateStoryOutline", request)
        story_outline, elapsed = run_and_time(
            loop, drama.GenerateStoryOutline(ctx, request)
        )
        elapsed_times["4. GenerateStoryOutline"] = elapsed
        save_and_print(
            test_data_dir,
            f"4. GenerateStoryOutline（生成故事大纲）  ⏱ {elapsed:.1f}s",
            "04_generate_story_outline_req.json",
            "04_generate_story_outline_rsp.json",
            request,
            story_outline,
        )

    # --------------------------------------------- #
    # 5. 重新生成故事大纲
    if 5 in run_steps:
        if not run_all or 4 not in run_steps:
            script_proposals = load_pb_json(
                test_data_dir,
                "03_generate_point_plan_rsp.json",
                pb.GenerateScriptProposalRsp,
            )
        generate_data = pb.GenerateStoryOutline(
            story_outline=script_proposals.story_outline,
            script_proposal=script_proposals.script_proposal,
        )
        regenerate_data = pb.RegenerateStoryOutline(
            suggestion="增加主角的戏份，强化主线剧情",
            season_id=0,
            generate_data=generate_data,
        )
        request = pb.RegenerateStoryOutlineByFictionReq(
            story_info=story_info,
            generate_input=generate_input,
            regenerate_data=regenerate_data,
        )
        print_estimate_time(loop, drama, ctx, "RegenerateStoryOutline", request)
        re_story_outline, elapsed = run_and_time(
            loop, drama.RegenerateStoryOutline(ctx, request)
        )
        elapsed_times["5. RegenerateStoryOutline"] = elapsed
        save_and_print(
            test_data_dir,
            f"5. RegenerateStoryOutline（重新生成故事大纲）  ⏱ {elapsed:.1f}s",
            "05_regenerate_story_outline_req.json",
            "05_regenerate_story_outline_rsp.json",
            request,
            re_story_outline,
        )

    # --------------------------------------------- #
    # 6. 生成集大纲
    if 6 in run_steps:
        if not run_all or 3 not in run_steps:
            script_proposals = load_pb_json(
                test_data_dir,
                "03_generate_point_plan_rsp.json",
                pb.GenerateScriptProposalRsp,
            )
        if not run_all or 4 not in run_steps:
            story_outline = load_pb_json(
                test_data_dir,
                "04_generate_story_outline_rsp.json",
                pb.GenerateStoryOutlineRsp,
            )
        generated_data = pb.GeneratedData(
            script_proposal=script_proposals.script_proposal,
            story_outline=story_outline.story_outline,
        )
        select_range = [pb.SelectRange(season_id=0, episode_ids=list(range(3)))]
        generate_data = pb.GenerateEpisodeOutline(
            suggestion="增加主角的戏份，强化主线剧情",
            select_range=select_range,
        )
        request = pb.GenerateEpisodeOutlineByFictionReq(
            story_info=story_info,
            generate_input=generate_input,
            generated_data=generated_data,
            generate_data=generate_data,
        )
        print_estimate_time(loop, drama, ctx, "GenerateEpisodeOutline", request)
        episode_outline, elapsed = run_and_time(
            loop, drama.GenerateEpisodeOutline(ctx, request)
        )
        elapsed_times["6. GenerateEpisodeOutline"] = elapsed
        save_and_print(
            test_data_dir,
            f"6. GenerateEpisodeOutline（生成集大纲）  ⏱ {elapsed:.1f}s",
            "06_generate_episode_outline_req.json",
            "06_generate_episode_outline_rsp.json",
            request,
            episode_outline,
        )

    # --------------------------------------------- #
    # 7. 生成分场规划
    if 7 in run_steps:
        if not run_all or 3 not in run_steps:
            script_proposals = load_pb_json(
                test_data_dir,
                "03_generate_point_plan_rsp.json",
                pb.GenerateScriptProposalRsp,
            )
        if not run_all or 4 not in run_steps:
            story_outline = load_pb_json(
                test_data_dir,
                "04_generate_story_outline_rsp.json",
                pb.GenerateStoryOutlineRsp,
            )
        if not run_all or 6 not in run_steps:
            episode_outline = load_pb_json(
                test_data_dir,
                "06_generate_episode_outline_rsp.json",
                pb.GenerateEpisodeOutlineRsp,
            )
        generated_data = pb.GeneratedData(
            script_proposal=script_proposals.script_proposal,
            story_outline=story_outline.story_outline,
            episode_outline=episode_outline.episode_outline,
        )
        select_range = [pb.SelectRange(season_id=0, episode_ids=list(range(3)))]
        generate_data = pb.GenerateEpisodeOutline(
            suggestion="增加主角的戏份，强化主线剧情",
            select_range=select_range,
        )
        request = pb.GenerateEpisodeOutlineByFictionReq(
            story_info=story_info,
            generate_input=generate_input,
            generated_data=generated_data,
            generate_data=generate_data,
        )
        print_estimate_time(loop, drama, ctx, "GenerateEpisodeScenePlan", request)
        episode_outline, elapsed = run_and_time(
            loop, drama.GenerateEpisodeScenePlan(ctx, request)
        )
        elapsed_times["7. GenerateEpisodeScenePlan"] = elapsed
        save_and_print(
            test_data_dir,
            f"7. GenerateEpisodeScenePlan（生成分场规划）  ⏱ {elapsed:.1f}s",
            "07_generate_episode_scene_plan_req.json",
            "07_generate_episode_scene_plan_rsp.json",
            request,
            episode_outline,
        )

    # --------------------------------------------- #
    # 8. 生成剧本内容
    if 8 in run_steps:
        if not run_all or 3 not in run_steps:
            script_proposals = load_pb_json(
                test_data_dir,
                "03_generate_point_plan_rsp.json",
                pb.GenerateScriptProposalRsp,
            )
        if not run_all or 4 not in run_steps:
            story_outline = load_pb_json(
                test_data_dir,
                "04_generate_story_outline_rsp.json",
                pb.GenerateStoryOutlineRsp,
            )
        if not run_all or 7 not in run_steps:
            episode_outline = load_pb_json(
                test_data_dir,
                "07_generate_episode_scene_plan_rsp.json",
                pb.GenerateEpisodeOutlineRsp,
            )
        select_range = [
            pb.GenerateDrama.SelectRange(season_id=0, episode_ids=list(range(3))),
        ]
        generate_data = pb.GenerateDrama(
            script_proposal=script_proposals.script_proposal,
            story_outline=story_outline.story_outline,
            episode_outline=episode_outline.episode_outline,
            suggestion="增加主角的戏份，强化主线剧情",
            select_range=select_range,
        )
        request = pb.GenerateDramaByFictionReq(
            story_info=story_info,
            generate_input=generate_input,
            generate_data=generate_data,
        )
        print_estimate_time(loop, drama, ctx, "GenerateDrama", request)
        scripts, elapsed = run_and_time(loop, drama.GenerateDrama(ctx, request))
        elapsed_times["8. GenerateDrama"] = elapsed
        save_and_print(
            test_data_dir,
            f"8. GenerateDrama（生成剧本内容）  ⏱ {elapsed:.1f}s",
            "08_generate_drama_req.json",
            "08_generate_drama_rsp.json",
            request,
            scripts,
        )

    # --------------------------------------------- #
    # 9. 生成指定单场规划
    if 9 in run_steps:
        # 加载前置依赖：剧本策划、故事大纲、分场规划
        if not run_all or 3 not in run_steps:
            script_proposals = load_pb_json(
                test_data_dir,
                "03_generate_point_plan_rsp.json",
                pb.GenerateScriptProposalRsp,
            )
        if not run_all or 4 not in run_steps:
            story_outline = load_pb_json(
                test_data_dir,
                "04_generate_story_outline_rsp.json",
                pb.GenerateStoryOutlineRsp,
            )
        if not run_all or 7 not in run_steps:
            episode_outline = load_pb_json(
                test_data_dir,
                "07_generate_episode_scene_plan_rsp.json",
                pb.GenerateEpisodeOutlineRsp,
            )

        generated_data = pb.GeneratedData(
            script_proposal=script_proposals.script_proposal,
            story_outline=story_outline.story_outline,
            episode_outline=episode_outline.episode_outline,
        )
        generate_data = pb.GenerateSceneOutline(
            season_id=0,
            episode_id=0,
            scene_id=0,
            suggestion="让本场冲突更加激烈，增加悬念感",
        )
        request = pb.GenerateSceneOutlineByFictionReq(
            story_info=story_info,
            generate_input=generate_input,
            generated_data=generated_data,
            generate_data=generate_data,
        )
        print_estimate_time(loop, drama, ctx, "GenerateSceneOutline", request)
        scene_outline_rsp, elapsed = run_and_time(
            loop, drama.GenerateSceneOutline(ctx, request)
        )
        elapsed_times["9. GenerateSceneOutline"] = elapsed
        save_and_print(
            test_data_dir,
            f"9. GenerateSceneOutline（生成指定单场规划）  ⏱ {elapsed:.1f}s",
            "09_generate_scene_outline_req.json",
            "09_generate_scene_outline_rsp.json",
            request,
            scene_outline_rsp,
        )

    # --------------------------------------------- #
    # 10. 生成指定单场剧本
    if 10 in run_steps:
        # 加载前置依赖：剧本策划、故事大纲、分场规划、已有剧本
        if not run_all or 3 not in run_steps:
            script_proposals = load_pb_json(
                test_data_dir,
                "03_generate_point_plan_rsp.json",
                pb.GenerateScriptProposalRsp,
            )
        if not run_all or 4 not in run_steps:
            story_outline = load_pb_json(
                test_data_dir,
                "04_generate_story_outline_rsp.json",
                pb.GenerateStoryOutlineRsp,
            )
        if not run_all or 7 not in run_steps:
            episode_outline = load_pb_json(
                test_data_dir,
                "07_generate_episode_scene_plan_rsp.json",
                pb.GenerateEpisodeOutlineRsp,
            )
        if not run_all or 8 not in run_steps:
            scripts = load_pb_json(
                test_data_dir,
                "08_generate_drama_rsp.json",
                pb.GenerateDramaRsp,
            )

        generated_data = pb.GeneratedData(
            script_proposal=script_proposals.script_proposal,
            story_outline=story_outline.story_outline,
            episode_outline=episode_outline.episode_outline,
            drama=scripts.result,
        )
        generate_data = pb.GenerateSceneDrama(
            season_id=0,
            episode_id=0,
            scene_id=0,
            suggestion="增强本场的戏剧张力，台词更有个性",
        )
        request = pb.GenerateSceneDramaByFictionReq(
            story_info=story_info,
            generate_input=generate_input,
            generated_data=generated_data,
            generate_data=generate_data,
        )
        print_estimate_time(loop, drama, ctx, "GenerateSceneDrama", request)
        scene_drama_rsp, elapsed = run_and_time(
            loop, drama.GenerateSceneDrama(ctx, request)
        )
        elapsed_times["10. GenerateSceneDrama"] = elapsed
        save_and_print(
            test_data_dir,
            f"10. GenerateSceneDrama（生成指定单场剧本）  ⏱ {elapsed:.1f}s",
            "10_generate_scene_drama_req.json",
            "10_generate_scene_drama_rsp.json",
            request,
            scene_drama_rsp,
        )

    # --------------------------------------------- #
    # 11. 获取单场规划提示词
    if 11 in run_steps:
        # 加载前置依赖：剧本策划、故事大纲、分场规划
        if not run_all or 3 not in run_steps:
            script_proposals = load_pb_json(
                test_data_dir,
                "03_generate_point_plan_rsp.json",
                pb.GenerateScriptProposalRsp,
            )
        if not run_all or 4 not in run_steps:
            story_outline = load_pb_json(
                test_data_dir,
                "04_generate_story_outline_rsp.json",
                pb.GenerateStoryOutlineRsp,
            )
        if not run_all or 7 not in run_steps:
            episode_outline = load_pb_json(
                test_data_dir,
                "07_generate_episode_scene_plan_rsp.json",
                pb.GenerateEpisodeOutlineRsp,
            )

        generated_data = pb.GeneratedData(
            script_proposal=script_proposals.script_proposal,
            story_outline=story_outline.story_outline,
            episode_outline=episode_outline.episode_outline,
        )
        generate_data = pb.GenerateSceneOutline(
            season_id=0,
            episode_id=0,
            scene_id=0,
            suggestion="让本场冲突更加激烈，增加悬念感",
        )
        request = pb.GenerateSceneOutlineByFictionReq(
            story_info=story_info,
            generate_input=generate_input,
            generated_data=generated_data,
            generate_data=generate_data,
        )
        scene_outline_prompt_rsp, elapsed = run_and_time(
            loop, drama.GenerateSceneOutlinePrompt(ctx, request)
        )
        elapsed_times["11. GenerateSceneOutlinePrompt"] = elapsed
        save_and_print(
            test_data_dir,
            f"11. GenerateSceneOutlinePrompt（获取单场规划提示词）  ⏱ {elapsed:.1f}s",
            "11_generate_scene_outline_prompt_req.json",
            "11_generate_scene_outline_prompt_rsp.json",
            request,
            scene_outline_prompt_rsp,
        )

    # --------------------------------------------- #
    # 12. 获取单场剧本提示词
    if 12 in run_steps:
        # 加载前置依赖：剧本策划、故事大纲、分场规划、已有剧本
        if not run_all or 3 not in run_steps:
            script_proposals = load_pb_json(
                test_data_dir,
                "03_generate_point_plan_rsp.json",
                pb.GenerateScriptProposalRsp,
            )
        if not run_all or 4 not in run_steps:
            story_outline = load_pb_json(
                test_data_dir,
                "04_generate_story_outline_rsp.json",
                pb.GenerateStoryOutlineRsp,
            )
        if not run_all or 7 not in run_steps:
            episode_outline = load_pb_json(
                test_data_dir,
                "07_generate_episode_scene_plan_rsp.json",
                pb.GenerateEpisodeOutlineRsp,
            )
        if not run_all or 8 not in run_steps:
            scripts = load_pb_json(
                test_data_dir,
                "08_generate_drama_rsp.json",
                pb.GenerateDramaRsp,
            )

        generated_data = pb.GeneratedData(
            script_proposal=script_proposals.script_proposal,
            story_outline=story_outline.story_outline,
            episode_outline=episode_outline.episode_outline,
            drama=scripts.result,
        )
        generate_data = pb.GenerateSceneDrama(
            season_id=0,
            episode_id=0,
            scene_id=0,
            suggestion="增强本场的戏剧张力，台词更有个性",
        )
        request = pb.GenerateSceneDramaByFictionReq(
            story_info=story_info,
            generate_input=generate_input,
            generated_data=generated_data,
            generate_data=generate_data,
        )
        scene_drama_prompt_rsp, elapsed = run_and_time(
            loop, drama.GenerateSceneDramaPrompt(ctx, request)
        )
        elapsed_times["12. GenerateSceneDramaPrompt"] = elapsed
        save_and_print(
            test_data_dir,
            f"12. GenerateSceneDramaPrompt（获取单场剧本提示词）  ⏱ {elapsed:.1f}s",
            "12_generate_scene_drama_prompt_req.json",
            "12_generate_scene_drama_prompt_rsp.json",
            request,
            scene_drama_prompt_rsp,
        )

    print(f"\n{'='*60}")
    print(f"  测试数据目录: {test_data_dir}")
    print(f"{'='*60}")

    if elapsed_times:
        print(f"\n{'='*60}")
        print(f"  耗时统计")
        print(f"{'='*60}")
        for step_name, secs in elapsed_times.items():
            print(f"  {step_name:45s} {secs:>7.1f}s")
        total = sum(elapsed_times.values())
        print(f"{'─'*60}")
        print(f"  {'总计':45s} {total:>7.1f}s")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
