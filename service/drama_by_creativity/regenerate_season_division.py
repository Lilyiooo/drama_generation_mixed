import time
from drama_local.runtime import logger
from drama_local import runtime as context
from drama_local import models as pb

from .utils import *
from .llm_inference import llm_inference_json
from .prompts import REGENERATE_PLOT_POINT_PROMPT

import jinja2
from .local_llm import LLM

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def transfer_plot_point_to_str(script_proposal):
    """ 将卡点信息转为字符串形式 """
    plot_point_str = ""

    for proposal in script_proposal:
        plot_point_str += f"### {proposal.title}\n\n"

        for plot in proposal.plot_planning:
            plot_point_str += f"#### 卡点 {plot.plot_id}: {plot.title}\n"
            plot_point_str += f"集数范围: 第 {plot.chapter_from} 集 - 第 {plot.chapter_to} 集\n"

            if plot.chapters:
                for chapter in plot.chapters:
                    plot_point_str += f"- 第 {chapter.chapter_num} 集: {chapter.content}\n"

            plot_point_str += "\n"

        if proposal.point_planning:
            plot_point_str += f"#### 卡点说明\n{proposal.point_planning.description}\n\n"

        plot_point_str += "---\n\n"

    return plot_point_str[:-5] if plot_point_str else "暂无卡点信息"


def transfer_plot_point_to_pb(plot_point_json):
    """ 将卡点信息转为pb格式 """
    plot_planning = plot_point_json.get("plot_planning", [])

    plot_planning_pb = []
    for plot in plot_planning:
        chapters = plot.get("plot_desc", [])
        chapters_pb = []
        for chapter in chapters:
            chapters_pb.append(
                pb.SingleScriptProposal.PlotPlanning.Chapter(
                    chapter_num=chapter.get("episode_id", 0),
                    content=chapter.get("plot_desc", "情节解析失败")
                )
            )
        plot_planning_pb.append(
            pb.SingleScriptProposal.PlotPlanning(
                plot_id=plot.get("plot_id", -1),
                chapter_from=plot.get("episode_range", [1, 1])[0],
                chapter_to=plot.get("episode_range", [1, 1])[1],
                title=plot.get("title", ""),
                chapters=chapters_pb
            )
        )

    return plot_planning_pb


async def regenerate_season_division(ctx: context.Context, request: pb.RegenerateScriptProposalByCreativityReq) -> pb.GenerateScriptProposalRsp:
    """
    重新生成卡点规划（分季卡点）
    根据用户的 suggestion 仅重新生成卡点设计，不重新生成世界观和人设
    """

    # 获取修改建议
    suggestion = request.regenerate_data.suggestion
    if suggestion == "":
        logger.warning_context(ctx, f"重新生成卡点规划的修改建议为空，将使用默认建议。")
        suggestion = "请优化卡点规划"

    # 获取已生成的数据
    story_outline = request.regenerate_data.story_outline
    script_proposal = request.regenerate_data.script_proposal.script_proposals if request.regenerate_data.script_proposal else []

    # 获取已生成的世界观和人设（直接复用，不重新生成）
    world_view = story_outline.world_building[0] if len(story_outline.world_building) > 0 else "暂无"
    role_info = story_outline.role_info[0] if len(story_outline.role_info) > 0 else "暂无"
    created_plot_point = transfer_plot_point_to_str(script_proposal)

    # 获取剧集类型
    if request.story_info.plot_type == pb.PlotType.CARTOON:
        drama_type = "动漫"
    elif request.story_info.plot_type == pb.PlotType.SHORT:
        drama_type = "短剧"
    elif request.story_info.plot_type == pb.PlotType.LONG:
        drama_type = "长剧"
    elif request.story_info.plot_type == pb.PlotType.SHORT_CARTOON:
        drama_type = "短番"
    else:
        drama_type = "短剧"

    # 设置集数
    try:
        episode_nums = int(request.generate_input.common.episode_nums)
    except Exception as e:
        logger.error_context(ctx, f"剧集类型为{drama_type}，但未正确获取到指定集数，默认值指定为60集。")
        episode_nums = 60

    # 定义卡点生成模型
    plot_point_config = get_model_config("plot_point_model")
    plot_point_llm_service = LLM(
        model=plot_point_config["model"],
        system_prompt=plot_point_config["system_prompt"],
        temperature=plot_point_config["temperature"],
        top_p=plot_point_config["top_p"],
        top_k=plot_point_config["top_k"],
        max_length=plot_point_config["max_length"],
        max_new_tokens=plot_point_config["max_new_tokens"],
        template=jinja2.Template(REGENERATE_PLOT_POINT_PROMPT)
    )

    # 重新生成卡点规划
    logger.info_context(ctx, f"开始根据修改建议重新生成卡点规划。")
    plot_point_params = {
        "Topic": request.generate_input.topic or "暂无",
        "WorldView": world_view or "暂无",
        "StoryOutline": request.generate_input.core_story or "暂无",
        "RoleInfo": role_info or "暂无",
        "Reference": request.generate_input.reference or "暂无",
        "DramaType": drama_type,
        "EpisodeNums": str(episode_nums),
        "CreatedPlotPoint": created_plot_point,
        "Suggestion": suggestion
    }
    logger.info_context(ctx, f"重新生成卡点规划的query为:\n{plot_point_params}")

    start_plot_point_time = time.time()
    plot_point_ret_json = await llm_inference_json(
        params=plot_point_params,
        llm_service=plot_point_llm_service,
        task_name="重新生成卡点规划",
        ctx=ctx
    )
    plot_planning_pb = transfer_plot_point_to_pb(plot_point_ret_json)
    logger.info_context(ctx, f"任务：重新生成卡点规划 成功生成结果:\n{plot_point_ret_json}\n耗时：{time.time() - start_plot_point_time}s。")

    # 构建返回结果
    script_propsoal = pb.SingleScriptProposal(
        title="整体卡点规划",
        plot_planning=plot_planning_pb,
        point_planning=pb.SingleScriptProposal.PointPlanning(
            points=[
                pb.SingleScriptProposal.PointPlanning.Point(
                    point_id=0,
                    point_name="第一季",
                    start_plot_id=0,
                    end_plot_id=episode_nums-1
                )
            ],
            description=plot_point_ret_json.get("stuck_point_desc", "卡点说明解析失败")
        ),
        storylines=[],
        other_adaptation_proposal="",
        season_id=-1
    )

    return pb.GenerateScriptProposalRsp(
        story_outline=pb.StoryOutline(
            story_outline=[""],
            role_info=[role_info],
            world_building=[world_view]
        ),
        script_proposal=pb.ScriptProposal(script_proposals=[script_propsoal])
    )
