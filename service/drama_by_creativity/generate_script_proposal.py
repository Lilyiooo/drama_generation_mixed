import time
import random
from drama_local.runtime import logger
from drama_local import runtime as context
from drama_local import models as pb

from .utils import *
from .llm_inference import llm_inference_json
from .prompts import GENERATE_WORLD_VIEW_PROMPT, GENERATE_ROLE_SETTING_PROMPT, GENERATE_PLOT_POINT_PROMPT
from .realtime_output import save_realtime

import jinja2
from .local_llm import LLM

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def transfer_worldview_to_str(world_view_json):
    """ 将世界观信息转为字符串形式 """

    world_view_str = world_view_json.get("world_view", "未正确解析世界观")
    return world_view_str

def transfer_role_to_str(role_json):
    """ 将角色信息转为字符串形式 """

    all_role_str = ""

    for role_info in role_json:
        role_name = role_info.get("name", "未正确解析角色名")
        role_type = role_info.get("role", "未正确解析角色类型")
        role_age = role_info.get("age", "未正确解析角色年龄")
        role_gender = role_info.get("gender", "未正确解析角色性别")
        role_appearance = role_info.get("appearance", "未正确解析角色外貌")
        role_personality = role_info.get("personality", "未正确解析角色性格")
        role_motivation = role_info.get("motivation", "未正确解析角色动机")
        role_background = role_info.get("background", "未正确解析角色背景")
        role_relationships = role_info.get("relationships", "未正确解析角色关系")
        role_languagetype = role_info.get("languagetype", "未正确解析角色语言风格")
        role_core = role_info.get("rolecore", "未正确解析角色内核")
        role_others = role_info.get("others", "未正确解析角色其他信息")

        all_role_str += f"#### {role_name}\n"
        all_role_str += f"性别：{role_gender} | 年龄：{role_age} | 类型：{role_type}\n\n"
        all_role_str += f"#### 角色内核\n{role_core}\n\n"
        all_role_str += f"#### 外貌\n{role_appearance}\n\n"
        all_role_str += f"#### 性格\n{role_personality}\n\n"
        all_role_str += f"#### 动机\n{role_motivation}\n\n"
        all_role_str += f"#### 背景\n{role_background}\n\n"
        all_role_str += f"#### 角色关系\n{role_relationships}\n\n"
        all_role_str += f"#### 语言风格\n{role_languagetype}"

        all_role_str += "\n\n---\n\n"

    return all_role_str[:-7]


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
        
        if plot.get("episode_range", [1, 1]) != []:
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


async def generate_script_proposal(ctx: context.Context, request: pb.GenerateScriptProposalByCreativityReq) -> pb.GenerateScriptProposalRsp:
    # 检查输入是否为空
    if request.generate_input.core_story == "":
        logger.error_context(ctx, f"基于创意生成剧本提案的输入”创意和想法“输入为空。")

    # 获取剧集类型
    drama_type = get_drama_type_name_by_plot_type(request.story_info.plot_type)

    # 设置集数
    try:
        episode_nums = int(request.generate_input.common.episode_nums)
    except Exception as e:
        logger.error_context(ctx, f"剧集类型为{drama_type}，但正确获取到指定集数，默认值指定为60集。")
        episode_nums = 60

    # 定义模型
    world_view_config = get_model_config("world_view_model")
    world_view_llm_service = LLM(
        model=world_view_config["model"],
        system_prompt=world_view_config["system_prompt"],
        temperature=world_view_config["temperature"],
        top_p=world_view_config["top_p"],
        top_k=world_view_config["top_k"],
        max_length=world_view_config["max_length"],
        max_new_tokens=world_view_config["max_new_tokens"],
        template=jinja2.Template(GENERATE_WORLD_VIEW_PROMPT)
    )

    role_config = get_model_config("role_model")
    role_llm_service = LLM(
        model=role_config["model"],
        system_prompt=role_config["system_prompt"],
        temperature=role_config["temperature"],
        top_p=role_config["top_p"],
        top_k=role_config["top_k"],
        max_length=role_config["max_length"],
        max_new_tokens=role_config["max_new_tokens"],
        template=jinja2.Template(GENERATE_ROLE_SETTING_PROMPT)
    )

    plot_point_config = get_model_config("plot_point_model")
    plot_point_llm_service = LLM(
        model=plot_point_config["model"],
        system_prompt=plot_point_config["system_prompt"],
        temperature=plot_point_config["temperature"],
        top_p=plot_point_config["top_p"],
        top_k=plot_point_config["top_k"],
        max_length=plot_point_config["max_length"],
        max_new_tokens=plot_point_config["max_new_tokens"],
        template=jinja2.Template(GENERATE_PLOT_POINT_PROMPT)
    )

    # 生成世界观
    logger.info_context(ctx, f"开始基于创意生成世界观。")
    world_view_params = {
        "Topic": request.generate_input.topic or "暂无",
        "WorldView": request.generate_input.world_view or "暂无",
        "StoryOutline": request.generate_input.core_story or "暂无",
        "RoleInfo": request.generate_input.role_setting or "暂无",
        "Reference": request.generate_input.reference or "暂无",
        "DramaType": drama_type
    }
    logger.info_context(ctx, f"[{world_view_config['model']}] 生成世界观的query为:\n{world_view_params}")

    start_world_view_time = time.time()
    world_view_ret_json = await llm_inference_json(
        params=world_view_params,
        llm_service=world_view_llm_service,
        task_name="生成世界观",
        ctx=ctx
    )
    save_realtime(["01_script_proposal", "01_world_view.json"], world_view_ret_json)

    final_world_view = transfer_worldview_to_str(world_view_ret_json)
    logger.info_context(ctx, f"任务：世界观 成功生成结果:\n{final_world_view}\n耗时：{time.time() - start_world_view_time}s。")


    # 生成人设
    logger.info_context(ctx, f"开始基于创意生成人设信息。")
    role_params = {
        "Topic": request.generate_input.topic or "暂无",
        "WorldView": final_world_view or "暂无",
        "StoryOutline": request.generate_input.core_story or "暂无",
        "RoleInfo": request.generate_input.role_setting or "暂无",
        "Reference": request.generate_input.reference or "暂无",
        "RoleNums": "3至5",
        "DramaType": drama_type
    }
    logger.info_context(ctx, f"[{role_config['model']}] 生成人设的query为:\n{role_params}")

    start_role_time = time.time()
    role_ret_json = await llm_inference_json(
        params=role_params,
        llm_service=role_llm_service,
        task_name="生成角色信息",
        ctx=ctx
    )
    save_realtime(["01_script_proposal", "02_role_setting.json"], role_ret_json)

    final_role = transfer_role_to_str(role_ret_json)
    logger.info_context(ctx, f"任务：角色 成功生成结果:\n{final_role}\n耗时：{time.time() - start_role_time}s。")

    
    # 生成卡点
    logger.info_context(ctx, f"开始基于创意生成卡点信息。")
    plot_point_params = {
        "Topic": request.generate_input.topic or "暂无",
        "WorldView": final_world_view or "暂无",
        "StoryOutline": request.generate_input.core_story or "暂无",
        "RoleInfo": final_role or "暂无",
        "Reference": request.generate_input.reference or "暂无",
        "DramaType": drama_type,
        "EpisodeNums": str(episode_nums)
    }
    logger.info_context(ctx, f"[{plot_point_config['model']}] 生成卡点的query为:\n{plot_point_params}")

    start_plot_point_time = time.time()
    plot_point_ret_json = await llm_inference_json(
        params=plot_point_params,
        llm_service=plot_point_llm_service,
        task_name="生成卡点信息",
        ctx=ctx
    )
    save_realtime(["01_script_proposal", "03_plot_point.json"], plot_point_ret_json)
    plot_planning_pb = transfer_plot_point_to_pb(plot_point_ret_json)
    logger.info_context(ctx, f"任务：卡点 成功生成结果:\n{plot_point_ret_json}\n耗时：{time.time() - start_plot_point_time}s。")

    script_propsoal = pb.SingleScriptProposal(
        title="整体卡点规划",
        plot_planning=plot_planning_pb,
        point_planning=pb.SingleScriptProposal.PointPlanning(points=[pb.SingleScriptProposal.PointPlanning.Point(point_id=0, point_name="第一季", start_plot_id=0, end_plot_id=episode_nums-1)], description=plot_point_ret_json.get("stuck_point_desc", "卡点说明解析失败")),
        storylines=[],
        other_adaptation_proposal="",
        season_id=-1
    )

    story_assets = pb.StoryOutline(story_outline=[""], role_info=[final_role], world_building=[final_world_view])
    save_realtime(["01_script_proposal", "story_assets.json"], story_assets.to_dict())
    return pb.GenerateScriptProposalRsp(
        story_outline=story_assets,
        script_proposal=pb.ScriptProposal(script_proposals=[script_propsoal])
    )
