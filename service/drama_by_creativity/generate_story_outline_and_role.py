import time
from drama_local.runtime import logger
from drama_local import runtime as context
from drama_local import models as pb

from .utils import *
from .llm_inference import llm_inference_json
from .prompts import GENERATE_OUTLINE_PROMPT, GENERATE_ROLE_PROMPT, GENERATE_OUTLINE_BY_SCRIPT_PROMPT

import jinja2
from .local_llm import LLM

from .eventline_refinement import refine_story_outline_eventlines
from .realtime_output import save_realtime

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

def transfer_outline_to_str(outline_json):
    """ 将故事大纲转为字符串形式 """

    outline_str = ""

    play_name = outline_json.get("title", "未正确解析剧名")
    core_story = outline_json.get("background", "未正确解析核心故事")
    framework = outline_json.get("framework", [])
    highlights = outline_json.get("highlights", "未正确解析高光时刻")
    summary = outline_json.get("summary", "未正确解析总结")

    outline_str = f"#### 剧名：{play_name}\n\n"
    outline_str += f"#### 核心背景\n{core_story}\n\n"
    outline_str += f"#### 本剧亮点\n{highlights}\n\n"
    outline_str += "#### 故事框架：\n"
    
    if len(framework) == 0:
        return outline_str

    for item in framework:
        stage = item.get("stage", "未正确解析阶段")
        events = item.get("event_list", [])
        outline_str += f"### {stage}\n"

        for event in events:
            event_content = event.get("event", "未正确解析事件")
            outline_str += f"- {event_content}\n"

    outline_str += f"\n\n## 总结\n{summary}"
    
    return outline_str
        
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


def transfer_script_proposal_to_str(script_proposal):
    """ 将整体剧本策划转为字符串形式 """
    
    if not script_proposal or not script_proposal.script_proposals:
        return "暂无整体剧本策划"
    
    result_str = ""
    
    for proposal in script_proposal.script_proposals:
        # 标题
        if proposal.title:
            result_str += f"### 剧名：{proposal.title}\n\n"
        
        # 季节信息
        if proposal.season_id >= 0:
            result_str += f"**季数**：第{proposal.season_id + 1}季\n\n"
        else:
            result_str += "**整体规划**\n\n"
        
        # 情节规划
        if proposal.plot_planning:
            result_str += "#### 情节规划\n"
            for plot in proposal.plot_planning:
                result_str += f"**{plot.plot_id}. {plot.title}** (第{plot.chapter_from}-{plot.chapter_to}集)\n"
                result_str += f"{plot.description}\n"
                if plot.chapters:
                    for chapter in plot.chapters:
                        result_str += f"  - 第{chapter.chapter_num}集: {chapter.content}\n"
                result_str += "\n"
        
        # 卡点规划
        if proposal.point_planning and proposal.point_planning.points:
            result_str += "#### 卡点规划\n"
            for point in proposal.point_planning.points:
                result_str += f"- **{point.point_name}**：第{point.start_plot_id}-{point.end_plot_id}幕\n"
            if proposal.point_planning.description:
                result_str += f"卡点说明：{proposal.point_planning.description}\n"
            result_str += "\n"
        
        # 故事线
        if proposal.storylines:
            result_str += "#### 故事线\n"
            for storyline in proposal.storylines:
                result_str += f"**{storyline.name}**：{storyline.description}\n"
                if storyline.detailed_story_lines:
                    for detail in storyline.detailed_story_lines:
                        result_str += f"  - 第{detail.start_chapter_id}-{detail.end_chapter_id}集：{detail.title} - {detail.content}\n"
                result_str += "\n"
        
        # 其他改编建议
        if proposal.other_adaptation_proposal:
            result_str += f"#### 其他改编建议\n{proposal.other_adaptation_proposal}\n\n"
        
        result_str += "---\n\n"
    
    return result_str.rstrip("---\n\n") if result_str else "暂无整体剧本策划"

async def generate_story_outline_and_role(ctx: context.Context, request: pb.GenerateStoryOutlineByCreativityReq, cos=None):

    # 检查输入是否为空
    if request.generate_input.core_story == "":
        logger.error_context(ctx, f"基于创意生成故事大纲的输入‘创意和想法’输入为空。")

    # 判断剧集类型
    plot_type = request.story_info.plot_type
    is_short_drama = plot_type in [pb.PlotType.SHORT, pb.PlotType.SHORT_CARTOON]
    
    if is_short_drama:
        # SHORT 和 SHORT_CARTOON 类型：根据整体剧本策划，只生成故事大纲，不生成人设
        return await generate_outline_by_script(ctx, request, cos)
    else:
        # 其他类型：按现有链路生成故事大纲和人设
        return await generate_outline_and_role_default(ctx, request)


async def generate_outline_by_script(ctx: context.Context, request: pb.GenerateStoryOutlineByCreativityReq, cos):
    """
    基于整体剧本策划生成故事大纲（SHORT 和 SHORT_CARTOON 类型专用）
    只生成故事大纲，不生成人设
    参考整体剧本策划（ScriptProposal）来写故事大纲
    """
    # 获取整体剧本策划信息
    script_proposal_str = transfer_script_proposal_to_str(request.generate_data.script_proposal)
    logger.info_context(ctx, f"整体剧本策划内容: {script_proposal_str[:500]}...")
    
    # 设置 LLM 服务
    story_outline_config = get_model_config("story_outline_model")
    outline_llm_service = LLM(
        model=story_outline_config["model"],
        system_prompt=story_outline_config["system_prompt"],
        temperature=story_outline_config["temperature"],
        top_p=story_outline_config["top_p"],
        top_k=story_outline_config["top_k"],
        max_length=story_outline_config["max_length"],
        max_new_tokens=story_outline_config["max_new_tokens"],
        template=jinja2.Template(GENERATE_OUTLINE_BY_SCRIPT_PROMPT)
    )
    
    # 生成故事大纲
    logger.info_context(ctx, f"开始基于故事背景和整体剧本策划生成故事大纲。")
    # 获取剧集类型描述
    role_desc = get_role_desc_by_plot_type(request.story_info.plot_type)
    drama_type_name = get_drama_type_name_by_plot_type(request.story_info.plot_type)

    outline_params = {
        "Topic": request.generate_input.topic or "暂无",
        "WorldView": request.generate_input.world_view or "暂无",
        "StoryOutline": request.generate_input.core_story or "暂无",
        "Reference": request.generate_input.reference or "暂无",
        "ScriptProposal": script_proposal_str,
        "RoleDescription": role_desc,
        "DramaType": drama_type_name
    }
    logger.info_context(ctx, f"生成故事大纲的query为:\\n{outline_params}")

    start_outline_time = time.time()
    outline_ret_json = await llm_inference_json(
        params=outline_params,
        llm_service=outline_llm_service,
        task_name="基于策划生成故事大纲",
        ctx=ctx
    )

    # 将总纲中的粗粒度 beats 扩写为详细事件线，并通过 v9 树搜索/突变抑制同质化。
    # 后续分集大纲直接消费细化后的总纲，因此无需额外的手工续跑入口。
    outline_ret_json = await refine_story_outline_eventlines(ctx, outline_ret_json)

    final_outline = transfer_outline_to_str(outline_ret_json)
    logger.info_context(ctx, f"任务：基于策划生成故事大纲 成功生成结果:\\n{final_outline}\\n耗时：{time.time() - start_outline_time}s。")

    save_realtime(["03_story_outline", "outline.json"], outline_ret_json)
    # 短剧类型只返回故事大纲，人设为空
    story_assets = pb.StoryOutline(
        story_outline=[final_outline], role_info=[request.generate_input.role_setting],
        world_building=[request.generate_input.world_view],
    )
    save_realtime(["03_story_outline", "story_assets.json"], story_assets.to_dict())
    return pb.GenerateStoryOutlineRsp(story_outline=story_assets)


async def generate_outline_and_role_default(ctx: context.Context, request: pb.GenerateStoryOutlineByCreativityReq):
    """
    默认链路：生成故事大纲和人设（非短剧类型）
    """
    story_outline_config = get_model_config("story_outline_model")
    outline_llm_service = LLM(
        model=story_outline_config["model"],
        system_prompt=story_outline_config["system_prompt"],
        temperature=story_outline_config["temperature"],
        top_p=story_outline_config["top_p"],
        top_k=story_outline_config["top_k"],
        max_length=story_outline_config["max_length"],
        max_new_tokens=story_outline_config["max_new_tokens"],
        template=jinja2.Template(GENERATE_OUTLINE_PROMPT)
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
        template=jinja2.Template(GENERATE_ROLE_PROMPT)
    )

    # 获取剧集类型描述
    role_desc = get_role_desc_by_plot_type(request.story_info.plot_type)
    drama_type_name = get_drama_type_name_by_plot_type(request.story_info.plot_type)

    # 生成故事大纲
    logger.info_context(ctx, f"开始基于创意生成故事大纲。")
    outline_params = {
        "Topic": request.generate_input.topic or "暂无",
        "WorldView": request.generate_input.world_view or "暂无",
        "StoryOutline": request.generate_input.core_story or "暂无",
        "RoleInfo": request.generate_input.role_setting or "暂无",
        "Reference": request.generate_input.reference or "暂无",
        "RoleDescription": role_desc,
        "DramaType": drama_type_name
    }
    logger.info_context(ctx, f"生成故事大纲的query为:\\n{outline_params}")

    start_outline_time = time.time()
    outline_ret_json = await llm_inference_json(
        params=outline_params,
        llm_service=outline_llm_service,
        task_name="生成故事大纲",
        ctx=ctx
    )

    outline_ret_json = await refine_story_outline_eventlines(ctx, outline_ret_json)

    final_outline = transfer_outline_to_str(outline_ret_json)
    logger.info_context(ctx, f"任务：故事大纲 成功生成结果:\\n{final_outline}\\n耗时：{time.time() - start_outline_time}s。")

    # 生成角色
    logger.info_context(ctx, f"开始基于创意生成角色信息。")
    role_params = {
        "Topic": request.generate_input.topic or "暂无",
        "WorldView": request.generate_input.world_view or "暂无",
        "StoryOutline": request.generate_input.core_story or "暂无",
        "RoleInfo": request.generate_input.role_setting or "暂无",
        "Reference": request.generate_input.reference or "暂无",
        "CreatedOutline": final_outline,
        "RoleNums": "3至5",
        "RoleDescription": role_desc,
        "DramaType": drama_type_name
    }
    logger.info_context(ctx, f"生成角色的query为:\\n{role_params}")

    start_role_time = time.time()
    role_ret_json = await llm_inference_json(
        params=role_params,
        llm_service=role_llm_service,
        task_name="生成角色信息",
        ctx=ctx
    )

    final_role = transfer_role_to_str(role_ret_json)
    logger.info_context(ctx, f"任务：角色 成功生成结果:\\n{final_role}\\n耗时：{time.time() - start_role_time}s。")

    save_realtime(["03_story_outline", "outline.json"], outline_ret_json)
    save_realtime(["03_story_outline", "role.json"], role_ret_json)
    story_assets = pb.StoryOutline(story_outline=[final_outline], role_info=[final_role],
                                  world_building=[request.generate_input.world_view])
    save_realtime(["03_story_outline", "story_assets.json"], story_assets.to_dict())
    return pb.GenerateStoryOutlineRsp(story_outline=story_assets)
