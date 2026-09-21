import json
import time
from drama_local.runtime import logger
from drama_local import runtime as context
from drama_local import models as pb

from .utils import *
from .llm_inference import llm_inference_json
from .prompts import REGENERATE_OUTLINE_PROMPT, REGENERATE_OUTLINE_BY_SCRIPT_PROMPT
from .generate_story_outline_and_role import transfer_script_proposal_to_str
from .realtime_output import save_realtime

import jinja2
from .local_llm import LLM


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

async def regenerate_story_outline(ctx: context.Context, request: pb.RegenerateStoryOutlineByCreativityReq):
    # 检查输入是否为空
    if len(request.regenerate_data.generate_data.story_outline.story_outline) > 0:
        raw_outline = request.regenerate_data.generate_data.story_outline.story_outline[0]
    else:
        logger.error_context(ctx, f"基于创意重新生成故事大纲的输入‘原始大纲’输入为空。")
        raw_outline = ""

    # 判断剧集类型
    plot_type = request.story_info.plot_type
    is_short_drama = plot_type in [pb.PlotType.SHORT, pb.PlotType.SHORT_CARTOON, pb.PlotType.CARTOON]
    
    if is_short_drama:
        # SHORT、SHORT_CARTOON 和 CARTOON 类型：参考整体剧本策划重新生成故事大纲
        return await regenerate_outline_with_script_proposal(ctx, request, raw_outline)
    else:
        # 其他类型：按现有链路重新生成故事大纲
        return await regenerate_outline_default(ctx, request, raw_outline)


async def regenerate_outline_with_script_proposal(ctx: context.Context, request: pb.RegenerateStoryOutlineByCreativityReq, raw_outline: str):
    """
    基于整体剧本策划重新生成故事大纲（SHORT 和 SHORT_CARTOON 类型专用）
    """
    logger.info_context(ctx, f"剧集类型为短剧，参考整体剧本策划重新生成故事大纲。")
    
    # 获取整体剧本策划信息
    script_proposal_str = transfer_script_proposal_to_str(request.regenerate_data.generate_data.script_proposal)
    logger.info_context(ctx, f"整体剧本策划内容: {script_proposal_str[:500]}...")
    
    # 设置模型
    story_outline_config = get_model_config("story_outline_model")
    llm_service = LLM(
        model=story_outline_config["model"],
        system_prompt=story_outline_config["system_prompt"],
        temperature=story_outline_config["temperature"],
        top_p=story_outline_config["top_p"],
        top_k=story_outline_config["top_k"],
        max_length=story_outline_config["max_length"],
        max_new_tokens=story_outline_config["max_new_tokens"],
        template=jinja2.Template(REGENERATE_OUTLINE_BY_SCRIPT_PROMPT)
    )

    # 获取剧集类型描述
    role_desc = get_role_desc_by_plot_type(request.story_info.plot_type)
    drama_type_name = get_drama_type_name_by_plot_type(request.story_info.plot_type)

    # 重新生成故事大纲
    regenerate_outline_params = {
        "Topic": request.generate_input.topic or "暂无",
        "WorldView": request.generate_input.world_view or "暂无",
        "StoryOutline": request.generate_input.core_story or "暂无",
        "Reference": request.generate_input.reference or "暂无",
        "ScriptProposal": script_proposal_str,
        "Outline": raw_outline,
        "Suggestion": request.regenerate_data.suggestion or "暂无",
        "RoleDescription": role_desc,
        "DramaType": drama_type_name
    }
    logger.info_context(ctx, f"重新生成故事大纲的query为:\n{regenerate_outline_params}")

    start_outline_time = time.time()
    outline_ret_json = await llm_inference_json(
        params=regenerate_outline_params,
        llm_service=llm_service,
        task_name="重新生成故事大纲（短剧类型）",
        ctx=ctx
    )

    final_outline = transfer_outline_to_str(outline_ret_json)
    logger.info_context(ctx, f"任务：重新生成故事大纲（短剧类型） 成功生成结果:\n{final_outline}\n耗时：{time.time() - start_outline_time}s。")
    save_realtime(["regenerate_story_outline", "outline_short.json"], outline_ret_json)

    return pb.RegenerateStoryOutlineRsp(result=final_outline)


async def regenerate_outline_default(ctx: context.Context, request: pb.RegenerateStoryOutlineByCreativityReq, raw_outline: str):
    """
    默认链路：重新生成故事大纲（非短剧类型）
    """
    logger.info_context(ctx, f"剧集类型为非短剧，按默认链路重新生成故事大纲。")
    
    # 设置模型
    story_outline_config = get_model_config("story_outline_model")
    llm_service = LLM(
        model=story_outline_config["model"],
        system_prompt=story_outline_config["system_prompt"],
        temperature=story_outline_config["temperature"],
        top_p=story_outline_config["top_p"],
        top_k=story_outline_config["top_k"],
        max_length=story_outline_config["max_length"],
        max_new_tokens=story_outline_config["max_new_tokens"],
        template=jinja2.Template(REGENERATE_OUTLINE_PROMPT)
    )

    # 获取剧集类型描述
    role_desc = get_role_desc_by_plot_type(request.story_info.plot_type)
    drama_type_name = get_drama_type_name_by_plot_type(request.story_info.plot_type)

    # 重新生成故事大纲
    regenerate_outline_params = {
        "Topic": request.generate_input.topic or "暂无",
        "WorldView": request.generate_input.world_view or "暂无",
        "StoryOutline": request.generate_input.core_story or "暂无",
        "RoleInfo": request.generate_input.role_setting or "暂无",
        "Reference": request.generate_input.reference or "暂无",
        "Outline": raw_outline,
        "Suggestion": request.regenerate_data.suggestion or "暂无",
        "RoleDescription": role_desc,
        "DramaType": drama_type_name
    }
    logger.info_context(ctx, f"重新生成故事大纲的query为:\n{regenerate_outline_params}")

    start_outline_time = time.time()
    outline_ret_json = await llm_inference_json(
        params=regenerate_outline_params,
        llm_service=llm_service,
        task_name="重新生成故事大纲",
        ctx=ctx
    )

    final_outline = transfer_outline_to_str(outline_ret_json)
    logger.info_context(ctx, f"任务：重新生成故事大纲 成功生成结果:\n{final_outline}\n耗时：{time.time() - start_outline_time}s。")
    save_realtime(["regenerate_story_outline", "outline.json"], outline_ret_json)

    return pb.RegenerateStoryOutlineRsp(result=final_outline)
