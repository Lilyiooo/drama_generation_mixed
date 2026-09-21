import json
import time
from drama_local.runtime import logger
from drama_local import runtime as context
from drama_local import models as pb

from .utils import *
from .llm_inference import llm_inference_json
from .prompts import REGENERATE_ROLE_PROMPT
from .realtime_output import save_realtime

import jinja2
from .local_llm import LLM


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

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

async def regenerate_role(ctx: context.Context, request: pb.RegenerateStoryOutlineByCreativityReq):
    # 检查输入是否为空
    if len(request.regenerate_data.generate_data.story_outline.role_info) > 0 and len(request.regenerate_data.generate_data.story_outline.story_outline) > 0:
        raw_role_info = request.regenerate_data.generate_data.story_outline.role_info[0]
        raw_outline = request.regenerate_data.generate_data.story_outline.story_outline[0]
    else:
        logger.error_context(ctx, f"基于创意重新生成角色的输入”原始角色信息“或“原始故事大纲”输入为空。")

    # 设置模型
    role_config = get_model_config("role_model")
    llm_service = LLM(
        model=role_config["model"],
        system_prompt=role_config["system_prompt"],
        temperature=role_config["temperature"],
        top_p=role_config["top_p"],
        top_k=role_config["top_k"],
        max_length=role_config["max_length"],
        max_new_tokens=role_config["max_new_tokens"],
        template=jinja2.Template(REGENERATE_ROLE_PROMPT)
    )

    # 获取剧集类型描述
    role_desc = get_role_desc_by_plot_type(request.story_info.plot_type)
    drama_type_name = get_drama_type_name_by_plot_type(request.story_info.plot_type)

    # 重新生成故事大纲
    regenerate_role_params = {
        "Topic": request.generate_input.topic or "暂无",
        "WorldView": request.generate_input.world_view or "暂无",
        "StoryOutline": request.generate_input.core_story or "暂无",
        "RoleInfo": request.generate_input.role_setting or "暂无",
        "Reference": request.generate_input.reference or "暂无",
        "CreatedOutline": raw_outline,
        "RoleNums": "3至5",
        "CreatedRole": raw_role_info,
        "Suggestion": request.regenerate_data.suggestion or "暂无",
        "RoleDescription": role_desc,
        "DramaType": drama_type_name
    }
    logger.info_context(ctx, f"重新生成角色的query为:\n{regenerate_role_params}")

    start_role_time = time.time()
    role_ret_json = await llm_inference_json(
        params=regenerate_role_params,
        llm_service=llm_service,
        task_name="重新生成角色",
        ctx=ctx
    )

    final_role = transfer_role_to_str(role_ret_json)
    logger.info_context(ctx, f"任务：重新生成角色 成功生成结果:\n{final_role}\n耗时：{time.time() - start_role_time}s。")
    save_realtime(["regenerate_role", "role.json"], role_ret_json)

    return pb.RegenerateStoryOutlineRsp(result=final_role)
