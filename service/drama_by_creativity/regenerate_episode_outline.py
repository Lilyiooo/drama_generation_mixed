import time
from drama_local import runtime as context
from drama_local import models as pb
from drama_local.runtime import logger

from .utils import *
from .llm_inference import llm_inference_json
from .process_coherence import polish_plot
from .update_plot_abs import update_plot_abs
from .prompts import REGENERATE_SINGLE_EPISODE_OUTLINE_PROMPT
from .realtime_output import save_realtime

import jinja2
from .local_llm import LLM


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

def transfer_outline_to_str(single_outline_ret_json):
    episode = single_outline_ret_json

    outline_str = ""
    episode_id = episode.get("episode_id", -1)
    episode_name = episode.get("title", "未正确解析本集标题")
    episode_core_plot = episode.get("core_plot", "未正确解析本集核心情节")
    episode_roles = episode.get("roles", [])
    all_roles = "、".join(episode_roles)
    episode_plot = episode.get("highlights", "未正确解析本集亮点")
    episode_role_growth = episode.get("character_growth", "未正确解析本集角色成长")
    episode_relationship = episode.get("relationship_changes", "未正确解析本集角色关系")
    episode_storyline = episode.get("main_storyline_progression", "未正确解析本集故事线")    
    episode_hook = episode.get("ending_hook", "未正确解析本集钩子")    

    outline_str += f"### 第{episode_id}集：{episode_name}\n"
    outline_str += f"涉及角色：{all_roles}\n\n"
    outline_str += f"#### 核心情节\n{episode_core_plot}\n\n"
    outline_str += f"#### 本集亮点\n{episode_plot}\n\n"
    outline_str += f"#### 角色成长\n{episode_role_growth}\n\n"
    outline_str += f"#### 角色关系变化\n{episode_relationship}\n\n"
    outline_str += f"#### 主线剧情推进程度\n{episode_storyline}\n\n"
    outline_str += f"#### 本集钩子\n{episode_hook}"

    return outline_str

def _extract_context_outlines(generated_data, episode_id, context_range=2):
    """从generated_data中提取当前集前后各context_range集的大纲作为上下文参考。
    
    Args:
        generated_data: GeneratedData，包含已生成的episode_outline
        episode_id: 当前待重新生成的集id（0-based）
        context_range: 前后各取多少集，默认2
    
    Returns:
        str: 格式化的前后集大纲文本
    """
    if not generated_data or not generated_data.episode_outline or not generated_data.episode_outline.seasons:
        return "暂无"
    
    # 收集所有集的大纲内容
    all_episodes = []
    for season in generated_data.episode_outline.seasons:
        for episode in season.episodes:
            all_episodes.append(episode)
    
    if not all_episodes:
        return "暂无"
    
    # 找到当前集在列表中的索引位置
    current_idx = None
    for idx, ep in enumerate(all_episodes):
        if ep.episode_id == episode_id:
            current_idx = idx
            break
    
    # 如果没找到精确匹配，尝试用episode_id作为索引
    if current_idx is None:
        current_idx = episode_id if episode_id < len(all_episodes) else None
    
    if current_idx is None:
        return "暂无"
    
    # 提取前后各context_range集
    start_idx = max(0, current_idx - context_range)
    end_idx = min(len(all_episodes), current_idx + context_range + 1)
    
    context_parts = []
    for idx in range(start_idx, end_idx):
        if idx == current_idx:
            continue  # 跳过当前集本身
        ep = all_episodes[idx]
        position = "前" if idx < current_idx else "后"
        distance = abs(idx - current_idx)
        ep_content = ep.content if ep.content else "暂无内容"
        context_parts.append(f"### {position}{distance}集（第{ep.episode_id + 1}集）\n{ep_content}")
    
    if not context_parts:
        return "暂无"
    
    return "\n\n".join(context_parts)


async def regenerate_episode_outline_single(ctx: context.Context, request: pb.RegenerateEpisodeOutlineSingleByCreativityReq) -> pb.RegenerateEpisodeOutlineSingleRsp:
    # 检查输入是否为空
    if request.episode_outline == "" or request.suggestion == "":
        logger.error_context(ctx, f"基于创意重新生成单集大纲的输入“原始集大纲”或“修改意见”输入为空。")
        return pb.RegenerateEpisodeOutlineSingleRsp()
    
    # 获取剧集类型描述
    role_desc = get_role_desc_by_plot_type(request.story_info.plot_type)
    drama_type_name = get_drama_type_name_by_plot_type(request.story_info.plot_type)

    # 从generated_data中提取前后集大纲作为上下文
    context_outlines = _extract_context_outlines(request.generated_data, request.episode_id)
    logger.info_context(ctx, f"提取前后集大纲上下文:\n{context_outlines[:200]}...")

    # 设置模型
    episode_outline_config = get_model_config("episode_outline_model")
    llm_service = LLM(
        model=episode_outline_config["model"],
        system_prompt=episode_outline_config["system_prompt"],
        temperature=episode_outline_config["temperature"],
        top_p=episode_outline_config["top_p"],
        top_k=episode_outline_config["top_k"],
        max_length=episode_outline_config["max_length"],
        max_new_tokens=episode_outline_config["max_new_tokens"],
        template=jinja2.Template(REGENERATE_SINGLE_EPISODE_OUTLINE_PROMPT)
    )

    # 重新生成故事大纲
    regenerate_episode_outline_params = {
        "Topic": request.generate_input.topic or "暂无",
        "WorldView": request.generate_input.world_view or "暂无",
        "StoryOutline": request.generate_input.core_story or "暂无",
        "Reference": request.generate_input.reference or "暂无",
        "GeneratedOutline": request.episode_outline or "暂无",
        "ContextOutlines": context_outlines,
        "Suggestion": request.suggestion or "暂无",
        "EpisodeIndex": str(request.episode_id+1),
        "RoleDescription": role_desc,
        "DramaType": drama_type_name
    }
    logger.info_context(ctx, f"重新生成单集大纲的query为:\n{regenerate_episode_outline_params}")

    start_outline_time = time.time()
    outline_ret_json = await llm_inference_json(
        params=regenerate_episode_outline_params,
        llm_service=llm_service,
        task_name="重新生成单集大纲",
        ctx=ctx
    )

    final_outline_str = transfer_outline_to_str(outline_ret_json)
    logger.info_context(ctx, f"任务：重新生成单集大纲 成功生成结果:\n{final_outline_str}\n耗时：{time.time() - start_outline_time}s。")
    save_realtime(["regenerate_single_episode_outline", "outline.json"], outline_ret_json)

    return pb.RegenerateEpisodeOutlineSingleRsp(episode_outline=final_outline_str)
