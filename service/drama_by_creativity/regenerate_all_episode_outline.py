import time
import random
from drama_local import runtime as context
from drama_local import models as pb
from drama_local.runtime import logger

from .utils import *
from .llm_inference import llm_inference_json
from .process_coherence import polish_plot
from .update_plot_abs import update_plot_abs
from .prompts import REGENERATE_ALL_EPISODE_OUTLINE_PROMPT
from .realtime_output import save_realtime

import jinja2
from .local_llm import LLM

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

def transfer_outline_to_str(outline_ret_json):
    all_outline_list = []
    all_outline_str = ""

    for episode in outline_ret_json:
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

        all_outline_str += outline_str
        all_outline_str += "\n\n---\n\n"
        all_outline_list.append(outline_str)

    return all_outline_str[:-7], all_outline_list

async def regenerate_all_episode_outline(ctx: context.Context, request: pb.RegenerateEpisodeOutlineAllByCreativityReq) -> pb.GenerateEpisodeOutlineRsp:
    # 检查输入是否为空
    if request.regenerate_data.suggestion == "":
        logger.error_context(ctx, f"基于创意重新生成所有集大纲的输入”修改建议“输入为空。")

    # 获取原始集大纲
    raw_episode_outline = ""
    all_episode_outline = request.generated_data.episode_outline.seasons[0]
    for episode_info in all_episode_outline.episodes:
        raw_episode_outline += episode_info.content + "\n\n"
    
    # 设置重新生成集数
    # episode_nums = len(all_episode_outline.episodes)

    if request.story_info.plot_type == pb.PlotType.CARTOON:
        try:
            episode_nums = request.generate_input.common.episode_nums
            logger.info_context(ctx, f"基于创意生成集大纲的输入“集数”为：{episode_nums}")
        except:
            episode_nums = random.randint(14, 17)
            logger.info_context(ctx, f"获取用户自定义集数失败，使用随机集数：{episode_nums}")
    elif request.story_info.plot_type == pb.PlotType.SHORT:
        try:
            episode_nums = request.generate_input.common.episode_nums
            logger.info_context(ctx, f"基于创意生成集大纲的输入“集数”为：{episode_nums}")
        except:
            episode_nums = random.randint(38, 70)
            logger.info_context(ctx, f"获取用户自定义集数失败，使用随机集数：{episode_nums}")
    else:
        episode_nums = 15 # 默认值
        logger.info_context(ctx, f"未获取到正确的剧本类型，使用默认集数：{episode_nums}")

    # 获取剧集类型描述
    role_desc = get_role_desc_by_plot_type(request.story_info.plot_type)
    drama_type_name = get_drama_type_name_by_plot_type(request.story_info.plot_type)

    # 拆分集数
    if request.story_info.plot_type == pb.PlotType.SHORT:
        if episode_nums > 90:
            chunk_nums = 3
            logger.info_context(ctx, f"创作类型：{request.story_info.plot_type}，基于创意生成集大纲的输入“集数”为：{episode_nums}，拆分为{chunk_nums}份。")
        else:
            chunk_nums = 2
            logger.info_context(ctx, f"创作类型：{request.story_info.plot_type}，基于创意生成集大纲的输入“集数”为：{episode_nums}，拆分为{chunk_nums}份。")
    else:
        chunk_nums = 1
        logger.info_context(ctx, f"创作类型：{request.story_info.plot_type}，基于创意生成集大纲的输入“集数”为：{episode_nums}，拆分为{chunk_nums}份。")
        

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
        template=jinja2.Template(REGENERATE_ALL_EPISODE_OUTLINE_PROMPT)
    )

    # 生成集大纲
    base_chunk_size = episode_nums // chunk_nums
    extra_episodes = episode_nums % chunk_nums

    final_outline_json = []
    start_outline_time = time.time()
    for i in range(chunk_nums):
        start_episode_id = i * base_chunk_size + 1

        if i == chunk_nums - 1:
            chunk_size = base_chunk_size + extra_episodes
        else:
            chunk_size = base_chunk_size
        end_episode_id = start_episode_id + chunk_size - 1

        # 重新生成故事大纲
        regenerate_episode_all_outline_params = {
            "Topic": request.generate_input.topic or "暂无",
            "WorldView": request.generate_input.world_view or "暂无",
            "StoryOutline": request.generate_input.core_story or "暂无",
            "Reference": request.generate_input.reference or "暂无",
            "RoleInfo": request.generated_data.story_outline.story_outline[0] or "暂无",
            "Outline": request.generated_data.story_outline.role_info[0] or "暂无",
            "StartEpisodeId": str(start_episode_id),
            "EndEpisodeId": str(end_episode_id),
            "ReDesignedOutline": str(final_outline_json),
            "GeneratedOutline": raw_episode_outline,
            "Suggestion": request.regenerate_data.suggestion or "暂无",
            "RoleDescription": role_desc,
            "DramaType": drama_type_name
        }
        logger.info_context(ctx, f"重新生成所有集大纲的query为:\n{regenerate_episode_all_outline_params}")

        # 调用模型
        outline_ret_json = await llm_inference_json(
            params=regenerate_episode_all_outline_params,
            llm_service=llm_service,
            task_name="重新生成所有集大纲",
            ctx=ctx
        )
        final_outline_json.extend(outline_ret_json)
        save_realtime(["regenerate_all_episode_outline", f"chunk_{len(final_outline_json)}.json"], outline_ret_json)

    final_outline_str, final_outline_list = transfer_outline_to_str(final_outline_json)
    logger.info_context(ctx, f"任务：重新生成所有集大纲 成功生成结果:\n{final_outline_str}\n耗时：{time.time() - start_outline_time}s。")

    # 构建返回结果
    episode_rets = []
    for i, episode_str in enumerate(final_outline_list):
        episode_rets.append(pb.EpisodeOutline.Episode(
            season_id=0,
            episode_id=i,
            chapter_from=0,
            chapter_to=0,
            content=episode_str
            )
        )

    final_outline = pb.GenerateEpisodeOutlineRsp(
        episode_outline=pb.EpisodeOutline(
            seasons=[
                pb.EpisodeOutline.Season(
                    season_id=0,
                    episodes=episode_rets
                )
            ]
        )
    )

    return final_outline
