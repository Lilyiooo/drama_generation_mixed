import os
import time
import random
from drama_local.runtime import logger
from drama_local import runtime as context
from drama_local import models as pb

from .utils import *
from .llm_inference import llm_inference_json
from .prompts import GENERATE_ALL_EPISODE_OUTLINE_PROMPT, GENERATE_EPISODE_OUTLINE_BY_SCRIPT_PROMPT
from .plot_library_service import search_plot_from_library, format_plot_references, PLOT_SEARCH_SIZE_OUTLINE
from .plot_retrieval_control import is_plot_retrieval_disabled, select_prompt_reference
from .realtime_output import save_realtime
from .future_map import generate_future_map_artifacts
from .summary_memory import summary_baseline_enabled

import jinja2
from .local_llm import LLM


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

# 集大纲校验重试最大次数
MAX_VALIDATION_RETRIES = 2


def validate_episode_outline_chunk(outline_chunk_json, start_episode_id, end_episode_id, ctx=None):
    """
    校验单个 chunk 生成的集大纲是否合法。
    返回 (is_valid, error_msg)
    """
    expected_count = end_episode_id - start_episode_id + 1
    actual_count = len(outline_chunk_json)

    # 检查数量是否匹配
    if actual_count != expected_count:
        return False, f"集大纲数量不匹配：期望 {expected_count} 集（第{start_episode_id}~{end_episode_id}集），实际生成 {actual_count} 集"

    # 检查每集的 episode_id 是否正确且内容非空
    generated_ids = []
    for idx, episode in enumerate(outline_chunk_json):
        ep_id = episode.get("episode_id", -1)
        generated_ids.append(ep_id)

        # 检查核心字段是否为空
        core_plot = episode.get("core_plot", "")
        title = episode.get("title", "")
        if not core_plot or not title or core_plot.strip() == "" or title.strip() == "":
            return False, f"第{ep_id}集大纲内容为空（标题或核心情节缺失）"

    # 检查 episode_id 是否连续且完整
    expected_ids = list(range(start_episode_id, end_episode_id + 1))
    if sorted(generated_ids) != expected_ids:
        missing_ids = set(expected_ids) - set(generated_ids)
        extra_ids = set(generated_ids) - set(expected_ids)
        error_parts = []
        if missing_ids:
            error_parts.append(f"缺失集: {sorted(missing_ids)}")
        if extra_ids:
            error_parts.append(f"多余集: {sorted(extra_ids)}")
        return False, f"集大纲 episode_id 不连续或不完整：{'; '.join(error_parts)}，期望 {expected_ids}，实际 {sorted(generated_ids)}"

    return True, ""


def validate_final_episode_outline(final_outline_json, episode_nums, ctx=None):
    """
    校验最终合并后的所有集大纲是否合法。
    返回 (is_valid, error_msg)
    """
    actual_count = len(final_outline_json)

    # 检查总集数
    if actual_count != episode_nums:
        return False, f"最终集大纲总数不匹配：期望 {episode_nums} 集，实际 {actual_count} 集"

    # 检查所有 episode_id 是否从1开始连续
    expected_ids = list(range(1, episode_nums + 1))
    actual_ids = [ep.get("episode_id", -1) for ep in final_outline_json]
    if sorted(actual_ids) != expected_ids:
        missing_ids = set(expected_ids) - set(actual_ids)
        duplicate_ids = [eid for eid in actual_ids if actual_ids.count(eid) > 1]
        error_parts = []
        if missing_ids:
            error_parts.append(f"缺失集: {sorted(missing_ids)}")
        if duplicate_ids:
            error_parts.append(f"重复集: {sorted(set(duplicate_ids))}")
        return False, f"最终集大纲 episode_id 校验失败：{'; '.join(error_parts)}"

    # 检查每集内容是否为空
    empty_episodes = []
    for episode in final_outline_json:
        ep_id = episode.get("episode_id", -1)
        core_plot = episode.get("core_plot", "")
        title = episode.get("title", "")
        if not core_plot or not title or core_plot.strip() == "" or title.strip() == "":
            empty_episodes.append(ep_id)

    if empty_episodes:
        return False, f"以下集的大纲内容为空（标题或核心情节缺失）: {empty_episodes}"

    return True, ""


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
        episode_hook = episode.get("ending_hook", "")    

        outline_str += f"### 第{episode_id}集：{episode_name}\n"
        outline_str += f"涉及角色：{all_roles}\n\n"
        outline_str += f"#### 核心情节\n{episode_core_plot}\n\n"
        outline_str += f"#### 本集亮点\n{episode_plot}\n\n"
        outline_str += f"#### 角色成长\n{episode_role_growth}\n\n"
        outline_str += f"#### 角色关系变化\n{episode_relationship}\n\n"
        outline_str += f"#### 主线剧情推进程度\n{episode_storyline}\n\n"
        if episode_hook:
            outline_str += f"#### 本集钩子\n{episode_hook}"

        all_outline_str += outline_str
        all_outline_str += "\n\n---\n\n"
        all_outline_list.append(outline_str)

    return all_outline_str[:-7], all_outline_list

async def generate_episode_outline(ctx: context.Context, request: pb.GenerateEpisodeOutlineByCreativityReq, cos=None):
    return await generate_episode_outline_default(ctx, request)

async def generate_episode_outline_by_script(ctx: context.Context, request: pb.GenerateEpisodeOutlineByCreativityReq, cos):
    return await _generate_with_stage_plan(ctx, request)


async def generate_episode_outline_default(ctx: context.Context, request: pb.GenerateEpisodeOutlineByCreativityReq):
    return await _generate_with_stage_plan(ctx, request)


async def _generate_with_stage_plan(ctx, request):
    from .eight_field_episode_outline import generate_eight_fields

    data = request.generated_data.story_outline
    if not data.story_outline or not data.story_outline[0]:
        raise ValueError("分阶段生成集大纲需要非空的详细故事大纲")
    story_outline = data.story_outline[0]
    episode_nums = request.generate_input.common.episode_nums
    plot_reference = ""
    if not is_plot_retrieval_disabled(request) and request.generate_input.core_story:
        plots = await search_plot_from_library(
            ctx, query=request.generate_input.core_story, size=PLOT_SEARCH_SIZE_OUTLINE)
        plot_reference = format_plot_references(plots)
    common = {
        "Topic": request.generate_input.topic or "暂无",
        "WorldView": request.generate_input.world_view or "暂无",
        "StoryOutline": request.generate_input.core_story or "暂无",
        "Reference": select_prompt_reference(request.generate_input, plot_reference),
        "RoleInfo": data.role_info[0] if data.role_info else request.generate_input.role_setting or "暂无",
        "RoleDescription": get_role_desc_by_plot_type(request.story_info.plot_type),
        "DramaType": get_drama_type_name_by_plot_type(request.story_info.plot_type),
    }
    # 所有集数均在阶段规划对应的范围内生成。
    outlines = await generate_eight_fields(ctx, story_outline, episode_nums, common)
    save_realtime(["episode_outlines.json"], outlines)
    if os.environ.get("DRAMA_OUTLINE_ONLY") != "1" and not summary_baseline_enabled(request.generate_input):
        await generate_future_map_artifacts(ctx, outlines)
    final_outline_str, final_outline_list = transfer_outline_to_str(outlines)
    logger.info_context(ctx, f"分阶段集大纲生成完成，共{len(outlines)}集。\n{final_outline_str}")
    episodes = [pb.EpisodeOutline.Episode(
        season_id=0, episode_id=item["episode_id"] - 1, chapter_from=0, chapter_to=0, content=content
    ) for item, content in zip(outlines, final_outline_list)]
    return pb.GenerateEpisodeOutlineRsp(episode_outline=pb.EpisodeOutline(seasons=[
        pb.EpisodeOutline.Season(season_id=0, episodes=episodes)
    ]))
