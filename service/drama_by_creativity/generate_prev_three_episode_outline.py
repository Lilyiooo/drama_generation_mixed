"""
前3集大纲生成模块

该模块用于从剧本整体规划（ScriptProposal）生成前3集的集大纲。
无论用户输入要求生成多少集，该服务只负责生成前3集。

输入：GenerateEpisodeOutlineByCreativityReq
输出：GenerateEpisodeOutlineRsp（只包含前3集）
"""

import time
import random
from drama_local.runtime import logger
from drama_local import runtime as context
from drama_local import models as pb

from .utils import *
from .llm_inference import llm_inference_json
from .plot_library_service import search_plot_from_library, format_plot_references, PLOT_SEARCH_SIZE_OUTLINE
from .plot_retrieval_control import is_plot_retrieval_disabled, select_prompt_reference
from .realtime_output import save_realtime

import jinja2
from .local_llm import LLM

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

# 固定只生成前3集
PREV_EPISODE_COUNT = 3

# Prompt：从剧本整体规划生成前3集大纲
GENERATE_PREV_THREE_EPISODE_OUTLINE_PROMPT = """
{{RoleDescription}}

现在请根据给定的剧本整体规划，设计第 1 集到第 {{EndEpisodeId}} 集的具体剧情。

## 题材
{{Topic}}

---

## 世界观
{{WorldView}}

---

## 故事梗概
{{StoryOutline}}

---

## 主要人物设定
{{RoleInfo}}

---

{% if Reference %}## 创作参考套路
{{Reference}}

---
{% endif %}

## 剧本整体规划（卡点规划）
{{ScriptProposal}}

---

## 具体要求
1. 请根据剧本整体规划中的卡点信息，遵循三幕式结构，即发展、高潮、结尾，设计第 1 集到第 {{EndEpisodeId}} 集的具体剧情。确保前{{EndEpisodeId}}集的剧情能够快速吸引观众，建立故事基调和人物关系。
2. 结构：主线明确，简洁紧凑，聚焦核心。关注主角之间的互动，剧情必须高度浓缩，主线清晰，弱化支线或复杂背景，直奔核心冲突。
3. 节奏：快节奏，高密度反转。强调**"秒级抓人"**，前5-10秒需用强冲突、悬念或视觉冲击吸引观众；情节推进迅猛，频繁设置反转或悬念钩子。角色间的关系要保持连贯，不可突变。
4. 角色：少而鲜明，标签化。性格标签鲜明（如"腹黑总裁""复仇女主"），快速建立辨识度；人物弧光简化，较少复杂成长。
5. 主题：情绪优先，共鸣直接。侧重强情绪冲击（如复仇、逆袭、悬疑），主题直白（如"底层逆袭""手撕渣男"），追求即时情感共鸣。
6. 格式：按照剧本标准格式书写，包含场头、对白、场景和动作描述等，对白需要能够表达剧情，不可简略。直接生成剧本，不回答其他信息。
7. 逻辑：逻辑清晰，剧情发展通顺，可以适当添加闪回加强情节连贯性，避免矛盾。确保剧情逻辑连贯，避免角色行为或剧情发展出现逻辑矛盾。
8. 剧情的前因后果铺垫要恰当，不要过度铺垫或省略重要信息。
9. 每集大纲需要与剧本整体规划中对应的卡点保持一致，确保卡点描述的剧情内容在对应集中得到体现。
10. **重要**：前3集是整部剧的开篇，需要快速建立世界观、人物关系和核心冲突，吸引观众继续观看。

---

## 输出格式
请以JSON格式输出最终的结果，具体格式如下：
```json
[
    {
        "episode_id": xxx(整数型的编号，从1开始),
        "title": "xxx",
        "core_plot": "xxx(描述本集的核心的剧情内容，200字左右)",
        "roles": ["xxx(从给定的所有人物中选择本集中涉及的人物名字1)", "xxx(从给定的所有人物中选择本集中涉及的人物名字2)", ...],
        "highlights": "xxx(本集的看点和名场面，请给出1-3个看点和具体名场面)",
        "character_growth": "xxx(角色成长变化度)",
        "relationship_changes": "xxx(人物关系的变化)",
        "main_storyline_progression": "xxx(主线剧情的推进度)",
        "ending_hook": "xxx(本集的结尾的钩子)"
    },
    ...
]
```

(**注意**：检查输出的JSON格式，务必确保输出正确的JSON格式。集大纲内容请以中文输出，禁止参杂英文。)
"""


def transfer_outline_to_str(outline_ret_json):
    """
    将集大纲 JSON 转换为字符串格式
    """
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


def transfer_script_proposal_to_str(script_proposal: pb.SingleScriptProposal) -> str:
    """
    将剧本整体规划（ScriptProposal）转换为字符串格式，用于 Prompt 输入
    """
    proposal_str = ""
    
    # 处理卡点规划（PlotPlanning）
    if script_proposal.plot_planning:
        proposal_str += "### 卡点规划\n"
        for plot in script_proposal.plot_planning:
            proposal_str += f"#### 卡点 {plot.plot_id}: {plot.title}\n"
            proposal_str += f"集范围：第 {plot.chapter_from} 集 - 第 {plot.chapter_to} 集\n"
            if plot.description:
                proposal_str += f"描述：{plot.description}\n"
            if plot.chapters:
                proposal_str += "各集详情：\n"
                for chapter in plot.chapters:
                    proposal_str += f"- 第 {chapter.chapter_num} 集：{chapter.content}\n"
            proposal_str += "\n"
    
    # 处理分点规划（PointPlanning）
    if script_proposal.point_planning and script_proposal.point_planning.points:
        proposal_str += "### 分季/分点规划\n"
        for point in script_proposal.point_planning.points:
            proposal_str += f"- {point.point_name}：从卡点 {point.start_plot_id} 到卡点 {point.end_plot_id}\n"
        if script_proposal.point_planning.description:
            proposal_str += f"\n说明：{script_proposal.point_planning.description}\n"
        proposal_str += "\n"
    
    # 处理故事线（Storylines）
    if script_proposal.storylines:
        proposal_str += "### 故事线\n"
        for storyline in script_proposal.storylines:
            proposal_str += f"#### {storyline.name}\n"
            if storyline.description:
                proposal_str += f"{storyline.description}\n"
            if storyline.detailed_story_lines:
                for detail in storyline.detailed_story_lines:
                    proposal_str += f"- 第 {detail.start_chapter_id} 集至第 {detail.end_chapter_id} 集：{detail.title} - {detail.content}\n"
            proposal_str += "\n"
    
    return proposal_str if proposal_str else "暂无"


async def generate_prev_three_episode_outline(
    ctx: context.Context,
    request: pb.GenerateEpisodeOutlineByCreativityReq
) -> pb.GenerateEpisodeOutlineRsp:
    """
    从剧本整体规划生成前3集的集大纲
    
    无论用户输入要求生成多少集，该服务只负责生成前3集的大纲。
    
    Args:
        ctx: 上下文
        request: GenerateEpisodeOutlineByCreativityReq 请求对象，包含：
            - story_info: 项目信息
            - generate_input: 创意输入信息（题材、世界观、故事梗概等）
            - generated_data: 已生成的数据（script_proposal、story_outline等）
    
    Returns:
        GenerateEpisodeOutlineRsp: 集大纲响应（只包含前3集）
    """
    
    # 从请求中提取数据
    generate_input = request.generate_input
    generated_data = request.generated_data
    story_info = request.story_info
    
    # 获取角色信息
    role_info = ""
    if generated_data.story_outline and len(generated_data.story_outline.role_info) > 0:
        role_info = generated_data.story_outline.role_info[0]
    
    # 获取世界观
    world_view = ""
    if generated_data.story_outline and len(generated_data.story_outline.world_building) > 0:
        world_view = generated_data.story_outline.world_building[0]
    else:
        world_view = generate_input.world_view
    
    # 获取剧本整体规划（优先使用第一个）
    script_proposal = None
    if generated_data.script_proposal and len(generated_data.script_proposal.script_proposals) > 0:
        script_proposal = generated_data.script_proposal.script_proposals[0]
    
    plot_type = story_info.plot_type
    
    # 获取剧集类型描述
    role_desc = get_role_desc_by_plot_type(plot_type)
    drama_type_name = get_drama_type_name_by_plot_type(plot_type)
    
    logger.info_context(ctx, f"创作类型：{plot_type}，开始生成前{PREV_EPISODE_COUNT}集的集大纲。")
    
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
        template=jinja2.Template(GENERATE_PREV_THREE_EPISODE_OUTLINE_PROMPT)
    )

    # 将剧本整体规划转换为字符串
    script_proposal_str = transfer_script_proposal_to_str(script_proposal) if script_proposal else "暂无"
    logger.info_context(ctx, f"剧本整体规划字符串：\n{script_proposal_str}")

    # 从情节桥段库召回参考剧情（每次请求独立查询，无状态）
    core_story_text = generate_input.core_story or ""
    plot_reference = ""
    disable_plot_retrieval = is_plot_retrieval_disabled(generate_input)
    if disable_plot_retrieval:
        logger.info_context(ctx, f"桥段库检索已由请求参数关闭，跳过前{PREV_EPISODE_COUNT}集大纲的桥段召回。")
    elif core_story_text:
        logger.info_context(ctx, f"前{PREV_EPISODE_COUNT}集大纲生成前，用核心故事从桥段库召回参考剧情...")
        recalled_plots = await search_plot_from_library(ctx, query=core_story_text, size=PLOT_SEARCH_SIZE_OUTLINE)
        plot_reference = format_plot_references(recalled_plots)
        if recalled_plots:
            logger.info_context(ctx, f"成功召回 {len(recalled_plots)} 段相关剧情，将作为前{PREV_EPISODE_COUNT}集大纲生成的参考")
        else:
            logger.info_context(ctx, f"未召回到相关剧情，将使用原始参考信息")

    # 优先使用用户传入的参考套路，如果用户没有传入，则使用桥段库召回的结果
    enhanced_reference = select_prompt_reference(generate_input, plot_reference)

    # 设置提示词参数
    generate_episode_outline_params = {
        "Topic": generate_input.topic or "暂无",
        "WorldView": world_view or "暂无",
        "StoryOutline": generate_input.core_story or "暂无",
        "Reference": enhanced_reference,
        "RoleInfo": role_info or "暂无",
        "ScriptProposal": script_proposal_str,
        "EndEpisodeId": str(PREV_EPISODE_COUNT),
        "RoleDescription": role_desc,
        "DramaType": drama_type_name,
    }
    logger.info_context(ctx, f"生成前{PREV_EPISODE_COUNT}集大纲的query为：\n{generate_episode_outline_params}")

    start_outline_time = time.time()
    
    # 调用模型生成前3集大纲
    outline_ret_json = await llm_inference_json(
        params=generate_episode_outline_params,
        llm_service=llm_service,
        task_name=f"生成前{PREV_EPISODE_COUNT}集大纲",
        ctx=ctx
    )

    final_outline_str, final_outline_list = transfer_outline_to_str(outline_ret_json)
    logger.info_context(ctx, f"任务：生成前{PREV_EPISODE_COUNT}集大纲 成功生成结果：\n{final_outline_str}\n耗时：{time.time() - start_outline_time}s。")

    # 构造返回结果
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

    save_realtime(["02_demo_drama", "prev_three_outline.json"], outline_ret_json)
    return final_outline
