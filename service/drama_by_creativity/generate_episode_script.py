import asyncio
from drama_local.diagnostics import record_skip
import time
import random
import json # 确保引入json
from drama_local.runtime import logger
from drama_local import runtime as context
from drama_local import models as pb

from .utils import *
from .utils import calc_word_nums_by_word_count, calc_scene_nums_by_per_scene_word, calc_dialogue_ratio, calc_dialogue_ratio_per_scene
from .scene_outline_inference import scene_outline_inference_json
from .process_coherence import polish_plot
from .script_postprocessing import word_count_adjustment_enabled
from .narrative_memory import NarrativeMemory
from .state_lifecycle_memory import StateLifecycleMemory, assets_from_request, preserve_generation_assets, enabled as state_lifecycle_enabled
from .scene_state_gate import SceneStateGate, gate_mode, bind_policy
from .summary_memory import SummaryMemory, summary_baseline_enabled, memory_prompt
from .plot_library_service import search_plot_from_library, format_plot_references
from .plot_retrieval_control import is_plot_retrieval_disabled, select_prompt_reference
# 假设你有一个用于生成整集剧本的Prompt，如果没有，可以使用场次Prompt修改，或者新建一个
from .prompts import GENERATE_SCENE_OUTLINE_PROMPT, GENERATE_SCENE_PLOT_PROMPT, GENERATE_WHOLE_EPISODE_PROMPT, SCRIPT_FORMAT_PROMPT, ADJUST_WORD_COUNT_PROMPT
from .realtime_output import save_realtime
from . import drama_type_demand

from drama_local.runtime import ScriptContentFile

import jinja2
from .local_llm import LLM


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
DRAM_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drama_configs.yaml")
RETRY_TIMES = 3
RETRY_INTERVAL = 60
MAX_ADJUST_ROUNDS = 1
MAX_ADJUST_MODEL_CALLS_PER_ROUND = 1


def parse_word_nums_range(word_nums: str):
    """
    解析字数范围字符串，返回 (min_words, max_words) 元组。
    支持格式："1000-1400" 或 "500"（单个数字时上下浮动20%）
    """
    word_nums = word_nums.strip()
    if "-" in word_nums:
        parts = word_nums.split("-")
        return int(parts[0]), int(parts[1])
    else:
        base = int(word_nums)
        return int(base * 0.8), int(base * 1.2)


async def check_and_adjust_word_count(
    ctx,
    episode_plot: str,
    word_nums: str,
    script_config: dict,
    narrative_memory_context: str = "暂无",
) -> str:
    """
    检查剧本字数和台词旁白比例是否达标，如果不达标则调用模型进行微调。
    策略：
    1. 传给模型的目标上限缩减100，使模型生成偏保守，减少超标概率。
    2. 每集最多调用一次调整模型；调整后即使仍不达标，也直接返回本次结果。
    3. 字数超出上限时缩减，字数低于下限时扩写。
    4. 同时检测台词与旁白比例，如果台词占比 < 65%，也触发调整。
    
    Args:
        ctx: 上下文
        episode_plot: 当前剧本内容
        word_nums: 目标字数范围字符串，如 "1000-1400"
        script_config: 模型配置
    
    Returns:
        str: 调整后的剧本内容（如果字数和比例均已达标则返回原内容）
    """
    if not word_count_adjustment_enabled():
        return episode_plot
    import re
    
    # 解析实际目标字数范围
    min_words, max_words = parse_word_nums_range(word_nums)
    
    # 传给模型的目标上限缩减100，使模型输出更保守
    WORD_COUNT_BUFFER = 100
    adjusted_max_words = max_words - WORD_COUNT_BUFFER
    # 确保缩减后的上限不低于下限
    if adjusted_max_words < min_words:
        adjusted_max_words = min_words
    
    current_plot = episode_plot
    
    for round_idx in range(MAX_ADJUST_ROUNDS + 1):  # +1 是因为第0轮是初始检查
        current_word_count = len(re.sub(r'\s+', '', current_plot))
        
        # 计算台词旁白比例（整集 + 每场）
        per_scene_info = calc_dialogue_ratio_per_scene(current_plot)
        ratio_info = per_scene_info["episode_ratio"]
        dialogue_ratio_pct = int(ratio_info["dialogue_ratio"] * 100)
        narration_ratio_pct = int(ratio_info["narration_ratio"] * 100)
        is_episode_ratio_ok = ratio_info["is_ratio_ok"]
        
        # 每场比例检查
        scenes_not_ok = per_scene_info["scenes_not_ok"]
        all_scenes_ok = per_scene_info["all_scenes_ok"]
        
        # 整集比例达标 且 所有场次比例都达标，才算真正达标
        is_ratio_ok = is_episode_ratio_ok and all_scenes_ok
        
        if round_idx == 0:
            logger.info_context(ctx, f"字数检验：当前字数={current_word_count}，目标范围={min_words}-{max_words}；台词占比={dialogue_ratio_pct}%，旁白占比={narration_ratio_pct}%，整集比例{'达标' if is_episode_ratio_ok else '不达标'}，各场比例{'全部达标' if all_scenes_ok else '存在不达标场次'}")
        else:
            logger.info_context(ctx, f"字数检验（第{round_idx}轮调整后）：当前字数={current_word_count}，目标范围={min_words}-{max_words}；台词占比={dialogue_ratio_pct}%，旁白占比={narration_ratio_pct}%，整集比例{'达标' if is_episode_ratio_ok else '不达标'}，各场比例{'全部达标' if all_scenes_ok else '存在不达标场次'}")
        
        # 输出每场比例详情日志
        if not all_scenes_ok:
            for scene in scenes_not_ok:
                scene_dialogue_pct = int(scene["dialogue_ratio"] * 100)
                scene_narration_pct = int(scene["narration_ratio"] * 100)
                logger.info_context(ctx, f"  不达标场次：{scene['scene_header']}，台词占比={scene_dialogue_pct}%，旁白占比={scene_narration_pct}%，台词{scene['dialogue_chars']}字，旁白{scene['narration_chars']}字")
        
        # 判断字数是否在范围内
        word_count_ok = min_words <= current_word_count <= max_words
        
        # 如果字数和比例都达标，直接通过
        if word_count_ok and is_ratio_ok:
            if round_idx == 0:
                logger.info_context(ctx, f"字数检验通过，台词比例达标（整集及各场均达标），无需调整。")
            else:
                logger.info_context(ctx, f"字数检验通过（经过{round_idx}轮调整）：最终字数={current_word_count}字，目标范围={min_words}-{max_words}；台词占比={dialogue_ratio_pct}%")
            return current_plot
        
        # 字数达标但比例不达标的情况
        if word_count_ok and not is_ratio_ok:
            if round_idx == 0:
                if not is_episode_ratio_ok:
                    logger.info_context(ctx, f"字数达标但整集台词比例不达标（当前台词占比={dialogue_ratio_pct}%，目标≥65%），需要调整比例。")
                elif not all_scenes_ok:
                    logger.info_context(ctx, f"字数达标且整集比例达标，但有{len(scenes_not_ok)}个场次台词比例不达标，需要调整。")
            # 比例不达标时继续进入调整流程
        
        # 字数或比例不达标，需要调整
        if round_idx >= MAX_ADJUST_ROUNDS:
            # 已达到最大调整轮次，返回当前结果
            if current_word_count > max_words:
                logger.warning_context(ctx, f"字数调整已达最大轮次({MAX_ADJUST_ROUNDS}次)仍超标：当前字数={current_word_count}字，目标上限={max_words}字，返回最后一次调整结果。")
            elif current_word_count < min_words:
                logger.warning_context(ctx, f"字数调整已达最大轮次({MAX_ADJUST_ROUNDS}次)仍不足：当前字数={current_word_count}字，目标下限={min_words}字，返回最后一次调整结果。")
            elif not is_ratio_ok:
                logger.warning_context(ctx, f"字数调整已达最大轮次({MAX_ADJUST_ROUNDS}次)台词比例仍不达标：整集台词占比={dialogue_ratio_pct}%，不达标场次数={len(scenes_not_ok)}，返回最后一次调整结果。")
            return current_plot
        
        if current_word_count > max_words:
            logger.info_context(ctx, f"字数超出上限（当前={current_word_count}字，上限={max_words}字），开始第{round_idx+1}轮字数微调（传给模型的目标范围={min_words}-{adjusted_max_words}）。\n调整前剧本内容：\n{current_plot}")
        elif current_word_count < min_words:
            logger.info_context(ctx, f"字数低于下限（当前={current_word_count}字，下限={min_words}字），开始第{round_idx+1}轮字数扩写（传给模型的目标范围={min_words}-{adjusted_max_words}）。\n调整前剧本内容：\n{current_plot}")
        else:
            logger.info_context(ctx, f"字数达标但台词比例不达标（整集台词占比={dialogue_ratio_pct}%，不达标场次数={len(scenes_not_ok)}），开始第{round_idx+1}轮比例调整。\n调整前剧本内容：\n{current_plot}")
        
        adjust_llm_service = LLM(
            model=script_config["model"],
            system_prompt=script_config["system_prompt"],
            temperature=script_config["temperature"],
            top_p=script_config["top_p"],
            top_k=script_config["top_k"],
            max_length=script_config["max_length"],
            max_new_tokens=script_config["max_new_tokens"],
            template=jinja2.Template(ADJUST_WORD_COUNT_PROMPT)
        )
        
        # 构建不达标场次信息，传给prompt模板
        scenes_not_ok_for_prompt = []
        for scene in scenes_not_ok:
            scenes_not_ok_for_prompt.append({
                "scene_header": scene["scene_header"],
                "dialogue_ratio_pct": int(scene["dialogue_ratio"] * 100),
                "narration_ratio_pct": int(scene["narration_ratio"] * 100),
                "dialogue_chars": scene["dialogue_chars"],
                "narration_chars": scene["narration_chars"],
            })
        
        adjust_params = {
            "Script": current_plot,
            "CurrentWordCount": str(current_word_count),
            "TargetWordNums": f"{min_words}-{adjusted_max_words}",
            "DialogueRatio": str(dialogue_ratio_pct),
            "NarrationRatio": str(narration_ratio_pct),
            "RatioNeedAdjust": not is_episode_ratio_ok,
            "ScenesNotOk": scenes_not_ok_for_prompt,
            "NarrativeMemory": narrative_memory_context,
        }
        
        adjust_success = False
        for retry_idx in range(MAX_ADJUST_MODEL_CALLS_PER_ROUND):
            try:
                response = await adjust_llm_service.request(ctx, adjust_params, f"{random.randint(0, 1000)}")
                adjusted_plot = response.response
                adjusted_word_count = len(re.sub(r'\s+', '', adjusted_plot))
                logger.info_context(ctx, f"第{round_idx+1}轮字数微调完成：调整前={current_word_count}字，调整后={adjusted_word_count}字，目标范围={min_words}-{max_words}\n调整后剧本内容：\n{adjusted_plot}")
                current_plot = adjusted_plot
                adjust_success = True
                break
            except Exception as e:
                logger.error_context(ctx, f"第{round_idx+1}轮字数微调调用失败（本集不再重试）\n{e}")
                continue
        
        if not adjust_success:
            logger.error_context(ctx, f"第{round_idx+1}轮字数微调失败；为保证每集最多调用一次调整模型，不再重试，返回当前剧本（字数={current_word_count}字，目标范围={min_words}-{max_words}）。")
            return current_plot
        
        # 继续下一轮检查（循环回到顶部会重新计算字数并判断）
    
    return current_plot

async def generate_episode_script(ctx: context.Context, request: pb.GenerateDramaByCreativityReq, cos, *, scene_gate_factory=None) -> pb.GenerateDramaRsp:
    scene_gate_mode = gate_mode()
    if scene_gate_factory is not None and (scene_gate_mode != 'off' or not state_lifecycle_enabled()):
        raise ValueError('Custom scene gate requires lifecycle memory and the legacy gate disabled')
    if scene_gate_mode != 'off' and not state_lifecycle_enabled():
        raise ValueError('Scene state gate requires lifecycle state and its hybrid retrieval view')
    bind_policy(os.environ['DRAMA_OUTPUT_DIR'], scene_gate_mode)
    # 读取drama_config.yaml
    all_dram_configs = read_yaml(DRAM_CONFIG_PATH)

    # 配置剧集类型
    if request.story_info.plot_type == pb.PlotType.SHORT:
        drama_config = all_dram_configs["drama_types"].get("short")
    elif request.story_info.plot_type == pb.PlotType.CARTOON:
        drama_config = all_dram_configs["drama_types"].get("cartoon")
    elif request.story_info.plot_type == pb.PlotType.LONG:
        drama_config = all_dram_configs["drama_types"].get("long")
    elif request.story_info.plot_type == pb.PlotType.SHORT_CARTOON:
        drama_config = all_dram_configs["drama_types"].get("short_cartoon")
    else:
        drama_config = all_dram_configs["drama_types"].get("default")

    # 获取输入信息
    episode_outline_list = []
    episode_outline = ""
    if len(request.generate_data.episode_outline.seasons) > 0 and len(request.generate_data.episode_outline.seasons[0].episodes) > 0:
        for episode_info in request.generate_data.episode_outline.seasons[0].episodes:
            episode_outline_list.append(episode_info.content)
            episode_outline += episode_info.content + "\n\n---\n\n"
    else:
        logger.error_context(ctx, f"集大纲列表为空。")
        raise Exception("集大纲列表为空，无法生成剧本")

    # 设置输入信息
    # 世界观优先使用generate_data中生成的世界观，而非用户最初输入的世界观
    if len(request.generate_data.story_outline.world_building) > 0:
        world_view = request.generate_data.story_outline.world_building[0]
    else:
        world_view = request.generate_input.world_view
    suggestion = request.generate_data.suggestion
    if len(request.generate_data.select_range) > 0:
        episode_ids = request.generate_data.select_range[0].episode_ids
    else:
        episode_ids = []
    if len(request.generate_data.story_outline.role_info) > 0:
        role_info = request.generate_data.story_outline.role_info[0]
    else:
        role_info = "暂无"

    # 判断是否为短剧类型
    is_short_drama = request.story_info.plot_type == pb.PlotType.SHORT or request.story_info.plot_type == pb.PlotType.SHORT_CARTOON or request.story_info.plot_type == pb.PlotType.CARTOON

    # Future Map 推动关联与 World State 保存在当前输出目录，可在中断后恢复。
    baseline = summary_baseline_enabled(request.generate_input)
    narrative_memory = SummaryMemory(request.story_info.story_id) if baseline else NarrativeMemory(request.story_info.story_id)
    state_lifecycle = StateLifecycleMemory(request.story_info.story_id) if state_lifecycle_enabled() else None
    if state_lifecycle is not None and baseline:
        raise ValueError("State lifecycle experiment requires the native narrative memory path")
    generated_assets = assets_from_request(request)
    preserve_generation_assets(generated_assets, os.environ["DRAMA_OUTPUT_DIR"])
    world_view = generated_assets["world_view"]
    role_info = generated_assets["role_info"]
    if state_lifecycle is not None:
        await state_lifecycle.initialize(ctx, generated_assets)
    logger.info_context(ctx, f"叙事记忆目录：{narrative_memory.story_dir}")

    # 设置模型
    scene_outline_config = get_model_config("scene_outline_model")
    scene_outline_llm_service = LLM(
        model=scene_outline_config["model"],
        system_prompt=scene_outline_config["system_prompt"],
        temperature=scene_outline_config["temperature"],
        top_p=scene_outline_config["top_p"],
        top_k=scene_outline_config["top_k"],
        max_length=scene_outline_config["max_length"],
        max_new_tokens=scene_outline_config["max_new_tokens"],
        template=jinja2.Template(memory_prompt(GENERATE_SCENE_OUTLINE_PROMPT, baseline))
    )

    script_config = get_model_config("script_model")
    scene_plot_llm_service = LLM(
        model=script_config["model"],
        system_prompt=script_config["system_prompt"],
        temperature=script_config["temperature"],
        top_p=script_config["top_p"],
        top_k=script_config["top_k"],
        max_length=script_config["max_length"],
        max_new_tokens=script_config["max_new_tokens"],
        template=jinja2.Template(memory_prompt(GENERATE_SCENE_PLOT_PROMPT, baseline))
    )

    whole_episode_llm_service = LLM(
        model=script_config["model"],
        system_prompt=script_config["system_prompt"],
        temperature=script_config["temperature"],
        top_p=script_config["top_p"],
        top_k=script_config["top_k"],
        max_length=script_config["max_length"],
        max_new_tokens=script_config["max_new_tokens"],
        template=jinja2.Template(memory_prompt(GENERATE_WHOLE_EPISODE_PROMPT, baseline)) 
    )

    # 检查是否重新生成剧本
    if suggestion != "":
        outline_suggestion_prompt = f"\n\n---\n\n## 生成建议\n{suggestion}\n(**注意**：请检查生成意见是否与本集/本场次剧情相关，如果相关，更根据生成建议，生成大纲)\n"
        plot_suggestion_prompt = f"\n\n---\n\n## 生成建议\n{suggestion}\n(**注意**：请检查生成意见是否与本集/本场次剧情相关，如果相关，更根据生成建议，生成剧情)\n"
        generating_episode_ids = sorted(set(episode_ids))
        
    else:
        outline_suggestion_prompt = ""
        plot_suggestion_prompt = ""
        generating_episode_ids = sorted(set(episode_ids))

    # 生成剧本内容
    final_rets = []
    for episode_id in generating_episode_ids:
        logger.info_context(ctx, f"开始生成第 {episode_id+1} 集的剧情。")
        start_episode_plot_time = time.time()
        if hasattr(ctx, "fields"):
            ctx.fields["episode_number"] = episode_id + 1

        try:
            single_episode_outline = episode_outline_list[episode_id]
        except IndexError:
            logger.error_context(ctx, f"集大纲列表索引错误，episode_id={episode_id}，列表长度={len(episode_outline_list)}。")
            raise Exception(f"集大纲列表索引错误，episode_id={episode_id} 超出范围，列表长度为{len(episode_outline_list)}")
        
        # 设置之前剧情 (用于连贯性)
        prev_plot = ""
        prev_episode_full_script = ""  # 上一集完整剧本（用于场次大纲生成）
        if episode_id == 0:
            prev_plot = ""
            prev_episode_full_script = ""
        else:
            try:
                script_content_file = await ScriptContentFile.download(ctx, cos, project_id=request.story_info.story_id, episode_id=episode_id-1)
                # 获取上一集的最后一部分内容，用于衔接
                prev_plot = script_content_file.result.split("\n\n---\n\n")[-1]
                # 获取上一集完整剧本，用于场次大纲生成时参考
                prev_episode_full_script = script_content_file.result
            except Exception as e:
                logger.error_context(ctx, f"获取cos中第{episode_id-1}集的剧情失败\n{e}")
                prev_plot = ""
                prev_episode_full_script = ""

        # 设置当前集剧情 (用于重生成时的参考)
        now_episode_script_content_prompt = ""
        if request.story_info.plot_type == pb.PlotType.SHORT and suggestion != "":
            try:
                now_episode_script_content_file = await ScriptContentFile.download(ctx, cos, project_id=request.story_info.story_id, episode_id=episode_id)
                now_episode_script_content_prompt = f"\n\n---\n\n## 之前创作过的本集剧本\n{now_episode_script_content_file.result}\n"
            except Exception as e:
                logger.error_context(ctx, f"获取cos中第{episode_id}集的剧情失败\n{e}")
                now_episode_script_content_prompt = ""

        # ------------------------------------------------------------------
        # 步骤0：用本集大纲从桥段库召回参考剧情
        # ------------------------------------------------------------------
        plot_reference = ""
        disable_plot_retrieval = is_plot_retrieval_disabled(request)
        if disable_plot_retrieval:
            logger.info_context(ctx, f"桥段库检索已由请求参数关闭，跳过第 {episode_id+1} 集的桥段召回。")
        elif single_episode_outline:
            logger.info_context(ctx, f"第 {episode_id+1} 集剧本生成前，用本集大纲从桥段库召回参考剧情...")
            start_plot_search_time = time.time()
            recalled_plots = await search_plot_from_library(ctx, query=single_episode_outline)
            plot_reference = format_plot_references(recalled_plots)
            plot_search_cost = time.time() - start_plot_search_time
            if recalled_plots:
                logger.info_context(ctx, f"成功召回 {len(recalled_plots)} 段相关剧情，耗时：{plot_search_cost:.2f}s，将作为第 {episode_id+1} 集场次大纲和剧本生成的参考")
            else:
                logger.info_context(ctx, f"未召回到相关剧情，耗时：{plot_search_cost:.2f}s")

        # 优先使用用户传入的参考套路，如果用户没有传入，则使用桥段库召回的结果
        plot_reference_for_prompt = select_prompt_reference(request.generate_input, plot_reference)

        # ------------------------------------------------------------------
        # 步骤1：按当前集直接 FM 及其子树分类检索既有 FD，再生成场次大纲
        # ------------------------------------------------------------------
        if baseline:
            memory_context = narrative_memory.retrieve(episode_id)
            recent_episode_summaries = ""
        else:
            memory_context, memory_retrieval = narrative_memory.retrieve(single_episode_outline, episode_id)
            logger.info_context(
                ctx,
                f"第 {episode_id+1} 集 Future Map 记忆检索完成："
                f"直接 FM {len(memory_retrieval.get('direct_node_ids', []))} 个，"
                f"直接相关 FD {sum(len(items) for items in memory_retrieval.get('direct_items', {}).values())} 条，"
                f"子树相关 FD {len(memory_retrieval.get('subtree_items', []))} 条。"
            )
            recent_episode_summaries = narrative_memory._format_recent_episode_summaries(episode_id)
        if state_lifecycle is not None:
            memory_context += state_lifecycle.context(episode_id, single_episode_outline)
        logger.info_context(ctx, f"开始生成第 {episode_id+1}/{len(episode_outline_list)} 集的场次大纲。")
        
        # === 判断动漫类型是否需要走逐场生成 ===
        cartoon_use_scene_by_scene = False
        if request.story_info.plot_type == pb.PlotType.CARTOON:
            cartoon_word_threshold = drama_config.get("cartoon_word_threshold", 3000)
            try:
                max_word_count_check = request.generate_input.common.max_word_count_per_episode
                if max_word_count_check and max_word_count_check > cartoon_word_threshold:
                    cartoon_use_scene_by_scene = True
                    logger.info_context(ctx, f"动漫类型：每集字数上限={max_word_count_check} > 阈值={cartoon_word_threshold}，将使用逐场生成策略。")
                else:
                    logger.info_context(ctx, f"动漫类型：每集字数上限={max_word_count_check} <= 阈值={cartoon_word_threshold}，使用一次性生成策略。")
            except Exception as e:
                logger.warning_context(ctx, f"动漫类型：获取字数上限失败，使用一次性生成策略。Error: {e}")

        # 根据每集字数上下限动态计算建议场次数范围（作为约束边界传给模型）
        try:
            min_word_count = request.generate_input.common.min_word_count_per_episode
            max_word_count = request.generate_input.common.max_word_count_per_episode
            scene_nums_min, scene_nums_max = calc_scene_nums_by_per_scene_word(drama_config, min_word_count, max_word_count)
            logger.info_context(ctx, f"根据每集字数范围={min_word_count}-{max_word_count}和每场字数配置，动态计算建议场次数范围：{scene_nums_min}-{scene_nums_max}")
        except Exception as e:
            scene_nums_min = drama_config["scene_nums_min"]
            scene_nums_max = drama_config["scene_nums_max"]
            logger.warning_context(ctx, f"获取字数范围失败，使用配置默认场次数：{scene_nums_min}-{scene_nums_max}，Error: {e}")
        
        # 计算字数相关参数（用于传入场次大纲提示词）
        try:
            word_nums_for_outline = calc_word_nums_by_word_count(drama_config, min_word_count, max_word_count)
        except Exception:
            word_nums_for_outline = drama_config.get("word_nums", "500")
        
        # 计算每场建议字数
        per_scene_config = drama_config.get("per_scene_word_range")
        if per_scene_config:
            per_scene_word_nums = f"{per_scene_config['min']}-{per_scene_config['max']}"
        elif scene_nums_max > 0 and min_word_count and max_word_count:
            # 没有per_scene_word_range配置时，根据整集字数和场次数估算
            est_per_scene_min = int(min_word_count / scene_nums_max)
            est_per_scene_max = int(max_word_count / max(1, scene_nums_min))
            per_scene_word_nums = f"{est_per_scene_min}-{est_per_scene_max}"
        else:
            per_scene_word_nums = "600-1200"
        
        # 限制传入场次大纲Prompt的集大纲范围：只传当前集及前后各2集的大纲，避免信息过载
        nearby_start = max(0, episode_id - 2)
        nearby_end = min(len(episode_outline_list), episode_id + 3)  # 不含右端点
        nearby_episode_outline = ""
        for idx in range(nearby_start, nearby_end):
            nearby_episode_outline += f"### 第{idx+1}集大纲\n{episode_outline_list[idx]}\n\n---\n\n"
        nearby_episode_outline = nearby_episode_outline.rstrip("\n-\n") if nearby_episode_outline else "暂无"

        # 根据生成策略决定传入场次大纲的上一集剧本内容：
        # - 一次性生成整集（短剧/短番/动漫字数少）：传入上一集完整剧本，提升整体连续性
        # - 逐场生成（电视剧/动漫字数多）：只传入上一集最后一场剧本，避免token过长
        if is_short_drama and not cartoon_use_scene_by_scene:
            # 一次性生成整集的类型，传入完整上一集剧本
            prev_episode_script_for_outline = prev_episode_full_script
        else:
            # 逐场生成的类型，只传入上一集最后一场剧本
            prev_episode_script_for_outline = prev_plot

        if prev_episode_script_for_outline:
            prev_episode_script_prompt = f"\n\n---\n\n## 上一集剧本内容（请参考以保持剧情连续性）\n{prev_episode_script_for_outline}\n"
        else:
            prev_episode_script_prompt = ""

        generate_scene_outline_params = {
            "WorldView": world_view or "暂无",
            "RoleInfo": role_info or "暂无",
            "EpisodeOutline": nearby_episode_outline or "暂无",
            "Outline": single_episode_outline or "暂无",
            "SceneNumsMin": str(scene_nums_min),
            "SceneNumsMax": str(scene_nums_max),
            "WordNums": word_nums_for_outline,
            "PerSceneWordNums": per_scene_word_nums,
            "EpisodeIdx": str(episode_id+1),
            "Suggestion": outline_suggestion_prompt,
            "CreatedNowEpisodeContent": now_episode_script_content_prompt,
            "RoleDescription": drama_config["role_desc"],
            "DramaType": drama_config["name"],
            "PlotReference": plot_reference_for_prompt,
            "PrevEpisodeScript": prev_episode_script_prompt,
            "NarrativeMemory": memory_context,
            "RecentEpisodeSummaries": recent_episode_summaries,
        }
        
        start_scene_outline_time = time.time()
        scene_gate = SceneStateGate.from_memory(state_lifecycle, episode_id, generate_scene_outline_params, scene_gate_mode) if scene_gate_mode != 'off' else None
        if scene_gate_factory is not None:
            scene_gate = scene_gate_factory(state_lifecycle, episode_id, generate_scene_outline_params)
        scene_outline_ret_json = scene_gate.saved_scenes() if scene_gate is not None else None
        if scene_outline_ret_json is None:
            scene_outline_ret_json = await scene_outline_inference_json(
                params=generate_scene_outline_params,
                llm_service=scene_outline_llm_service,
                task_name=f"生成第{episode_id+1}集场次大纲",
                ctx=ctx
            )
        logger.info_context(ctx, f"任务：第 {episode_id+1} 集场次大纲 成功生成结果:\n{scene_outline_ret_json}\n耗时：{time.time() - start_scene_outline_time}s。")
        if (
            not isinstance(scene_outline_ret_json, list)
            or not scene_outline_ret_json
            or any(not isinstance(scene, dict) or not scene for scene in scene_outline_ret_json)
        ):
            record_skip(episode_id + 1, "场次大纲经原始生成和纠错后仍未通过校验：" + str(getattr(scene_outline_llm_service, "last_error", "未知错误")), scene_outline_llm_service,
                        parsed_output=scene_outline_ret_json)
            logger.error_context(ctx, f"第 {episode_id+1} 集场次大纲经原始生成和纠错后仍未通过校验，跳过本集。具体原因：{getattr(scene_outline_llm_service, 'last_error', '未知错误')}")
            continue

        # ------------------------------------------------------------------
        # 步骤2：生成剧情 (根据类型分叉)
        # ------------------------------------------------------------------
        if scene_gate is not None:
            scene_outline_ret_json = await scene_gate.apply(ctx, scene_outline_ret_json)
        episode_plot = ""
        
        # 保留原 Prompt 参数名以兼容模板，但内容已替换为检索后的结构化叙事记忆。
        prev_abs = memory_context

        # === 分支 A: 短剧/短番/动漫(字数少) (一次性生成整集) ===
        if is_short_drama and not cartoon_use_scene_by_scene:
            logger.info_context(ctx, f"检测到短剧类型，开始一次性生成第 {episode_id+1} 集完整剧本。")
            
            # 设置上一场剧情提示 (仅用于开头衔接)
            if prev_plot != "":
                prev_plot_prompt = f"\n\n---\n\n## 上一集结尾剧情\n{prev_plot}\n"
            else:
                prev_plot_prompt = ""

            # 根据每集字数上下限计算本集字数范围
            try:
                min_word_count = request.generate_input.common.min_word_count_per_episode
                max_word_count = request.generate_input.common.max_word_count_per_episode
                word_nums = calc_word_nums_by_word_count(drama_config, min_word_count, max_word_count)
                logger.info_context(ctx, f"根据每集字数上下限，计算字数范围：{word_nums}")
            except Exception as e:
                word_nums = drama_config.get("word_nums", "300")
                logger.warning_context(ctx, f"计算字数范围失败，使用默认值：{word_nums}，Error: {e}")

            # 构造整集生成的参数
            script_format = SCRIPT_FORMAT_PROMPT.replace("{{EpisodeIndex}}", str(episode_id+1))
            generate_whole_episode_params = {
                "RoleDescription": drama_config["role_desc"],
                "DramaType": drama_config["name"],
                "StyleConfig": getattr(drama_type_demand, drama_config["style_config"]),
                "EpisodeIndex": str(episode_id+1),
                "WorldView": world_view or "暂无",
                "RoleInfo": role_info or "暂无",
                "EpisodeOutline": single_episode_outline, # 本集大纲
                "AllSceneOutlines": json.dumps(scene_outline_ret_json, ensure_ascii=False), # 所有场次大纲
                "PrevPlot": prev_plot_prompt,
                "PrevAbs": prev_abs,
                "Suggestion": plot_suggestion_prompt,
                "CreatedNowEpisodeContent": now_episode_script_content_prompt,
                "ScriptFormat": script_format,
                "WordNums": word_nums,
                "PlotReference": plot_reference_for_prompt,
                "RecentEpisodeSummaries": recent_episode_summaries
            }
            
            start_whole_plot_time = time.time()
            whole_plot_status = False
            
            for i in range(RETRY_TIMES):
                try:
                    # 调用一次性生成的 LLM
                    response = await whole_episode_llm_service.request(ctx, generate_whole_episode_params, f"{random.randint(0, 1000)}")
                    episode_plot = response.response
                    if not isinstance(episode_plot, str) or not episode_plot.strip():
                        raise ValueError("模型返回空剧本")
                    whole_plot_status = True
                    logger.info_context(ctx, f"第 {episode_id+1} 集完整剧本生成成功，耗时：{time.time() - start_whole_plot_time}s。")
                    break
                except Exception as e:
                    logger.error_context(ctx, f"生成第 {episode_id+1} 集完整剧本失败，重试第{i+1}次\n{e}")
                    await asyncio.sleep(RETRY_INTERVAL)
                    continue
            
            if not whole_plot_status:
                record_skip(episode_id + 1, f"整集正文生成失败，已尝试{RETRY_TIMES}次", whole_episode_llm_service)
                logger.error_context(ctx, f"生成第 {episode_id+1} 集完整剧本失败，已重试{RETRY_TIMES}次。")
                continue

            # 加强与上一集的连贯性 (Polish Plot)
            if prev_plot != "":
                _, polished_now_plot = await polish_plot(ctx, prev_plot, episode_plot)
                episode_plot = polished_now_plot

            # 字数检验与调整
            episode_plot = await check_and_adjust_word_count(ctx, episode_plot, word_nums, script_config, memory_context)

            # 保存剧本到本地（保留---分隔符，便于后续集获取上一集结尾内容）
            script_content_file = ScriptContentFile(result=episode_plot)
            await script_content_file.upload(ctx, cos, project_id=request.story_info.story_id, episode_id=episode_id)
            logger.info_context(ctx, f"剧本成功保存至当前输出目录 local_store/{request.story_info.story_id}/script_content/episode_{episode_id}.json")
            save_realtime(["05_drama", f"episode_{episode_id+1:02d}.json"], {"episode_id": episode_id, "content": episode_plot})


        # === 分支 C: 动漫(字数多) (逐场生成) ===
        elif cartoon_use_scene_by_scene:
            logger.info_context(ctx, f"动漫逐场生成模式：开始逐场生成第 {episode_id+1} 集剧本。")
            
            # 计算每场的字数范围
            min_word_count = request.generate_input.common.min_word_count_per_episode
            max_word_count = request.generate_input.common.max_word_count_per_episode
            word_nums_episode = calc_word_nums_by_word_count(drama_config, min_word_count, max_word_count)
            
            per_scene_config = drama_config.get("per_scene_word_range", {"min": 800, "max": 1200})
            scene_count = len(scene_outline_ret_json)
            # 每场字数 = 整集字数 / 场次数，但不超过per_scene_word_range的上限
            per_scene_word_min = max(per_scene_config["min"], int(min_word_count / scene_count))
            per_scene_word_max = min(per_scene_config["max"] * 2, int(max_word_count / scene_count))
            # 确保min不超过max
            if per_scene_word_min > per_scene_word_max:
                per_scene_word_min = per_scene_word_max
            word_nums_per_scene = f"{per_scene_word_min}-{per_scene_word_max}"
            logger.info_context(ctx, f"动漫逐场生成：整集字数范围={word_nums_episode}，场次数={scene_count}，每场字数范围={word_nums_per_scene}")

            new_episode = True
            scene_generation_failed = False
            now_plot = ""
            for scene_id, single_scene_outline in enumerate(scene_outline_ret_json):
                logger.info_context(ctx, f"开始生成第 {episode_id+1} 集第 {scene_id+1}/{scene_count} 场次的剧情。")

                # 设置上一场的剧情
                if prev_plot != "":
                    prev_plot_prompt = f"\n\n---\n\n## 上一场剧情\n{prev_plot}\n"
                else:
                    prev_plot_prompt = ""

                # 设置剧情query
                script_format = SCRIPT_FORMAT_PROMPT.replace("{{EpisodeIndex}}", str(episode_id+1))
                
                generate_scene_plot_params = {
                    "EpisodeIndex": str(episode_id+1),
                    "SceneIdx": str(scene_id+1),
                    "WorldView": world_view or "暂无",
                    "RoleInfo": role_info or "暂无",
                    "Outline": str(scene_outline_ret_json),
                    "SceneOutline": str(single_scene_outline),
                    "PrevPlot": prev_plot_prompt,
                    "PrevAbs": prev_abs,
                    "Suggestion": plot_suggestion_prompt,
                    "CreatedNowEpisodeContent": now_episode_script_content_prompt,
                    "WordNums": word_nums_per_scene,
                    "ScriptFormat": script_format,
                    "RoleDescription": drama_config["role_desc"],
                    "DramaType": drama_config["name"],
                    "PlotReference": plot_reference_for_prompt,
                    "RecentEpisodeSummaries": recent_episode_summaries
                }
                logger.info_context(ctx, f"生成第 {episode_id+1} 集第 {scene_id+1} 场次剧情的query为:\n{generate_scene_plot_params}")

                start_scene_plot_time = time.time()

                scene_plot_status = False
                for i in range(RETRY_TIMES):
                    try:
                        scene_plot = (await scene_plot_llm_service.request(ctx, generate_scene_plot_params, f"{random.randint(0, 1000)}")).response
                        if not isinstance(scene_plot, str) or not scene_plot.strip():
                            raise ValueError("模型返回空场次剧本")
                        scene_plot_status = True
                        logger.info_context(ctx, f"生成第 {episode_id+1} 集第 {scene_id+1} 场次的剧情为:\n{scene_plot}\n耗时：{time.time() - start_scene_plot_time}s。")
                        break
                    except Exception as e:
                        logger.error_context(ctx, f"生成第 {episode_id+1} 集第 {scene_id+1} 场次的剧情失败，重试第{i+1}次\n{e}")
                        await asyncio.sleep(RETRY_INTERVAL)
                        continue

                if not scene_plot_status:
                    logger.error_context(ctx, f"生成第 {episode_id+1} 集第 {scene_id+1} 场次的剧情失败，已重试{RETRY_TIMES}次；放弃本集，避免形成残缺记忆。")
                    scene_generation_failed = True
                    break

                # 加强连贯性
                if prev_plot != "":
                    prev_plot, now_plot = await polish_plot(ctx, prev_plot, scene_plot)

                    if not new_episode:
                        episode_plot += prev_plot
                        episode_plot += "\n\n\n---\n\n\n"

                    prev_plot = now_plot
                    new_episode = False

                else:
                    prev_plot = scene_plot
                    now_plot = scene_plot
                    new_episode = False

            if scene_generation_failed:
                record_skip(episode_id + 1, f"第{scene_id+1}场正文生成失败，已尝试{RETRY_TIMES}次",
                            scene_plot_llm_service, partial_script=episode_plot)
                continue

            # 循环结束后，处理最后一场
            episode_plot += now_plot

            # 字数检验与调整（用整集字数范围做兜底检验）
            episode_plot = await check_and_adjust_word_count(ctx, episode_plot, word_nums_episode, script_config, memory_context)

            # 保存剧本到本地
            script_content_file = ScriptContentFile(result=episode_plot)
            await script_content_file.upload(ctx, cos, project_id=request.story_info.story_id, episode_id=episode_id)
            logger.info_context(ctx, f"剧本成功保存至当前输出目录 local_store/{request.story_info.story_id}/script_content/episode_{episode_id}.json")

        # === 分支 B: 普通剧本/长剧 (逐场生成) ===
        else:
            new_episode = True
            scene_generation_failed = False
            now_plot = ""
            word_nums_b = drama_config.get("word_nums", "500")
            for scene_id, single_scene_outline in enumerate(scene_outline_ret_json):
                logger.info_context(ctx, f"开始生成第 {episode_id+1} 集第 {scene_id+1} 场次的剧情。")

                # 设置后一场的剧情
                next_plot = ""
                if scene_id == len(scene_outline_ret_json) - 1 and suggestion != "":
                    try:
                        next_script_content_file = await ScriptContentFile.download(ctx, cos, project_id=request.story_info.story_id, episode_id=episode_id+1)
                        next_plot = next_script_content_file.result.split("\n\n---\n\n")[0]
                    except Exception as e:
                        logger.error_context(ctx, f"获取cos中第{episode_id+1}集的剧情失败\n{e}")

                # 设置上一场的剧情
                if prev_plot != "":
                    prev_plot_prompt = f"\n\n---\n\n## 上一场剧情\n{prev_plot}\n\n--\n\n## 后一场次剧情{next_plot}\n"
                else:
                    prev_plot_prompt = ""

                # 设置剧情query
                script_format = SCRIPT_FORMAT_PROMPT.replace("{{EpisodeIndex}}", str(episode_id+1))
                
                # 根据每集字数上下限计算字数范围
                try:
                    min_word_count = request.generate_input.common.min_word_count_per_episode
                    max_word_count = request.generate_input.common.max_word_count_per_episode
                    word_nums_b = calc_word_nums_by_word_count(drama_config, min_word_count, max_word_count)
                except Exception:
                    word_nums_b = drama_config.get("word_nums", "500")
                
                generate_scene_plot_params = {
                    "EpisodeIndex": str(episode_id+1),
                    "SceneIdx": str(scene_id+1),
                    "WorldView": world_view or "暂无",
                    "RoleInfo": role_info or "暂无",
                    "Outline": str(scene_outline_ret_json),
                    "SceneOutline": str(single_scene_outline),
                    "PrevPlot": prev_plot_prompt,
                    "PrevAbs": prev_abs,
                    "Suggestion": plot_suggestion_prompt,
                    "CreatedNowEpisodeContent": now_episode_script_content_prompt,
                    "WordNums": word_nums_b,
                    "ScriptFormat": script_format,
                    "RoleDescription": drama_config["role_desc"],
                    "DramaType": drama_config["name"],
                    "PlotReference": plot_reference_for_prompt,
                    "RecentEpisodeSummaries": recent_episode_summaries
                }
                logger.info_context(ctx, f"生成第 {episode_id+1} 集第 {scene_id+1} 场次剧情的query为:\n{generate_scene_plot_params}")

                start_scene_plot_time = time.time()

                scene_plot_status = False
                for i in range(RETRY_TIMES):
                    try:
                        scene_plot = (await scene_plot_llm_service.request(ctx, generate_scene_plot_params, f"{random.randint(0, 1000)}")).response
                        if not isinstance(scene_plot, str) or not scene_plot.strip():
                            raise ValueError("模型返回空场次剧本")
                        scene_plot_status = True
                        logger.info_context(ctx, f"生成第 {episode_id+1} 集第 {scene_id+1} 场次的剧情为:\n{scene_plot}\n耗时：{time.time() - start_scene_plot_time}s。")
                        break
                    except Exception as e:
                        logger.error_context(ctx, f"生成第 {episode_id+1} 集第 {scene_id+1} 场次的剧情失败，重试第{i+1}次\n{e}")
                        await asyncio.sleep(RETRY_INTERVAL)
                        continue

                if not scene_plot_status:
                    logger.error_context(ctx, f"生成第 {episode_id+1} 集第 {scene_id+1} 场次的剧情失败，已重试{RETRY_TIMES}次；放弃本集，避免形成残缺记忆。")
                    scene_generation_failed = True
                    break

                # 加强连贯性
                if prev_plot != "":
                    prev_plot, now_plot = await polish_plot(ctx, prev_plot, scene_plot)

                    if not new_episode:
                        episode_plot += prev_plot
                        episode_plot += "\n\n\n---\n\n\n"

                    prev_plot = now_plot
                    new_episode = False

                else:
                    prev_plot = scene_plot
                    now_plot = scene_plot
                    new_episode = False

            if scene_generation_failed:
                record_skip(episode_id + 1, f"第{scene_id+1}场正文生成失败，已尝试{RETRY_TIMES}次",
                            scene_plot_llm_service, partial_script=episode_plot)
                continue

            # 循环结束后，处理最后一场
            episode_plot += now_plot

            # 字数检验与调整
            episode_plot = await check_and_adjust_word_count(ctx, episode_plot, word_nums_b, script_config, memory_context)

            # 保存剧本到本地（保留---分隔符，便于后续集获取上一集结尾内容）
            script_content_file = ScriptContentFile(result=episode_plot)
            await script_content_file.upload(ctx, cos, project_id=request.story_info.story_id, episode_id=episode_id)
            logger.info_context(ctx, f"剧本成功保存至当前输出目录 local_store/{request.story_info.story_id}/script_content/episode_{episode_id}.json")
            save_realtime(["05_drama", f"episode_{episode_id+1:02d}.json"], {"episode_id": episode_id, "content": episode_plot})

        # ------------------------------------------------------------------
        # 用经过润色和字数调整的最终剧本并行更新 Future Map 推动关联与 World State
        # ------------------------------------------------------------------
        if baseline:
            await narrative_memory.update_from_episode(ctx, episode_id, episode_plot)
        else:
            try:
                memory_update = await narrative_memory.update_from_episode(
                    ctx=ctx,
                    episode_id=episode_id,
                    episode_outline=single_episode_outline,
                    final_script=episode_plot,
                )
                logger.info_context(
                    ctx,
                    f"第 {episode_id+1} 集叙事记忆更新完成："
                    f"新增 Future Driving 节点 {len(memory_update['future_driving_nodes'])} 条，"
                    f"新增 FM 关联 {len(memory_update['future_map_edges'])} 条，"
                    f"状态变更 {len(memory_update['world_state_updates'])} 条。"
                )
            except Exception as e:
                logger.error_context(ctx, f"第 {episode_id+1} 集叙事记忆更新失败，剧本仍正常保存：{e}")

        # ------------------------------------------------------------------
        # 结束本集处理
        # ------------------------------------------------------------------
        # 返回给前端时去掉---分隔符，替换为双换行
        if state_lifecycle is not None:
            await state_lifecycle.update_from_episode(ctx, episode_id, single_episode_outline, episode_plot)
        episode_plot_for_return = episode_plot.replace("\n\n---\n\n", "\n\n")
        final_rets.append(
            pb.Drama.Episode(
                season_id=0,
                episode_id=episode_id,
                script_content=episode_plot_for_return
            )
        )
        logger.info_context(ctx, f"第 {episode_id+1} 集剧情生成完成。耗时：{time.time() - start_episode_plot_time}s。")

    return pb.GenerateDramaRsp(result = pb.Drama(seasons=[pb.Drama.Season(episodes=final_rets, season_id=0)]))
