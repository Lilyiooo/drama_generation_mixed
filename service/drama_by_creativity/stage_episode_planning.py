"""Plan stage ranges, then generate each stage without leaking other stages' events."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import jinja2
from drama_local.runtime import logger
from .local_llm import LLM
from .llm_inference import llm_inference_json
from .realtime_output import save_realtime
from .utils import get_model_config
from .prompts.stage_episode_plan import GENERATE_STAGE_EPISODE_PLAN_PROMPT


def extract_stages(outline):
    """Consume the request itself (JSON or canonical story-outline Markdown), not a stale disk file."""
    if isinstance(outline, str):
        try:
            payload = json.loads(outline)
        except (ValueError, TypeError):
            payload = None
        if isinstance(payload, dict):
            outline = payload
        else:
            sections = re.findall(r'^### ([^\n]+)\n(.*?)(?=^#{1,3} |\Z)', outline,
                                  flags=re.MULTILINE | re.DOTALL)
            outline = {'framework': [
                {'stage': name.strip(), 'event_list': [
                    {'event': event.strip()} for event in re.findall(
                        r'^- (.*?)(?=^- |\Z)', body, flags=re.MULTILINE | re.DOTALL)]}
                for name, body in sections]}
    if not isinstance(outline, dict) or not isinstance(outline.get('framework'), list):
        raise ValueError('详细故事大纲缺少可识别的 framework / 阶段事件列表')
    stages = []
    for item in outline['framework']:
        if not isinstance(item, dict) or not isinstance(item.get('stage'), str) or not item['stage'].strip():
            raise ValueError('故事大纲包含无效阶段名')
        events = item.get('event_list')
        if not isinstance(events, list) or not events:
            raise ValueError(f"阶段 {item['stage']} 的 event_list 为空或不是数组")
        cleaned = []
        for event in events:
            text = event.get('event') if isinstance(event, dict) else event
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"阶段 {item['stage']} 含空事件")
            cleaned.append({'event': text.strip()})
        stages.append({'stage': item['stage'].strip(), 'event_list': cleaned})
    names = [s['stage'] for s in stages]
    if not names or len(names) != len(set(names)):
        raise ValueError('故事大纲阶段不能为空或重复')
    return stages


def validate_stage_plan(plan, stages, total):
    if not isinstance(plan, list) or len(plan) != len(stages):
        raise ValueError(f'阶段规划必须为包含 {len(stages)} 个阶段的数组')
    next_id = 1
    for item, stage in zip(plan, stages):
        if not isinstance(item, dict) or item.get('stage') != stage['stage']:
            raise ValueError(f"阶段名称和顺序必须与大纲一致，当前应为 {stage['stage']}")
        start, end = item.get('start_episode_id'), item.get('end_episode_id')
        if type(start) is not int or type(end) is not int or start != next_id or end < start or end > total:
            raise ValueError(f"{stage['stage']} 范围无效：{start}—{end}，应从 {next_id} 开始，且不超过 {total}")
        if not isinstance(item.get('pacing_reason'), str) or not item['pacing_reason'].strip():
            raise ValueError(f"{stage['stage']} 缺少节奏分配理由")
        next_id = end + 1
    if next_id != total + 1:
        raise ValueError(f'阶段范围未覆盖全剧：应结束于第 {total} 集，实际为 {next_id - 1}')


def validate_episode_batch(episodes, start, end):
    if not isinstance(episodes, list) or len(episodes) != end - start + 1:
        raise ValueError(f'需完整生成第 {start}—{end} 集，共 {end-start+1} 条大纲')
    fields = ('title', 'core_plot', 'character_growth', 'relationship_changes',
              'main_storyline_progression', 'ending_hook')
    for expected, episode in zip(range(start, end + 1), episodes):
        if not isinstance(episode, dict) or type(episode.get('episode_id')) is not int or episode['episode_id'] != expected:
            raise ValueError(f'集号须按顺序连续，当前位置必须为第 {expected} 集；不得跳过前三集')
        for field in fields:
            if not isinstance(episode.get(field), str) or not episode[field].strip():
                raise ValueError(f'第 {expected} 集 {field} 必须是非空文字')
        roles = episode.get('roles')
        if not isinstance(roles, list) or not roles or any(not isinstance(r, str) or not r.strip() for r in roles):
            raise ValueError(f'第 {expected} 集 roles 必须是非空角色名数组')
        highlights = episode.get('highlights')
        if not ((isinstance(highlights, str) and highlights.strip()) or
                (isinstance(highlights, list) and highlights and all(isinstance(h, str) and h.strip() for h in highlights))):
            raise ValueError(f'第 {expected} 集 highlights 必须是非空文字或文字列表')


def make_outline_llm(prompt):
    config = get_model_config('episode_outline_model')
    return LLM(**{key: config[key] for key in ('model', 'system_prompt', 'temperature',
               'top_p', 'top_k', 'max_length', 'max_new_tokens')},
               template=jinja2.Template(prompt))


async def _validated_call(ctx, llm, params, task, artifact_key, validate):
    feedback = ''
    for attempt in range(1, 4):
        actual = dict(params, RetryFeedback=feedback)
        result = await llm_inference_json(params=actual, llm_service=llm, task_name=task, ctx=ctx)
        try:
            validate(result)
        except ValueError as error:
            feedback = str(error)
            save_realtime(['04_stage_plan', 'attempts', f'{artifact_key}_{attempt:02}.json'],
                          {'status': 'rejected', 'reason': feedback, 'result': result,
                           'source': getattr(llm, 'last_trace_dir', None)})
            logger.warning_context(ctx, f'{task} 第{attempt}/3次校验失败：{feedback}')
        else:
            save_realtime(['04_stage_plan', 'attempts', f'{artifact_key}_{attempt:02}.json'],
                          {'status': 'accepted', 'result': result})
            return result
    raise RuntimeError(f'{task} 连续3次结构校验失败，停止生成，禁止使用缺集或错位大纲继续：{feedback}')


async def generate_stage_episode_outlines(ctx, story_outline, total, common_params, episode_prompt, *,
                                        batch_validator=None, review_callback=None):
    validate_batch = batch_validator or validate_episode_batch
    stages = extract_stages(story_outline)
    if type(total) is not int or total < len(stages):
        raise ValueError(f'总集数必须为整数且不少于阶段数 {len(stages)}，实际为 {total}')
    output = os.environ.get('DRAMA_OUTPUT_DIR')
    if output and list((Path(output) / '04_episode_outline').glob('*.json')):
        raise FileExistsError('输出目录已有分集大纲，请使用新的 DRAMA_OUTPUT_DIR，避免混入旧批次')
    plan_params = dict(common_params, FullOutline=story_outline,
                       Stages=json.dumps(stages, ensure_ascii=False), TotalEpisodeNums=str(total))
    plan = await _validated_call(ctx, make_outline_llm(GENERATE_STAGE_EPISODE_PLAN_PROMPT),
                                 plan_params, '规划阶段集数范围', 'stage_plan',
                                 lambda value: validate_stage_plan(value, stages, total))
    save_realtime(['04_stage_plan', 'stage_plan.json'],
                  {'total_episode_nums': total, 'stages': plan})
    logger.info_context(ctx, f'阶段集数规划完成：{json.dumps(plan, ensure_ascii=False)}')
    # Only names/ranges for the whole play, never the other stages' event lists or reasoning.
    ranges = [{key: item[key] for key in ('stage', 'start_episode_id', 'end_episode_id')} for item in plan]
    all_episodes = []
    llm = make_outline_llm(episode_prompt)
    for index, (allocation, stage) in enumerate(zip(plan, stages), 1):
        start, end = allocation['start_episode_id'], allocation['end_episode_id']
        params = dict(common_params, TotalEpisodeNums=str(total), StageName=stage['stage'],
                      StageRanges=json.dumps(ranges, ensure_ascii=False),
                      PacingReason=allocation['pacing_reason'],
                      Outline=json.dumps(stage['event_list'], ensure_ascii=False),
                      EpisodeOutline=json.dumps(all_episodes, ensure_ascii=False),
                      StartEpisodeId=str(start), EndEpisodeId=str(end))
        episodes = await _validated_call(ctx, llm, params, f"生成{stage['stage']}阶段集大纲",
                                         f'stage_{index:02}',
                                         lambda value: validate_batch(value, start, end))
        save_realtime(['04_episode_outline', f'chunk_{index:02}_ep{start:02}-{end:02}.json'], episodes)
        all_episodes.extend(episodes)
    validate_batch(all_episodes, 1, total)
    from .episode_outline_review import review_episode_outlines
    review = review_callback or review_episode_outlines
    all_episodes = await review(ctx, all_episodes, story_outline, plan, common_params)
    for index, allocation in enumerate(plan, 1):
        start, end = allocation['start_episode_id'], allocation['end_episode_id']
        save_realtime(['04_episode_outline', f'chunk_{index:02}_ep{start:02}-{end:02}.json'],
                      all_episodes[start-1:end])
    save_realtime(['04_stage_plan', 'completion.json'], {'status': 'complete', 'episode_count': len(all_episodes)})
    return all_episodes
