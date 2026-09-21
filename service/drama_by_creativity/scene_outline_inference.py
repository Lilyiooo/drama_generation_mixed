"""Scene outlines: three original generations, then JSON-only repair."""
import json
import re
import uuid

import jinja2
from drama_local.diagnostics import save_event
from .local_llm import LLM
from .prompts.scene_repair import REPAIR_SCENE_OUTLINE_JSON_PROMPT
from .utils import get_model_config, post_process

TEXT_FIELDS = ('场次编号', '时间与地点', '核心功能', '情节概要', '悬念钩子')
LIST_FIELDS = ('出场人物', '冲突与张力', '主要情节')


def validate_scene_outline(raw, episode_number):
    value = json.loads(post_process(raw))
    if not isinstance(value, list) or not value:
        raise ValueError('场次大纲必须是非空对象列表')
    seen = set()
    for scene in value:
        if not isinstance(scene, dict) or not scene:
            raise ValueError('每个场次必须是非空对象')
        for key in TEXT_FIELDS:
            if not isinstance(scene.get(key), str) or not scene[key].strip():
                raise ValueError(f'场次缺少非空文字字段：{key}')
        for key in LIST_FIELDS:
            items = scene.get(key)
            if not isinstance(items, list) or not items or any(not isinstance(x, str) or not x.strip() for x in items):
                raise ValueError(f'场次字段必须为非空文字列表：{key}')
        motives = scene.get('人物动机与目标')
        if not isinstance(motives, list) or not motives or any(
            not isinstance(m, dict) or any(not isinstance(m.get(k),str) or not m[k].strip()
                                          for k in ('角色','动机与目标')) for m in motives
        ):
            raise ValueError('人物动机与目标必须包含非空角色和动机与目标')
        match = re.fullmatch(r'(\d+)-(\d+)(?:（[^）]*）|\([^)]*\))?', scene['场次编号'].strip())
        if not match or int(match[1]) != int(episode_number) or int(match[2]) < 1:
            raise ValueError(f'场次编号与本集不符或格式错误：{scene["场次编号"]}')
        scene_id = (int(match[1]), int(match[2]))
        if scene_id in seen:
            raise ValueError(f'场次编号重复：{scene["场次编号"]}')
        seen.add(scene_id)
    return value


def validate_repair_preserves_scenes(original, repaired):
    # Even malformed JSON generally retains readable scene IDs. Catch the
    # observed failure where repair returned only the first of several scenes.
    before = re.findall(r'"场次编号"\s*[:：]\s*"\s*(\d+-\d+)', original)
    after = [re.match(r'\d+-\d+', scene['场次编号'].strip()).group() for scene in repaired]
    if before and before != after:
        raise ValueError(f'修复改变了场次数量、编号或顺序：原文{before}，修复后{after}')


async def scene_outline_inference_json(params, llm_service, task_name, ctx,
                                       generation_attempts=3, repair_attempts=3):
    if generation_attempts < 1 or repair_attempts < 1:
        raise ValueError('attempt counts must be positive')
    episode_number = params['EpisodeIdx']
    operation_id = uuid.uuid4().hex
    last_error = None
    original = None

    async def attempt(service, request_params, generation, repair, preserve=None):
        nonlocal last_error
        raw = None
        try:
            raw = (await service.request(ctx, request_params, uuid.uuid4().hex)).response
            trace = save_event('json_attempts', task_name=task_name, operation_id=operation_id,
                               context=dict(getattr(ctx, 'fields', {})), generation_attempt=generation,
                               repair_attempt=repair, model_call=getattr(service, 'last_trace_dir', None), text=raw)
            llm_service.last_output = raw
            llm_service.last_trace_dir = trace
            result = validate_scene_outline(raw, episode_number)
            if preserve is not None:
                validate_repair_preserves_scenes(preserve, result)
            llm_service.last_error = None
            save_event('json_validation', task_name=task_name, operation_id=operation_id,
                       generation_attempt=generation, repair_attempt=repair, status='accepted',
                       source=trace)
            return result, raw
        except Exception as exc:
            last_error = f'{type(exc).__name__}: {exc}'
            if raw is None:
                llm_service.last_output = getattr(service, 'last_output', None)
                llm_service.last_trace_dir = getattr(service, 'last_trace_dir', None)
            llm_service.last_error = last_error
            save_event('json_validation', task_name=task_name, operation_id=operation_id,
                       generation_attempt=generation, repair_attempt=repair, status='rejected',
                       reason=last_error, source=getattr(llm_service, 'last_trace_dir', None), text=raw)
            return None, raw

    for generation in range(1, generation_attempts + 1):
        result, raw = await attempt(llm_service, params, generation, 0)
        if result is not None:
            return result
        if raw is not None:
            original = raw
        save_event('json_retry', task_name=task_name, operation_id=operation_id,
                   generation_attempt=generation, next_action='regenerate' if generation < generation_attempts else 'repair',
                   reason=last_error, source=getattr(llm_service, 'last_trace_dir', None))

    if original is not None:
        config = get_model_config('auxiliary_model')
        repair_service = LLM(**config, template=jinja2.Template(REPAIR_SCENE_OUTLINE_JSON_PROMPT))
        for repair in range(1, repair_attempts + 1):
            # Always repair the last original generation, never a repair that
            # may already have deleted scenes or changed the prose.
            result, _ = await attempt(repair_service, {
                'EpisodeIdx': episode_number, 'Ret': original, 'ValidationError': last_error,
            }, generation_attempts, repair, preserve=original)
            if result is not None:
                return result
            save_event('json_retry', task_name=task_name, operation_id=operation_id,
                       generation_attempt=generation_attempts, repair_attempt=repair,
                       next_action='repair' if repair < repair_attempts else 'return_empty_object',
                       reason=last_error, source=getattr(llm_service, 'last_trace_dir', None))
    save_event('scene_outline_failure', task_name=task_name, operation_id=operation_id,
               episode_number=episode_number, reason=last_error,
               source=getattr(llm_service, 'last_trace_dir', None), text=getattr(llm_service,'last_output',None))
    return {}
