"""Standalone experiment: plan stage ranges, then generate all episode outlines once."""
import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from drama_local.runtime import Context
from service.drama_by_creativity.stage_episode_planning import (
    extract_stages, validate_stage_plan, validate_episode_batch,
    make_outline_llm, _validated_call, save_realtime,
)
from tools.demo_stage_full_context import PLAN_PROMPT, dump
from tools.demo_full_episode_outline import PROMPT as FULL_PROMPT

PROMPT = FULL_PROMPT.replace(
    '完整详细故事大纲（包含所有阶段的事件）：\n{{FullOutline}}',
    '全剧背景与主题：\n{{StoryContext}}\n\n已确定的阶段集数与对应事件（必须按范围展开）：\n{{StageAssignments}}'
)
_sample_start = PROMPT.index('前三集样稿，')
_sample_end = PROMPT.index('要求：', _sample_start)
PROMPT = PROMPT[:_sample_start] + PROMPT[_sample_end:]
PROMPT = PROMPT.replace(
    '写作前先在内部统筹所有阶段与主要事件的篇幅，再输出逐集大纲。按照事件的复杂度、因果推进和情感发展合理分配集数，控制推进速度和整体叙事节奏。',
    '阶段集数范围已经确定，不得重新分配。严格在每段指定的集数内展开其对应事件，控制推进速度和整体叙事节奏。前一阶段不得提前发生后一阶段的事件、揭示其关键真相或完成其结局；本阶段末尾完成本阶段规定的进展，在下一阶段指定的起始集才进入下一组事件。'
)


def stage_assignments(plan, stages, total):
    validate_stage_plan(plan, stages, total)
    blocks = []
    for allocation, stage in zip(plan, stages):
        first, last = allocation['start_episode_id'], allocation['end_episode_id']
        events = '\n'.join(f"{i}. {item['event']}" for i, item in enumerate(stage['event_list'], 1))
        blocks.append(f"第{first}—{last}集（共{last-first+1}集），阶段：{stage['stage']}\n"
                      f"节奏说明：{allocation['pacing_reason']}\n对应事件：\n{events}")
    return '\n\n'.join(blocks)


async def generate(args, common, outline, stages, out):
    ctx = Context()
    total = args.episodes
    plan_params = dict(common, FullOutline=outline, Stages=json.dumps(stages, ensure_ascii=False), TotalEpisodeNums=str(total))
    plan = await _validated_call(ctx, make_outline_llm(PLAN_PROMPT), plan_params,
        '规划阶段集数范围', 'stage_plan', lambda v: validate_stage_plan(v, stages, total))
    save_realtime(['04_stage_plan', 'stage_plan.json'], {'total_episode_nums': total, 'stages': plan})
    background = json.loads(outline)
    background.pop('framework', None)
    params = dict(common, TotalEpisodeNums=str(total),
        StoryContext=json.dumps(background, ensure_ascii=False, indent=2),
        StageAssignments=stage_assignments(plan, stages, total))
    dump(out / 'generation_inputs.json', params)
    llm = make_outline_llm(PROMPT)
    prompt = llm.template.render(**params)
    (out / 'prompt_generation.txt').write_text(prompt, encoding='utf-8')
    messages = ([{'role': 'system', 'content': llm.system_prompt}] if llm.system_prompt else [])
    messages.append({'role': 'user', 'content': prompt})
    print(f'阶段规划完成，一次生成全部{total}集。', flush=True)
    try:
        response = await asyncio.to_thread(llm.request_messages, messages,
            {'parameters': params, 'experiment': 'planned_single_call_no_samples'})
        raw = response.response
        (out / 'raw_output.txt').write_text(raw, encoding='utf-8')
        dump(out / 'usage.json', response.usage)
        clean = raw.strip()
        if clean.startswith('```') and clean.endswith('```'):
            clean = clean.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        episodes = json.loads(clean)
        dump(out / 'parsed_output.json', episodes)
        validate_episode_batch(episodes, 1, total)
        dump(out / 'episode_outlines.json', episodes)
    finally:
        raw = getattr(llm, 'last_output', None)
        if isinstance(raw, str):
            (out / 'raw_output.txt').write_text(raw, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=ROOT / 'output/qwen36_27b_all_stages_review_20260914_090107')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--episodes', type=int, default=60)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    source = args.source_dir.resolve()
    outline_data = json.loads((source / '03_story_outline/outline.json').read_text())
    stages = extract_stages(outline_data)
    if args.episodes < len(stages):
        parser.error('总集数不能少于阶段数')
    common = None
    for record_path in (source / 'diagnostics').glob('*/model_calls/*/record.json'):
        record = json.loads(record_path.read_text())
        params = record.get('metadata', {}).get('parameters', {})
        if 'StageName' in params and 'EpisodeOutline' in params:
            common = {key: params.get(key, '') for key in ('RoleDescription', 'Topic', 'WorldView',
                'StoryOutline', 'RoleInfo', 'Reference', 'DramaType')}
            break
    if common is None:
        raise ValueError('未找到源运行的阶段大纲请求，无法还原故事信息')
    out = (args.output_dir or ROOT / 'output' / ('qwen36_27b_planned_full60_demo_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ['DRAMA_OUTPUT_DIR'] = str(out)
    os.environ.setdefault('DRAMA_LLM_MODEL', 'qwen-local')
    os.environ.setdefault('DRAMA_LLM_MAX_TOKENS', '0')
    outline = json.dumps(outline_data, ensure_ascii=False, indent=2)
    dump(out / 'inputs.json', dict(common, FullOutline=outline_data, TotalEpisodeNums=args.episodes))
    dump(out / 'source.json', {'source_dir': str(source), 'context_record': str(record_path),
        'experiment': 'planned_single_call_no_samples', 'semantic_review': False})
    (out / 'generation_prompt_template.txt').write_text(PROMPT, encoding='utf-8')
    plan_llm = make_outline_llm(PLAN_PROMPT)
    (out / 'prompt_plan.txt').write_text(plan_llm.template.render(**dict(common, FullOutline=outline,
        Stages=json.dumps(stages, ensure_ascii=False), TotalEpisodeNums=str(args.episodes))), encoding='utf-8')
    print(f'输出目录：{out}', flush=True)
    print(f'接口：{plan_llm.url}；模型：{plan_llm.model}', flush=True)
    status = {'status': 'prepared', 'episode_count_requested': args.episodes}
    dump(out / 'status.json', status)
    if args.prepare_only:
        return
    start = time.monotonic()
    status['status'] = 'running'
    dump(out / 'status.json', status)
    try:
        asyncio.run(generate(args, common, outline, stages, out))
        status['status'] = 'complete'
    except Exception as exc:
        status.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        status['elapsed_seconds'] = round(time.monotonic() - start, 2)
        dump(out / 'status.json', status)
        print(f"状态：{status['status']}；耗时：{status['elapsed_seconds']}秒", flush=True)


if __name__ == '__main__':
    main()
