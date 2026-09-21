"""Plan stage ranges, then expand each stage with the complete story outline."""
import argparse
import asyncio
import json
import os
import re
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
from service.drama_by_creativity.prompts.stage_episode_plan import GENERATE_STAGE_EPISODE_PLAN_PROMPT
from service.drama_by_creativity.prompts.episode_outline import GENERATE_EPISODE_OUTLINE_BY_SCRIPT_PROMPT

# Preserve the pipeline template except for full-story visibility and its scope instruction.
PROMPT = GENERATE_EPISODE_OUTLINE_BY_SCRIPT_PROMPT.replace(
    '下方事件列表仅属于该阶段，不是全剧事件线。',
    '下方提供完整详细故事大纲，供你辨认当前阶段与前后阶段的事件边界。'
).replace('### 当前阶段事件列表\n{{Outline}}', '''### 完整详细故事大纲（所有阶段）
{{Outline}}

本次只展开其中“{{StageName}}”阶段的event_list，生成第{{StartEpisodeId}}—{{EndEpisodeId}}集。
前序阶段以已生成的集大纲为准，不能重复生成；后续阶段的内容仅供识别边界，不得在本次集大纲中提及、展开或提前完成其事件、关键揭示和结局。
本阶段的集数必须用于展开本阶段已有事件及必要的行动、阻力、反制和过渡，不得借用后续阶段事件填充篇幅。阶段末尾停在当前阶段最后事件对应的进展，不跨入下一阶段。''')


# This experiment supplies no sample scripts to either planning or expansion.
PLAN_PROMPT = GENERATE_STAGE_EPISODE_PLAN_PROMPT.replace(
    '## 前三集样稿参考\n{{Episode1Script}}\n{{Episode2Script}}\n{{Episode3Script}}\n', ''
).replace('前三集样稿仅供风格与开局参考，不代表第1—3集已从本次规划中扣除。', '')
PROMPT = PROMPT.replace('我们已经有了前三集的剧本内容和故事大纲。', '我们已经有了完整故事大纲。')
PROMPT = re.sub(r'## 前三集剧本内容.*?## 已设计的集大纲', '## 已设计的集大纲', PROMPT, flags=re.S)
PROMPT = re.sub(r'^2\. 前三集剧本作为参考.*$', '2. 从第1集开始生成完整的集大纲，保持人物、风格和剧情的连续性，不得跳过任何集数。', PROMPT, flags=re.M)
PROMPT = PROMPT.replace('前三集样稿仅供参考；本阶段若包含第1—3集，必须同时生成其集大纲。', '本阶段若包含第1—3集，必须同时生成其集大纲。')


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


async def generate(args, common, outline, stages, out):
    ctx = Context()
    total = args.episodes
    plan_params = dict(common, FullOutline=outline, Stages=json.dumps(stages, ensure_ascii=False), TotalEpisodeNums=str(total))
    plan = await _validated_call(ctx, make_outline_llm(PLAN_PROMPT),
        plan_params, '规划阶段集数范围', 'stage_plan', lambda v: validate_stage_plan(v, stages, total))
    save_realtime(['04_stage_plan', 'stage_plan.json'], {'total_episode_nums': total, 'stages': plan})
    ranges = [{k: item[k] for k in ('stage', 'start_episode_id', 'end_episode_id')} for item in plan]
    episodes = []
    llm = make_outline_llm(PROMPT)
    for index, allocation in enumerate(plan, 1):
        start, end = allocation['start_episode_id'], allocation['end_episode_id']
        params = dict(common, TotalEpisodeNums=str(total), StageName=allocation['stage'],
            StageRanges=json.dumps(ranges, ensure_ascii=False), PacingReason=allocation['pacing_reason'],
            Outline=outline, EpisodeOutline=json.dumps(episodes, ensure_ascii=False),
            StartEpisodeId=str(start), EndEpisodeId=str(end))
        (out / f'prompt_stage_{index:02}.txt').write_text(llm.template.render(**params), encoding='utf-8')
        print(f"生成{allocation['stage']}：第{start}—{end}集", flush=True)
        batch = await _validated_call(ctx, llm, params, f"生成{allocation['stage']}阶段集大纲（全大纲上下文）",
            f'stage_{index:02}', lambda v: validate_episode_batch(v, start, end))
        save_realtime(['04_episode_outline', f'chunk_{index:02}_ep{start:02}-{end:02}.json'], batch)
        episodes.extend(batch)
    validate_episode_batch(episodes, 1, total)
    dump(out / 'episode_outlines.json', episodes)
    save_realtime(['04_stage_plan', 'completion.json'], {'status': 'complete', 'episode_count': len(episodes)})


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
    out = (args.output_dir or ROOT / 'output' / ('qwen36_27b_stage_full_context_demo_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ['DRAMA_OUTPUT_DIR'] = str(out)
    os.environ.setdefault('DRAMA_LLM_MODEL', 'qwen-local')
    os.environ.setdefault('DRAMA_LLM_MAX_TOKENS', '0')
    outline = json.dumps(outline_data, ensure_ascii=False, indent=2)
    dump(out / 'inputs.json', dict(common, FullOutline=outline_data, TotalEpisodeNums=args.episodes))
    dump(out / 'source.json', {'source_dir': str(source), 'context_record': str(record_path),
        'experiment': 'stage_full_context_no_samples', 'semantic_review': False})
    (out / 'stage_prompt_template.txt').write_text(PROMPT, encoding='utf-8')
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
