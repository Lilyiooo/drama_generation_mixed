"""独立实验：润色已有详细大纲，再生成、三轮审校并择优完整逐集大纲。"""
import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from drama_local.runtime import Context, atomic_json
from service.drama_by_creativity.stage_episode_planning import (
    extract_stages, _validated_call, make_outline_llm, generate_stage_episode_outlines,
)
from service.drama_by_creativity.prompts.episode_outline import GENERATE_ALL_EPISODE_OUTLINE_PROMPT
from service.drama_by_creativity.generate_story_outline_and_role import transfer_outline_to_str

POLISH_PROMPT = """你是故事大纲修订编剧。请通读原始详细事件大纲，输出经过润色、衔接和适度调整的完整详细故事大纲。本次不生成逐集大纲。
目标总集数：{{Total}}，请据此考虑整体叙事容量和节奏。
故事基本信息：
{{Background}}
原始详细事件大纲：
{{OriginalStory}}

要求：
1. 以原有详细事件大纲为主线骨架，保留主要人物、核心冲突、关键主线事件和结局方向，不另起一个故事。严格保持“开端、发展、高潮、结局”四阶段及其顺序。
2. 做好相邻事件和跨阶段之间的衔接。核对人物身份、动机、身体与心理状态、关系、知情程度、世界规则、时间地点及事件因果；发现叙事不一致或情节冲突时允许适度调整。把必要的恢复、移动、信息获取、道具交接和行动动机写成实际情节，不用“自然衔接”“恢复正常”等空泛说明代替。
3. 为丰富叙事，允许适量增添与主线及人物发展有关的合理支线，并安排其后续作用或收束；避免引入不必要的新主线或只有开头没有结果的悬念。
4. 为改善节奏，允许适度调整、增删、合并或重排事件，消除重复发生、过度拖延和过早收尾。同步处理调整对前后事件、伏笔与结果的影响，使四阶段整体推进自然。
5. 统一含混或冲突的关键设定，明确能力、治疗等规则的条件和限制，并同步修正背景、亮点和摘要中的相关表述。保留有效信息，不凭空替换核心故事。确保重要证据有取得过程、关键线索有作用、主要冲突有收束。
6. 返回全部四阶段的完整事件列表，包括无需修改的内容；每条事件写清具体行动、变化或结果，保持原大纲的细化程度。不输出集号、剧本正文或备选方案。
{% if RetryFeedback %}上次输出校验反馈：{{RetryFeedback}}{% endif %}
仅输出JSON对象，结构与原详细大纲一致：
{
 "title": "剧名",
 "background": "故事背景",
 "framework": [
  {"stage": "开端", "event_list": [{"event": "具体事件"}]},
  {"stage": "发展", "event_list": [{"event": "具体事件"}]},
  {"stage": "高潮", "event_list": [{"event": "具体事件"}]},
  {"stage": "结局", "event_list": [{"event": "具体事件"}]}
 ],
 "highlights": "故事亮点",
 "summary": "全剧故事摘要"
}
"""


def validate_story(value):
    stages = extract_stages(value)
    if [stage['stage'] for stage in stages] != ['开端', '发展', '高潮', '结局']:
        raise ValueError('必须保留开端、发展、高潮、结局四阶段及其顺序')
    for key in ('title', 'background', 'highlights', 'summary'):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f'详细大纲缺少非空文字字段 {key}')
    for stage in value['framework']:
        if any(not isinstance(event, dict) or not isinstance(event.get('event'), str)
               for event in stage['event_list']):
            raise ValueError('event_list 中每条必须为含 event 文字的对象')


async def generate(out, original, background, total, *, episode_generator=None):
    ctx = Context()
    params = {'Total': str(total), 'OriginalStory': json.dumps(original, ensure_ascii=False),
              'Background': json.dumps(background, ensure_ascii=False)}
    model = make_outline_llm(POLISH_PROMPT)
    (out/'03_story_outline_polish/prompt.txt').write_text(
        model.template.render(**params, RetryFeedback=''), encoding='utf-8')
    print('润色详细故事大纲：衔接、一致性、支线与节奏。', flush=True)
    polished = await _validated_call(ctx, model, params, '详细故事大纲润色', 'story_polish', validate_story)
    atomic_json(out/'03_story_outline_polish/after.json', polished)
    atomic_json(out/'03_story_outline/outline.json', polished)
    common = dict(background, RoleDescription='你是一位擅长叙事一致性与短剧节奏的专业编剧。', DramaType='短剧漫剧')
    common.setdefault('Reference', '')
    # The core story remains the original input; the polished event list drives stage expansion.
    atomic_json(out/'generation_background.json', common)
    print('润色完成；开始阶段集数规划、分阶段生成及三轮评审重写、四版评分择优。', flush=True)
    generate_episodes = episode_generator or generate_stage_episode_outlines
    result = await generate_episodes(
        ctx, transfer_outline_to_str(polished), total, common, GENERATE_ALL_EPISODE_OUTLINE_PROMPT)
    atomic_json(out/'episode_outlines.json', result)
    return result


def main(generate_fn=None, output_prefix="qwen36_27b_polish_story_demo_", *, polish=True):
    parser = argparse.ArgumentParser(description=__doc__ if polish else "直接从原始详细大纲生成逐集大纲，再三轮评审重写并评分择优")
    parser.add_argument('--source-dir', type=Path, default=ROOT/'output/qwen36_27b_outline_only_20260915_165713_621630661')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--episodes', type=int, default=60)
    parser.add_argument('--prepare-only', action='store_true', help='保存输入和提示词，不调用模型')
    args = parser.parse_args()
    if args.episodes < 4:
        parser.error('总集数不能少于四阶段')
    source = args.source_dir.resolve(strict=True)
    original_path = source/'03_story_outline/outline.json'
    background_path = source/'04_episode_outline_review/background.json'
    original = json.loads(original_path.read_text())
    validate_story(original)
    background = json.loads(background_path.read_text())
    for key in ('Topic', 'WorldView', 'StoryOutline', 'RoleInfo'):
        background.setdefault(key, '暂无')
    out = (args.output_dir or ROOT/'output'/(output_prefix+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    out.mkdir(parents=True, exist_ok=False)
    input_folder = out/('03_story_outline_polish' if polish else '03_story_outline')
    input_folder.mkdir()
    os.environ['DRAMA_OUTPUT_DIR'] = str(out)
    atomic_json(out/'source.json', {'story_outline': str(original_path), 'background': str(background_path), 'episodes': args.episodes})
    atomic_json(input_folder/('before.json' if polish else 'outline.json'), original)
    atomic_json(input_folder/'background.json', background)
    if polish:
        (input_folder/'prompt_template.txt').write_text(POLISH_PROMPT, encoding='utf-8')
    print(f'输出目录：{out}', flush=True)
    status = {'status': 'prepared', 'episodes': args.episodes}
    atomic_json(out/'status.json', status)
    if args.prepare_only:
        return
    started = time.monotonic()
    status.update(status='running', started_at_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(out/'status.json', status)
    try:
        asyncio.run((generate_fn or generate)(out, original, background, args.episodes))
        status['status'] = 'complete'
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        status.update(elapsed_seconds=round(time.monotonic()-started, 2), finished_at_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(out/'status.json', status)
        print(f"状态：{status['status']}；耗时：{status['elapsed_seconds']}秒", flush=True)


if __name__ == '__main__':
    main()
