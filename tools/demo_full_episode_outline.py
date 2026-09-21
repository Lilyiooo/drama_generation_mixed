"""Standalone one-request episode-outline experiment; never starts the pipeline."""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROMPT = """{{RoleDescription}}
请根据完整详细故事大纲，一次性生成全剧全部{{TotalEpisodeNums}}集的逐集大纲。

题材：{{Topic}}
世界观：{{WorldView}}
核心故事：{{StoryOutline}}
人物设定：{{RoleInfo}}
{% if Reference %}创作参考：{{Reference}}{% endif %}

完整详细故事大纲（包含所有阶段的事件）：
{{FullOutline}}

前三集样稿，仅供人物、风格与开场参考；第1—3集的大纲仍须在本次结果中生成：
{{Episode1Script}}
{{Episode2Script}}
{{Episode3Script}}

要求：
1. 本任务是将给定详细故事大纲扩展为完整的{{TotalEpisodeNums}}集，不是先用少数集讲完原故事，再续写新的故事。全剧的主要冲突、人物目标、事件范围与结局均以详细大纲为依据。
2. 写作前先在内部统筹所有阶段与主要事件的篇幅，再输出逐集大纲。按照事件的复杂度、因果推进和情感发展合理分配集数，控制推进速度和整体叙事节奏。开端建立人物与核心冲突，发展逐步升级阻力并积累决战条件，高潮集中解决核心冲突，结局用最后少量集数完成回收与收束。不要在前中段完成最终决战、解决核心反派或进入终局退隐。
3. 用已有事件的具体展开支撑集数：可以设计目标与阻力、调查与发现、行动与反制、代价与后果、人物选择与关系变化，让每集都产生实质进展。详细大纲的一条复杂事件可跨多集展开，但各集应承担不同推进步骤；不要一集草率讲完多项关键事件，导致后面无事可写，也不要反复演同一个结果。
4. 不得新增独立于原故事的反派集团、重大阴谋、冒险任务或案件主线来填充集数。情感和生活场景应服务于当前冲突、人物转变或必要收尾，不得用连续聚餐、节庆、四季日常等无主线进展的内容凑集数。若篇幅不足，应增加原有关键事件的有效展开，而不是在原结局之后续写。
5. 完整覆盖主要事件，遵守先后顺序和因果关系。同一事件若在详细大纲中既有展开又有总结，应合并为一次发生；已完成的事件不得重演为首次发生。最终决战与结局应自然相接，不能结束后再重启同一场决战。
6. 检查当前集与前序事件的人物身份、人物或世界状态与关系是否连续或存在情节逻辑矛盾，同时核对动机、知情状态、时间地点、重要道具和事件结果的承接。遵循既定世界观与角色能力，人物伤病、权力和关系的变化须有交代，不随意引入缺乏设定依据的技术。
7. 重要线索应有对应的揭示、作用或回收，关键人物介入和重大反转应提前铺垫。详细大纲自身若有局部矛盾，以故事整体自洽为目标作最小必要调整，保留主要事件和人物弧光。
8. 保持题材风格、人物特色和戏剧冲突，推进手法应有变化。情感发展融入行动与选择，各集结尾钩子自然连接下一集，不以新增无关谜团代替承接。
9. 每集core_plot约200字，完整写清本集具体发生的事情、行动及结果，后半段也保持相同细化程度。写确定的剧情，不夹带创作分析、模糊备选或“本集无剧情推进”等占位说明。
10. 输出前检查全部{{TotalEpisodeNums}}集：原主线是否合理展开至全剧后段，主要事件是否遗漏或重复，是否存在无关填充，关键状态与伏笔是否有交代。只输出最终JSON数组，不输出内部规划、正文或解释。集号从1到{{TotalEpisodeNums}}按顺序连续，包含所有集数和下列完整字段。

格式示例（实际须输出全部{{TotalEpisodeNums}}集）：
[
  {
    "episode_id": 1,
    "title": "本集标题",
    "core_plot": "本集完整核心剧情，约200字",
    "roles": ["人物姓名"],
    "highlights": "本集看点和名场面",
    "character_growth": "人物成长变化",
    "relationship_changes": "人物关系变化",
    "main_storyline_progression": "主线推进",
    "ending_hook": "本集结尾钩子"
  }
]
"""


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=ROOT / 'output/qwen36_27b_all_stages_review_20260914_090107')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--episodes', type=int, default=60)
    parser.add_argument('--prepare-only', action='store_true', help='保存输入和提示词，不调用模型')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    source = args.source_dir.resolve()
    outline_path = source / '03_story_outline/outline.json'
    outline = json.loads(outline_path.read_text(encoding='utf-8'))
    # Recover the same background and samples actually used in the source run.
    record_path = None
    for path in (source / 'diagnostics').glob('*/model_calls/*/record.json'):
        record = json.loads(path.read_text(encoding='utf-8'))
        params = record.get('metadata', {}).get('parameters', {})
        if 'StageName' in params and 'EpisodeOutline' in params:
            record_path = path
            break
    if record_path is None:
        raise ValueError('源目录没有阶段大纲调用记录，无法还原原始故事信息和前三集参考')
    keys = ('RoleDescription', 'Topic', 'WorldView', 'StoryOutline', 'RoleInfo',
            'Reference', 'Episode1Script', 'Episode2Script', 'Episode3Script')
    inputs = {key: params.get(key, '') for key in keys}
    inputs.update(FullOutline=json.dumps(outline, ensure_ascii=False, indent=2),
                  TotalEpisodeNums=args.episodes)
    from service.drama_by_creativity.stage_episode_planning import make_outline_llm, validate_episode_batch
    os.environ.setdefault('DRAMA_LLM_MODEL', 'qwen-local')
    os.environ.setdefault('DRAMA_LLM_MAX_TOKENS', '0')
    llm = make_outline_llm(PROMPT)
    prompt = llm.template.render(**inputs)
    out = (args.output_dir or ROOT / 'output' / ('qwen36_27b_full60_demo_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ['DRAMA_OUTPUT_DIR'] = str(out)
    dump(out / 'inputs.json', inputs)
    dump(out / 'source.json', {'source_dir': str(source), 'outline': str(outline_path), 'context_record': str(record_path)})
    (out / 'prompt.txt').write_text(prompt, encoding='utf-8')
    print(f'输出目录：{out}', flush=True)
    print(f'接口：{llm.url}；模型：{llm.model}', flush=True)
    status = {'status': 'prepared', 'episode_count_requested': args.episodes}
    dump(out / 'status.json', status)
    if args.prepare_only:
        return
    start = time.monotonic()
    try:
        messages = ([{'role': 'system', 'content': llm.system_prompt}] if llm.system_prompt else [])
        messages.append({'role': 'user', 'content': prompt})
        status['status'] = 'running'
        dump(out / 'status.json', status)
        response = llm.request_messages(messages, {'parameters': inputs, 'experiment': 'full_episode_outline_single_call_v2'})
        raw = response.response
        (out / 'raw_output.txt').write_text(raw, encoding='utf-8')
        status['usage'] = response.usage
        # Only remove an optional outer code fence; no model or heuristic JSON repair.
        clean = raw.strip()
        if clean.startswith('```') and clean.endswith('```'):
            clean = clean.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        episodes = json.loads(clean)
        dump(out / 'parsed_output.json', episodes)
        validate_episode_batch(episodes, 1, args.episodes)
        dump(out / 'episode_outlines.json', episodes)
        status['status'] = 'complete'
        print(f'全部{args.episodes}集结构校验通过。', flush=True)
    except Exception as exc:
        status.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raw = getattr(llm, 'last_output', None)
        if isinstance(raw, str):
            (out / 'raw_output.txt').write_text(raw, encoding='utf-8')
        raise
    finally:
        status['elapsed_seconds'] = round(time.monotonic() - start, 2)
        status['diagnostics_dir'] = str(getattr(llm, 'last_trace_dir', '') or '')
        dump(out / 'status.json', status)
        print(f'耗时：{status["elapsed_seconds"]}秒；状态：{status["status"]}', flush=True)


if __name__ == '__main__':
    main()
