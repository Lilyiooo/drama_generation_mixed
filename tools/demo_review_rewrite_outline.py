"""Review the original 2026-09-14 episode outlines, then rewrite the whole season."""
import argparse
import copy
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
    make_outline_llm, _validated_call, validate_episode_batch,
)

REVIEW_PROMPT = """你是全剧叙事审校编辑。请通读以下未经校准的全部逐集大纲，评审叙事一致性并提出可执行的问题清单。本轮只评审，不重写大纲。
故事基本信息：
{{Background}}
总集数：{{Total}}
未经校准的完整逐集大纲：
{{Original}}

评审要求：
1. 检查全剧事件的因果与时序、相邻集衔接、人物身份与动机、人物或世界状态与关系的连续性。检查事项不限于上述内容，重点发现具体矛盾以及不自然的衔接。
2. 核对已经发生的事件是否在后面被重演为首次发生，反派权力、人物伤病、信息获取、道具归属及契约状态是否被无交代地重置；检查核心冲突是否提前结束，后续剧情是否失去依据。
3. 检查重要线索与伏笔是否有铺垫、作用和回收，关键角色的介入是否合理；核对角色成长、关系变化和结尾钩子等字段是否与核心剧情一致。
4. 每个问题列出涉及集号和相关情节概述，允许概括或省略，不要求与原文逐字一致；说明冲突在哪里、对后续的影响及如何修正。区分明确矛盾、缺少交代和可优化项，不将合理的悬念误判为矛盾。
5. 提出一个全剧修订方案，解决问题时保留有效事件、主要人物和故事主题。涉及跨集影响时列出需要联动修改的集号，避免只修局部而制造新矛盾。
必要时，允许对已有大纲情节提出较大范围的修改建议，包括重组、替换或调整多个集的剧情；不要为了少修改而只改个别词句，应以解决情节矛盾、人物状态矛盾等叙事不一致问题为目标。
{% if RetryFeedback %}上次结构校验反馈：{{RetryFeedback}}{% endif %}
只输出JSON对象：
{
 "overall_assessment": "对全剧质量和主要问题的判断",
 "issues": [
  {"issue_id": "I01", "severity": "严重", "category": "叙事一致性",
   "episode_ids": [1, 2], "evidence": [{"episode_id": 1, "summary": "该集与问题相关的情节概述"}],
   "problem": "具体问题", "impact": "对人物或后续剧情的影响",
   "recommendation": "具体改法及联动修改建议"}
 ],
 "revision_strategy": "如何统筹60集的因果、节奏、关键状态和伏笔回收"
}
没有发现问题时issues输出空数组。severity仅使用严重、一般、建议。
"""

REWRITE_PROMPT = """你是全剧大纲修订编剧。依据评审意见和未经校准的原始逐集大纲，重新生成校准后的完整{{Total}}集逐集大纲。
故事基本信息：
{{Background}}
未经校准的原始逐集大纲：
{{Original}}
评审意见（evidence为模型的情节概述，source_episodes为程序按集号附上的原始条目，请据此核对）：
{{Review}}

修订要求：
1. 逐项核对评审提出的问题与原文，按全剧修订方案处理，重点修复情节因果、叙事一致性和跨集衔接；若某条意见缺乏原文依据，以故事自洽和保留有效剧情为准。
2. 这是对现有故事的完整修订。保留主要人物姓名、故事主题、核心冲突、有效剧情和重要线索。允许为消除重复、调整节奏而合并或移动事件，并补充必要的铺垫、行动和过渡；不要新增无关反派、冒险或日常来填充集数。
3. 全剧仍为{{Total}}集，episode_id从1到{{Total}}连续。统筹原主线在全剧的展开，避免过早完成决战后再重演或续写无关故事；收尾在最后少量集完成。原来有问题的剧情位置可以调整，不需要保留原阶段边界。
4. 对修改的前后影响一并处理：人物身份、人物或世界状态与关系、动机、知情状态、伤病、权力、关键道具、契约与伏笔，在全剧中保持连续。已经完成的事件不能无交代地重置。关键线索应有明确的作用或收束。
5. 每集core_plot约200字，写清具体行动与结果，保持后半段细化程度。所有字段互相一致，结尾钩子衔接后续。不要夹带分析、备选剧情或修改说明。
6. 输出前检查评审中的问题是否实际解决，而不是只在措辞上声称解决。输出全部集数，包含无需修改的集；每集使用下列固定字段名，不得改名。
必要时，允许对已有大纲情节进行较大范围的修改，包括重组、替换或调整多个集的剧情，只要能改善情节矛盾、人物状态矛盾等叙事不一致问题；不要为了少修改而只改个别词句，并同步处理修改对前后剧情的影响。
如果存在其他任何叙事不一致的问题，也应在重写时逐一修改大纲予以解决，而非只解决评审中提出的问题。
{% if RetryFeedback %}上次结构校验反馈：{{RetryFeedback}}{% endif %}
只输出一个JSON数组：
[
 {"episode_id": 1, "title": "标题", "core_plot": "本集完整剧情，约200字",
  "roles": ["人物姓名"], "highlights": "本集看点",
  "character_growth": "人物成长", "relationship_changes": "关系变化",
  "main_storyline_progression": "主线推进", "ending_hook": "结尾钩子"}
]
"""

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def validate_review(value, episodes):
    if not isinstance(value, dict):
        raise ValueError('评审必须是JSON对象')
    for key in ('overall_assessment', 'revision_strategy'):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f'缺少{key}')
    if not isinstance(value.get('issues'), list):
        raise ValueError('issues必须为数组')
    seen = set()
    for issue in value['issues']:
        if not isinstance(issue, dict):
            raise ValueError('问题条目必须是对象')
        for key in ('issue_id', 'category', 'problem', 'impact', 'recommendation'):
            if not isinstance(issue.get(key), str) or not issue[key].strip():
                raise ValueError(f'问题条目缺少{key}')
        if issue['issue_id'] in seen:
            raise ValueError('issue_id重复')
        seen.add(issue['issue_id'])
        if issue.get('severity') not in ('严重', '一般', '建议'):
            raise ValueError('severity无效')
        ids = issue.get('episode_ids')
        if not isinstance(ids, list) or not ids or any(type(i) is not int or not 1 <= i <= len(episodes) for i in ids):
            raise ValueError('问题关联集号无效')
        evidence = issue.get('evidence')
        if not isinstance(evidence, list) or not evidence:
            raise ValueError('缺少相关情节概述')
        for item in evidence:
            if not isinstance(item, dict) or type(item.get('episode_id')) is not int or not 1 <= item['episode_id'] <= len(episodes):
                raise ValueError('证据集号必须是有效的全剧集号')
            summary = item.get('summary', item.get('quote'))
            if not isinstance(summary, str) or not summary.strip():
                raise ValueError(f"{issue['issue_id']}缺少相关情节概述")


def attach_source_episodes(review, episodes):
    validate_review(review, episodes)
    result = copy.deepcopy(review)
    referenced = set()
    for issue in result['issues']:
        issue['episode_ids'] = sorted(set(issue['episode_ids']) | {item['episode_id'] for item in issue['evidence']})
        referenced.update(issue['episode_ids'])
        for item in issue['evidence']:
            if 'summary' not in item:
                item['summary'] = item.pop('quote')
    result['source_episodes'] = [copy.deepcopy(episodes[i-1]) for i in sorted(referenced)]
    return result


async def generate(out, episodes, background):
    ctx = Context()
    params = {'Total': str(len(episodes)), 'Original': json.dumps(episodes, ensure_ascii=False),
              'Background': json.dumps(background, ensure_ascii=False)}
    reviewer = make_outline_llm(REVIEW_PROMPT)
    (out/'prompt_review.txt').write_text(reviewer.template.render(**params), encoding='utf-8')
    print('第一步：评审未经校准的全剧大纲。', flush=True)
    review = await _validated_call(ctx, reviewer, params, '全剧问题评审', 'review', lambda v: validate_review(v, episodes))
    dump(out/'review_model_result.json', review)
    review = attach_source_episodes(review, episodes)
    dump(out/'review.json', review)
    print(f"评审提出{len(review['issues'])}项问题，第二步：重新生成完整大纲。", flush=True)
    rewrite_params = dict(params, Review=json.dumps(review, ensure_ascii=False))
    writer = make_outline_llm(REWRITE_PROMPT)
    (out/'prompt_rewrite.txt').write_text(writer.template.render(**rewrite_params), encoding='utf-8')
    result = await _validated_call(ctx, writer, rewrite_params, '完整大纲修订', 'rewrite',
                                   lambda v: validate_episode_batch(v, 1, len(episodes)))
    dump(out/'episode_outlines.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=ROOT/'output/qwen36_27b_all_stages_review_20260914_090107')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    source = args.source_dir.resolve()
    original = source/'04_episode_outline_review/before.json'
    episodes = json.loads(original.read_text(encoding='utf-8'))
    validate_episode_batch(episodes, 1, 60)
    background = None
    for record_path in (source/'diagnostics').glob('*/model_calls/*/record.json'):
        record = json.loads(record_path.read_text())
        params = record.get('metadata', {}).get('parameters', {})
        if 'StageName' in params and 'EpisodeOutline' in params:
            background = {k:params.get(k, '') for k in ('Topic', 'WorldView', 'StoryOutline', 'RoleInfo')}
            break
    if background is None:
        raise ValueError('未找到原运行的故事背景信息')
    out = (args.output_dir or ROOT/'output'/('qwen36_27b_review_rewrite_demo_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ['DRAMA_OUTPUT_DIR'] = str(out)
    os.environ.setdefault('DRAMA_LLM_MODEL', 'qwen-local')
    os.environ.setdefault('DRAMA_LLM_MAX_TOKENS', '0')
    dump(out/'before.json', episodes)
    dump(out/'background.json', background)
    dump(out/'source.json', {'original':str(original), 'background_record':str(record_path)})
    (out/'review_prompt_template.txt').write_text(REVIEW_PROMPT, encoding='utf-8')
    (out/'rewrite_prompt_template.txt').write_text(REWRITE_PROMPT, encoding='utf-8')
    print(f'输出目录：{out}', flush=True)
    status = {'status':'prepared', 'episode_count':60}
    dump(out/'status.json', status)
    if args.prepare_only: return
    start = time.monotonic()
    status['status'] = 'running'
    dump(out/'status.json', status)
    try:
        asyncio.run(generate(out, episodes, background))
        status['status'] = 'complete'
    except Exception as exc:
        status.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        status['elapsed_seconds'] = round(time.monotonic()-start, 2)
        dump(out/'status.json', status)
        print(f"状态：{status['status']}；耗时：{status['elapsed_seconds']}秒", flush=True)


if __name__ == '__main__':
    main()
