"""Three sequential rounds of full-season review and complete rewriting."""
import copy
import json
import math
from .prompts.episode_outline_full_review import REVIEW_PROMPT, REWRITE_PROMPT


def apply_review(episodes, review):
    """Legacy patch helper; the active pipeline uses full rewrites below."""
    from .stage_episode_planning import validate_episode_batch
    total = len(episodes)
    validate_episode_batch(episodes, 1, total)
    if not isinstance(review, dict) or type(review.get('has_changes')) is not bool:
        raise ValueError('审校结果必须包含布尔值 has_changes')
    if not isinstance(review.get('review_summary'), str) or not review['review_summary'].strip():
        raise ValueError('缺少 review_summary')
    changes = review.get('replacements')
    if not isinstance(changes, list) or bool(changes) != review['has_changes']:
        raise ValueError('replacements 必须是数组且与 has_changes 一致')
    result = copy.deepcopy(episodes)
    seen = set()
    for item in changes:
        if not isinstance(item, dict):
            raise ValueError('替换项必须是对象')
        eid = item.get('episode_id')
        if type(eid) is not int or not 1 <= eid <= total or eid in seen:
            raise ValueError('替换集号必须有效且不可重复')
        seen.add(eid)
        if not isinstance(item.get('reason'), str) or not item['reason'].strip():
            raise ValueError(f'第{eid}集缺少修改理由')
        related = item.get('related_episode_ids')
        if not isinstance(related, list) or not related or any(type(n) is not int or not 1 <= n <= total for n in related):
            raise ValueError(f'第{eid}集关联集号无效')
        new = item.get('replacement')
        validate_episode_batch([new], eid, eid)
        if set(new) != set(episodes[eid - 1]):
            raise ValueError(f'第{eid}集替换条目的字段必须与原条目一致')
        result[eid - 1] = copy.deepcopy(new)
    validate_episode_batch(result, 1, total)
    return result


def validate_review(value, episodes=None):
    """Only the score is machine-consumed; leave editorial details unrestricted."""
    if not isinstance(value, dict):
        raise ValueError('评审必须是包含 total_score 的 JSON 对象')
    score = value.get('total_score')
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 100:
        raise ValueError('total_score 必须是 0—100 范围内的有限数值')

async def review_episode_outlines(ctx, episodes, story_outline, plan, common, *,
                                  batch_validator=None, review_prompt=None, rewrite_prompt=None):
    from .stage_episode_planning import _validated_call, make_outline_llm, save_realtime, validate_episode_batch
    validate_batch = batch_validator or validate_episode_batch
    review_prompt = review_prompt or REVIEW_PROMPT
    rewrite_prompt = rewrite_prompt or REWRITE_PROMPT
    total = len(episodes)
    validate_batch(episodes, 1, total)
    current = copy.deepcopy(episodes)
    base = ['04_episode_outline_review']
    save_realtime(base + ['before.json'], current)
    background = {key: common[key] for key in
                  ('Topic', 'WorldView', 'StoryOutline', 'RoleInfo', 'Reference')
                  if common.get(key)}
    save_realtime(base + ['background.json'], background)
    status = {'status': 'running', 'rounds': 3, 'completed_rounds': 0}
    save_realtime(base + ['status.json'], status)
    candidates = []
    try:
        for round_id in range(1, 4):
            folder = base + [f'round_{round_id:02}']
            status.update(current_round=round_id, phase='review')
            save_realtime(base + ['status.json'], status)
            save_realtime(folder + ['before.json'], current)
            params = {'Total': str(total), 'Original': json.dumps(current, ensure_ascii=False),
                      'Background': json.dumps(background, ensure_ascii=False)}
            review = await _validated_call(
                ctx, make_outline_llm(review_prompt), params,
                f'第{round_id}/3轮全剧问题评审', f'outline_round_{round_id:02}_review',
                lambda value: validate_review(value, current))
            save_realtime(folder + ['review_model_result.json'], review)
            save_realtime(folder + ['review.json'], review)
            candidates.append({'version': round_id - 1, 'score': review['total_score'],
                               'episodes': copy.deepcopy(current), 'review': copy.deepcopy(review),
                               'outline_path': f'round_{round_id:02}/before.json',
                               'review_path': f'round_{round_id:02}/review.json'})
            status['phase'] = 'rewrite'
            save_realtime(base + ['status.json'], status)
            revised = await _validated_call(
                ctx, make_outline_llm(rewrite_prompt),
                dict(params, Review=json.dumps(review, ensure_ascii=False)),
                f'第{round_id}/3轮完整大纲重写', f'outline_round_{round_id:02}_rewrite',
                lambda value: validate_batch(value, 1, total))
            save_realtime(folder + ['after.json'], revised)
            current = copy.deepcopy(revised)
            status['completed_rounds'] = round_id
            save_realtime(base + ['status.json'], status)
        # Score the final rewrite with exactly the same rubric and inputs as earlier versions.
        status['phase'] = 'final_review'
        save_realtime(base + ['status.json'], status)
        params = {'Total': str(total), 'Original': json.dumps(current, ensure_ascii=False),
                  'Background': json.dumps(background, ensure_ascii=False)}
        final_review = await _validated_call(
            ctx, make_outline_llm(review_prompt), params,
            '第三轮重写结果最终评审与评分', 'outline_final_review', validate_review)
        save_realtime(base + ['final_review.json'], final_review)
        candidates.append({'version': 3, 'score': final_review['total_score'],
                           'episodes': copy.deepcopy(current), 'review': final_review,
                           'outline_path': 'round_03/after.json', 'review_path': 'final_review.json'})
        # Stable max: ties retain the earlier version, without presuming revisions are better.
        best = max(candidates, key=lambda item: item['score'])
        selection = {'selected_version': best['version'], 'selected_score': best['score'],
                     'tie_break': 'earliest_version',
                     'candidates': [{k: v for k, v in item.items() if k not in ('episodes', 'review')}
                                    for item in candidates]}
        save_realtime(base + ['selection.json'], selection)
        save_realtime(base + ['review.json'], best['review'])
        save_realtime(base + ['after.json'], best['episodes'])
        status.update(status='complete', phase='complete',
                      selected_version=best['version'], selected_score=best['score'])
        save_realtime(base + ['status.json'], status)
        return copy.deepcopy(best['episodes'])
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
        save_realtime(base + ['status.json'], status)
        raise
