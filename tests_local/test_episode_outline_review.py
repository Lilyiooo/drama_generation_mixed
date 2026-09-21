import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from service.drama_by_creativity.episode_outline_review import apply_review
from service.drama_by_creativity import stage_episode_planning as planning


def episodes():
    return [dict(episode_id=i,title='标题',core_plot='原剧情',roles=['主角'],highlights='看点',
                 character_growth='成长',relationship_changes='关系',main_storyline_progression='推进',ending_hook='钩子') for i in range(1,5)]


def review():
    es=episodes()
    return dict(has_changes=True,review_summary='修复跨集衔接',replacements=[
        dict(episode_id=i,related_episode_ids=[2,3],reason='修正人物状态',
             replacement=dict(es[i-1],core_plot=f'修正第{i}集')) for i in [3,2]])


def test_apply_complete_patches_and_noop():
    original=episodes();before=copy.deepcopy(original)
    result=apply_review(original,review())
    assert result[0]==original[0] and result[3]==original[3]
    assert result[1]['core_plot']=='修正第2集' and result[2]['core_plot']=='修正第3集'
    assert original==before
    assert apply_review(original,dict(has_changes=False,review_summary='通过',replacements=[]))==original


@pytest.mark.parametrize('kind',['duplicate','range','mismatch','missing','false','related'])
def test_invalid_patch_rejected_without_mutation(kind):
    original=episodes();r=review()
    if kind=='duplicate':r['replacements'].append(copy.deepcopy(r['replacements'][0]))
    if kind=='range':r['replacements'][0]['episode_id']=5
    if kind=='mismatch':r['replacements'][0]['replacement']['episode_id']=1
    if kind=='missing':del r['replacements'][0]['replacement']['core_plot']
    if kind=='false':r['has_changes']=False
    if kind=='related':r['replacements'][0]['related_episode_ids']=[True]
    with pytest.raises(ValueError):apply_review(original,r)
    assert original==episodes()


@pytest.mark.parametrize('fail_last', [False, True])
def test_three_rounds_use_latest_outline_and_publish_only_when_complete(tmp_path, monkeypatch, fail_last):
    monkeypatch.setenv('DRAMA_OUTPUT_DIR', str(tmp_path))
    plan = [dict(stage=s, start_episode_id=i, end_episode_id=i+1, pacing_reason='节奏')
            for s, i in [('开端', 1), ('结局', 3)]]
    story = {'framework': [{'stage': s, 'event_list': [{'event': s}]} for s in ['开端', '结局']]}
    current = episodes()
    rounds = []
    rewrite_attempts = 0
    async def infer(**kwargs):
        nonlocal current, rewrite_attempts
        p = kwargs['params']
        if 'Original' in p:
            assert json.loads(p['Original']) == current
            assert json.loads((tmp_path/'04_episode_outline/chunk_01_ep01-02.json').read_text()) == episodes()[:2]
            if 'Review' not in p:
                rounds.append('review')
                return {'total_score': len(rounds)*10, 'overall_assessment': '检查完成', 'issues': [], 'revision_strategy': '修正衔接'}
            rewrite_attempts += 1
            # Even an empty issue list still runs the complete rewrite.
            assert json.loads(p['Review'])['issues'] == []
            if rewrite_attempts == 1 or (fail_last and len(rounds) == 5):
                return current[:-1]  # Incomplete rewrites must retry without advancing a round.
            if rewrite_attempts == 2:
                assert p['RetryFeedback']
            rounds.append('rewrite')
            current = [dict(ep, core_plot=ep['core_plot'] + '修订') for ep in current]
            return current
        if 'Stages' in p:
            return plan
        return episodes()[int(p['StartEpisodeId'])-1:int(p['EndEpisodeId'])]
    with patch.object(planning, 'llm_inference_json', side_effect=infer), patch.object(planning, 'make_outline_llm', return_value=SimpleNamespace()):
        if fail_last:
            with pytest.raises(RuntimeError):
                asyncio.run(planning.generate_stage_episode_outlines(None, story, 4, {}, 'template'))
        else:
            result = asyncio.run(planning.generate_stage_episode_outlines(None, story, 4, {}, 'template'))
            assert result == current
    base = tmp_path/'04_episode_outline_review'
    status = json.loads((base/'status.json').read_text())
    assert json.loads((base/'before.json').read_text()) == episodes()
    assert json.loads((base/'round_02/before.json').read_text()) == json.loads((base/'round_01/after.json').read_text())
    assert json.loads((base/'round_03/before.json').read_text()) == json.loads((base/'round_02/after.json').read_text())
    if fail_last:
        assert status['status'] == 'failed' and status['completed_rounds'] == 2
        assert not (base/'after.json').exists()
        assert not (tmp_path/'04_stage_plan/completion.json').exists()
        assert json.loads((tmp_path/'04_episode_outline/chunk_01_ep01-02.json').read_text()) == episodes()[:2]
    else:
        assert rounds == ['review', 'rewrite'] * 3 + ['review']
        assert status['status'] == 'complete' and status['completed_rounds'] == 3
        assert json.loads((base/'after.json').read_text()) == current
        assert json.loads((tmp_path/'04_episode_outline/chunk_01_ep01-02.json').read_text()) == current[:2]
        assert json.loads((tmp_path/'04_episode_outline/chunk_02_ep03-04.json').read_text()) == current[2:]

@pytest.mark.parametrize('scores, expected', [([95, 70, 80, 90], 0), ([70, 95, 80, 90], 1),
                                            ([70, 80, 95, 90], 2), ([70, 80, 90, 95], 3),
                                            ([80, 80, 80, 80], 0)])
def test_selects_best_version_and_preserves_freeform_review(tmp_path, monkeypatch, scores, expected):
    from service.drama_by_creativity.episode_outline_review import review_episode_outlines
    monkeypatch.setenv('DRAMA_OUTPUT_DIR', str(tmp_path))
    review_count = 0
    rewrite_count = 0
    async def infer(**kwargs):
        nonlocal review_count, rewrite_count
        params = kwargs['params']
        if 'Review' not in params:
            score = scores[review_count]
            review_count += 1
            # Editorial schema is intentionally unrestricted, and forwarded intact.
            return {'total_score': score, 'issues': '第2集人物状态矛盾，请联动修改', 'extra': ['自由建议']}
        assert json.loads(params['Review'])['extra'] == ['自由建议']
        rewrite_count += 1
        original = json.loads(params['Original'])
        assert original[0]['core_plot'] == ('原剧情' if rewrite_count == 1 else str(rewrite_count-1))
        return [dict(ep, core_plot=str(rewrite_count)) for ep in original]
    with patch.object(planning, 'llm_inference_json', side_effect=infer), patch.object(planning, 'make_outline_llm', return_value=SimpleNamespace()):
        result = asyncio.run(review_episode_outlines(None, episodes(), 'unused', [], {}))
    assert review_count == 4 and rewrite_count == 3
    assert result[0]['core_plot'] == ('原剧情' if expected == 0 else str(expected))
    base = tmp_path/'04_episode_outline_review'
    selection = json.loads((base/'selection.json').read_text())
    assert selection['selected_version'] == expected
    assert [item['score'] for item in selection['candidates']] == scores
    assert json.loads((base/'after.json').read_text()) == result
    assert json.loads((base/'review.json').read_text())['total_score'] == scores[expected]

@pytest.mark.parametrize('score', [None, True, '90', -1, 101, float('nan'), float('inf')])
def test_invalid_scores_are_rejected(score):
    from service.drama_by_creativity.episode_outline_review import validate_review
    with pytest.raises(ValueError):
        validate_review({'total_score': score})
