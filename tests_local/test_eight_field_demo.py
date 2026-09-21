import asyncio
import json
from unittest.mock import patch
from types import SimpleNamespace
import jinja2
import pytest
from tools import demo_eight_field_episode_outline as demo
from service.drama_by_creativity import stage_episode_planning as planning


def episode(i):
    return dict(episode_id=i, title='标题', core_plot='剧情', roles=['主角'], highlights='看点', character_growth='成长', relationship_changes='关系', main_storyline_progression='推进')


@pytest.mark.parametrize('kind', ['extra', 'missing', 'id', 'empty'])
def test_rejects_invalid_eight_field_output(kind):
    ep = episode(1)
    if kind == 'extra': ep['ending_hook'] = '不要这个字段'
    if kind == 'missing': del ep['title']
    if kind == 'id': ep['episode_id'] = True
    if kind == 'empty': ep['roles'] = []
    with pytest.raises(ValueError): demo.validate_eight_fields([ep], 1, 1)


def test_eight_fields_through_stages_and_three_rewrites(tmp_path, monkeypatch):
    monkeypatch.setenv('DRAMA_OUTPUT_DIR', str(tmp_path))
    stages = ['开端', '发展', '高潮', '结局']
    story = {'framework': [dict(stage=s, event_list=[{'event':s}]) for s in stages]}
    reviews, rewrites = [], []
    async def infer(**kwargs):
        p = kwargs['params']
        rendered = kwargs['llm_service'].template.render(**p)
        if 'FullOutline' in p:
            return [dict(stage=s, start_episode_id=i, end_episode_id=i, pacing_reason='节奏') for i,s in enumerate(stages, 1)]
        assert '"ending_hook"' not in rendered
        if 'Review' in p or 'Original' not in p:
            assert '"highlights"' in rendered
        if 'Original' in p:
            original = json.loads(p['Original'])
            demo.validate_eight_fields(original, 1, 4)
            if 'Review' in p:
                rewrites.append(p)
                return [dict(ep, core_plot=ep['core_plot']+'改') for ep in original]
            reviews.append(p)
            return {'total_score': 70 + len(reviews)}
        return [episode(int(p['StartEpisodeId']))]
    with patch.object(planning,'llm_inference_json',side_effect=infer), patch.object(planning,'make_outline_llm',side_effect=lambda text:SimpleNamespace(template=jinja2.Template(text))):
        result = asyncio.run(demo.generate_eight_fields(None, story, 4, {}, 'unused'))
    assert len(reviews) == 4 and len(rewrites) == 3
    assert result[0]['core_plot'] == '剧情改改改'
    demo.validate_eight_fields(result, 1, 4)
    assert json.loads((tmp_path/'04_episode_outline_review/after.json').read_text()) == result


def test_demo_skips_polishing_and_passes_original_story(tmp_path):
    original = dict(title='原剧名', background='背景', highlights='亮点', summary='摘要',
                    framework=[dict(stage=s, event_list=[{'event': s+'原事件'}]) for s in ['开端','发展','高潮','结局']])
    async def downstream(ctx, story, total, common, prompt):
        assert story == demo.base.transfer_outline_to_str(original)
        return [episode(1)]
    with patch.object(demo.base, 'generate', side_effect=AssertionError('不得调用润色流程')), \
         patch.object(demo, 'generate_eight_fields', side_effect=downstream) as generate:
        asyncio.run(demo.generate(tmp_path, original, {}, 60))
    generate.assert_awaited_once()
    assert not (tmp_path/'03_story_outline_polish').exists()
    assert json.loads((tmp_path/'03_story_outline/outline.json').read_text()) == original
