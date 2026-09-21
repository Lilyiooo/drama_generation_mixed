import asyncio
import copy
import json
from unittest.mock import patch, AsyncMock
from types import SimpleNamespace
import jinja2
import pytest
from tools import demo_polish_story_to_episode_outline as demo


def story():
    return dict(title='测试', background='背景', highlights='亮点', summary='摘要', framework=[
        dict(stage=s, event_list=[{'event': s+'原事件'}]) for s in ['开端', '发展', '高潮', '结局']])


def test_polished_story_drives_downstream_generation(tmp_path):
    original = story()
    polished = copy.deepcopy(original)
    polished['framework'][1]['event_list'].append({'event': '补充伤势恢复与返京过渡'})
    (tmp_path/'03_story_outline_polish').mkdir()
    async def validated(ctx, llm, params, task, key, validate):
        assert json.loads(params['OriginalStory']) == original
        assert params['Total'] == '60'
        validate(polished)
        return polished
    async def downstream(ctx, outline, total, common, prompt):
        assert '补充伤势恢复与返京过渡' in outline
        assert total == 60
        assert common['StoryOutline'] == '原核心故事'
        assert prompt == demo.GENERATE_ALL_EPISODE_OUTLINE_PROMPT
        return [{'episode_id': 1, 'test': 'selected'}]
    with patch.object(demo, 'make_outline_llm', return_value=SimpleNamespace(template=jinja2.Template(demo.POLISH_PROMPT))), \
         patch.object(demo, '_validated_call', side_effect=validated), \
         patch.object(demo, 'generate_stage_episode_outlines', side_effect=downstream) as generate:
        asyncio.run(demo.generate(tmp_path, original, {'StoryOutline': '原核心故事'}, 60))
    generate.assert_awaited_once()
    assert original == story()
    assert json.loads((tmp_path/'03_story_outline/outline.json').read_text()) == polished
    assert json.loads((tmp_path/'episode_outlines.json').read_text())[0]['test'] == 'selected'


@pytest.mark.parametrize('invalid', ['stages', 'empty', 'metadata'])
def test_invalid_polish_output(invalid):
    value = story()
    if invalid == 'stages':
        value['framework'].reverse()
    elif invalid == 'empty':
        value['framework'][0]['event_list'] = []
    else:
        value['summary'] = ''
    with pytest.raises(ValueError):
        demo.validate_story(value)
