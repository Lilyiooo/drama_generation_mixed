"""Production entry, saved selection and downstream text must share the demo contract."""
import asyncio
import importlib
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

import jinja2
import pytest
from service.drama_by_creativity import stage_episode_planning as planning
from service.drama_by_creativity.eight_field_episode_outline import validate_eight_fields
from tests_local.test_eight_field_demo import episode

entry = importlib.import_module('service.drama_by_creativity.generate_episode_outline')

@pytest.mark.parametrize('summary,outline_only', [(False, False), (True, False), (False, True)])
def test_production_selection_and_downstream(tmp_path, monkeypatch, summary, outline_only):
    monkeypatch.setenv('DRAMA_OUTPUT_DIR', str(tmp_path))
    monkeypatch.setenv('DRAMA_OUTLINE_ONLY', '1' if outline_only else '0')
    stages = ['开端', '发展', '高潮', '结局']
    story = json.dumps({'framework': [dict(stage=s, event_list=[{'event': s}]) for s in stages]}, ensure_ascii=False)
    request = NS(generated_data=NS(story_outline=NS(story_outline=[story], role_info=['人物'])),
                 generate_input=NS(common=NS(episode_nums=4), core_story='核心故事', topic='主题', world_view='背景', role_setting='人物'),
                 story_info=NS(plot_type=0))
    reviews = []
    async def infer(**kwargs):
        p = kwargs['params']
        if 'FullOutline' in p:
            return [dict(stage=s, start_episode_id=i, end_episode_id=i, pacing_reason='分配') for i,s in enumerate(stages, 1)]
        rendered = kwargs['llm_service'].template.render(**p)
        assert '"ending_hook"' not in rendered
        if 'Original' in p:
            if 'Review' in p:
                return [dict(ep, core_plot=ep['core_plot']+'改') for ep in json.loads(p['Original'])]
            reviews.append(p)
            return {'total_score': [70, 80, 90, 75][len(reviews)-1]}
        return [episode(int(p['StartEpisodeId']))]
    with patch.object(planning, 'llm_inference_json', side_effect=infer), \
         patch.object(planning, 'make_outline_llm', side_effect=lambda text: NS(template=jinja2.Template(text))), \
         patch.object(entry, 'is_plot_retrieval_disabled', return_value=True), \
         patch.object(entry, 'select_prompt_reference', return_value=''), \
         patch.object(entry, 'summary_baseline_enabled', return_value=summary), \
         patch.object(entry, 'generate_future_map_artifacts', new_callable=AsyncMock) as future:
        rsp = asyncio.run(entry.generate_episode_outline(None, request))
    selected = json.loads((tmp_path/'episode_outlines.json').read_text())
    validate_eight_fields(selected, 1, 4)
    assert selected[0]['core_plot'] == '剧情改改'
    assert selected == json.loads((tmp_path/'04_episode_outline_review/after.json').read_text())
    chunks = [ep for f in sorted((tmp_path/'04_episode_outline').glob('*.json')) for ep in json.loads(f.read_text())]
    assert chunks == selected
    assert len(reviews) == 4
    episodes = rsp.episode_outline.seasons[0].episodes
    assert len(episodes) == 4 and episodes[0].episode_id == 0
    assert '剧情改改' in episodes[0].content
    assert '本集钩子' not in episodes[0].content
    if summary or outline_only:
        future.assert_not_called()
    else:
        future.assert_awaited_once_with(None, selected)


def test_resume_eight_and_legacy_nine_fields(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(entry.__file__).parent/'tests'))
    from service.drama_by_creativity.tests.run_from_episode_outline import load_episode_outlines, build_episode_outline
    folder = tmp_path/'04_episode_outline'
    folder.mkdir()
    for ep in [episode(1), dict(episode(1), ending_hook='旧版钩子')]:
        (folder/'chunk.json').write_text(json.dumps([ep]))
        loaded = load_episode_outlines(str(tmp_path))
        text = build_episode_outline(loaded).seasons[0].episodes[0].content
        assert '未正确解析本集钩子' not in text
        assert ('旧版钩子' in text) == ('ending_hook' in ep)
