"""Replay saved model outputs through the full 60-episode local pipeline.

Does not measure Qwen quality. No remote service calls; fresh temporary output.
Eventline search is covered separately by the existing eventline tests.
"""
import asyncio
from collections import Counter
import contextlib
import importlib
import json
import os
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import jinja2
from drama_local.runtime import Context, LocalStore, ScriptContentFile
from service.drama_by_creativity import DramaByCreativity, prompts
from service.drama_by_creativity import narrative_memory
from service.drama_by_creativity.local_llm import LLM
from service.drama_by_creativity.tests import run_test

SOURCE = Path(__file__).resolve().parents[1]/'output/60ep_0906_1302_v9_future5_state2_complete_noqd'


@unittest.skipUnless(SOURCE.exists(), 'Original generation fixtures not present')
class PipelineReplayTests(unittest.TestCase):
    baseline = False
    def test_full_pipeline(self):
        def read(path):
            return json.loads((SOURCE/path).read_text())
        names = {v:k for module_name,module in list(sys.modules.items()) if module_name.startswith("service.drama_by_creativity") for k,v in vars(module).items()
                 if k.endswith('_PROMPT') and isinstance(v,str)}
        from service.drama_by_creativity.summary_memory import memory_prompt
        if self.baseline:
            names.update({memory_prompt(text, True): name for text, name in list(names.items())})
        original_template = jinja2.Template
        def template(source, *args, **kwargs):
            result = original_template(source, *args, **kwargs)
            result.replay_name = names.get(source, 'UNKNOWN')
            return result
        calls = Counter()
        async def request(client, ctx, params, request_id):
            name = client.template.replay_name
            calls[name] += 1
            mapping = {
                'GENERATE_WORLD_VIEW_PROMPT':'01_script_proposal/01_world_view.json',
                'GENERATE_ROLE_SETTING_PROMPT':'01_script_proposal/02_role_setting.json',
                'GENERATE_PLOT_POINT_PROMPT':'01_script_proposal/03_plot_point.json',
                'GENERATE_PREV_THREE_EPISODE_OUTLINE_PROMPT':'02_demo_drama/prev_three_outline.json',
                'GENERATE_OUTLINE_BY_SCRIPT_PROMPT':'03_story_outline/outline.json',
                'GENERATE_FUTURE_MAP_PROMPT':'04_future_map/future_map.json',
                'GENERATE_EPISODE_CONTRIBUTIONS_PROMPT':'04_future_map/episode_contributions.json',
                'REGENERATE_SINGLE_EPISODE_OUTLINE_PROMPT':'regenerate_single_episode_outline/outline.json',
            }
            if 'Original' in params:
                value = (json.loads(params['Original']) if 'Review' in params else
                         {'total_score': 80, 'overall_assessment': '回放无需修改', 'issues': [], 'revision_strategy': '保持一致'})
            elif 'Stages' in params and 'FullOutline' in params:
                stages = json.loads(params['Stages'])
                bounds = [(1, 3), (4, 13), (14, 20), (21, 60)]
                value = [dict(stage=stage['stage'], start_episode_id=start,
                              end_episode_id=end, pacing_reason='回放用固定范围')
                         for stage, (start, end) in zip(stages, bounds)]
            elif name in mapping:
                value = read(mapping[name])
            elif 'StartEpisodeId' in params:
                all_eps = read('04_episode_outline/chunk_01_ep01-30.json') + read('04_episode_outline/chunk_02_ep31-60.json')
                value = [ep for ep in all_eps if int(params['StartEpisodeId']) <= ep['episode_id'] <= int(params['EndEpisodeId'])]
                # The preceding scripts must have survived local storage reads.
                self.assertNotIn('Episode1Script', params)
            elif name == 'SUMMARIZE_EPISODE_PROMPT':
                self.assertTrue(params['EpisodeScript'])
                value = '回放梗概'+params['EpisodeNumber']
            elif name == 'GENERATE_SCENE_OUTLINE_PROMPT':
                if self.baseline:
                    ep = int(params['EpisodeIdx'])
                    self.assertEqual(params['RecentEpisodeSummaries'], '')
                    for earlier in range(1, ep):
                        self.assertIn(f'第{earlier}集梗概：\n回放梗概{earlier}', params['NarrativeMemory'])
                    self.assertNotIn(f'第{ep}集梗概：', params['NarrativeMemory'])
                value = [{'场次编号': params['EpisodeIdx']+'-1', '时间与地点':'日，测试地点',
                          '出场人物':['苏九娘'], '核心功能':'推进剧情', '情节概要':'人物推进情节',
                          '人物动机与目标':[{'角色':'苏九娘','动机与目标':'推进剧情'}],
                          '冲突与张力':['目标受阻'], '主要情节':['人物解决困难'], '悬念钩子':'后续事件'}]
            elif name == 'GENERATE_WHOLE_EPISODE_PROMPT':
                value = read(f"05_drama/episode_{int(params['EpisodeIndex']):02d}.json")['content']
            elif name == 'POLISH_PLOT_PROMPT':
                value = params['PrevShortScript']+'\n####\n'+params['NextShortScript']
            elif name == 'ADJUST_WORD_COUNT_PROMPT':
                value = params['Script']
            elif name == 'FUTURE_MAP_UPDATE_PROMPT':
                record = read(f"04_future_map/future_driving_updates/episode_{int(params['EpisodeNumber']):03d}.json")
                value = {'episode_summary':'回放摘要', 'future_driving_nodes':record['raw_future_driving_nodes']}
            elif name == 'WORLD_STATE_REWRITE_PROMPT':
                value = {'world_state':[], 'deleted_states':[]}
            else:
                raise AssertionError(f'Unhandled model call: {name}; keys={list(params)}')
            return SimpleNamespace(response=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False))

        with tempfile.TemporaryDirectory() as output, patch.dict(os.environ, {
            'DRAMA_OUTPUT_DIR': output, 'DRAMA_DISABLE_PLOT_RETRIEVAL':'1',
            'DRAMA_EVENTLINE_REFINEMENT_ENABLED':'0',
            'DRAMA_SUMMARY_MEMORY_BASELINE':'1' if self.baseline else '0',
        }), patch('jinja2.Template', template), patch.object(LLM,'request',request), \
                open(os.devnull,'w') as sink, contextlib.redirect_stdout(sink):
            args=SimpleNamespace(story_id='replay',episode_nums=60,plot_type=run_test.pb.PlotType.SHORT_CARTOON)
            result=asyncio.run(run_test.run_full_pipeline(Context(),DramaByCreativity(LocalStore(output)),args))
            episodes=result['drama'].result.seasons[0].episodes
            self.assertEqual([ep.episode_id for ep in episodes],list(range(60)))
            for i in range(60):
                saved=asyncio.run(ScriptContentFile.download(None,LocalStore(output),project_id='replay',episode_id=i))
                self.assertEqual(saved.result,read(f'05_drama/episode_{i+1:02d}.json')['content'])
            self.assertEqual(len(list((Path(output)/'05_drama').glob('episode_*.json'))),60)
            if self.baseline:
                self.assertFalse((Path(output)/'04_future_map').exists())
                self.assertFalse((Path(output)/'narrative_memory').exists())
                state = json.loads((Path(output)/'summary_memory/replay/memory_state.json').read_text())
                self.assertEqual(len(state['episode_summaries']), 60)
                self.assertEqual(calls['SUMMARIZE_EPISODE_PROMPT'], 60)
                self.assertEqual(calls['FUTURE_MAP_UPDATE_PROMPT'], 0)
                self.assertEqual(calls['WORLD_STATE_REWRITE_PROMPT'], 0)
                self.assertEqual(calls['GENERATE_FUTURE_MAP_PROMPT'], 0)
            else:
                self.assertTrue((Path(output)/'04_future_map/future_map.json').exists())
                self.assertTrue((Path(output)/'narrative_memory/replay/world_state.json').exists())
                self.assertEqual(calls['FUTURE_MAP_UPDATE_PROMPT'],63)
                self.assertEqual(calls['WORLD_STATE_REWRITE_PROMPT'],63)
            self.assertTrue(result['regenerate_single'].episode_outline)


class SummaryBaselineReplayTests(PipelineReplayTests):
    baseline = True
