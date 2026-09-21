import asyncio
import copy
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from service.drama_by_creativity import stage_episode_planning as planning
from service.drama_by_creativity.generate_story_outline_and_role import transfer_outline_to_str
from service.drama_by_creativity.prompts import GENERATE_EPISODE_OUTLINE_BY_SCRIPT_PROMPT
import jinja2


def outline():
    return {'title': '测试故事', 'background': '原始背景', 'summary': '主题总结',
            'framework': [{'stage': s, 'event_list': [{'event': '专属事件' + s}]} for s in ['开端', '发展', '高潮', '结局']]}


def allocations():
    return [dict(stage=s['stage'], start_episode_id=2*i+1, end_episode_id=2*i+2,
                 pacing_reason='根据当前阶段冲突与情感安排') for i, s in enumerate(outline()['framework'])]


def episodes(start, end):
    return [dict(episode_id=i, title=f'标题{i}', core_plot=f'情节{i}', roles=['主角'],
                 highlights=['看点'], character_growth='成长', relationship_changes='关系',
                 main_storyline_progression='推进', ending_hook='收束') for i in range(start, end+1)]


class StagePlanningTests(unittest.TestCase):
    def test_extract_request_stages(self):
        expected = outline()['framework']
        self.assertEqual(planning.extract_stages(outline()), expected)
        self.assertEqual(planning.extract_stages(json.dumps(outline())), expected)
        self.assertEqual(planning.extract_stages(transfer_outline_to_str(outline())), expected)
        with self.assertRaises(ValueError):
            planning.extract_stages('没有阶段')

    def test_invalid_range_plans(self):
        good = allocations()
        planning.validate_stage_plan(good, outline()['framework'], 8)
        for key, value in [('start_episode_id', 4), ('start_episode_id', 2),
                           ('end_episode_id', 9), ('stage', '错误阶段'), ('start_episode_id', True)]:
            bad = copy.deepcopy(good)
            bad[1][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                planning.validate_stage_plan(bad, outline()['framework'], 8)
        bad = copy.deepcopy(good)
        bad[-1]['end_episode_id'] = 7
        with self.assertRaises(ValueError):
            planning.validate_stage_plan(bad, outline()['framework'], 8)

    def test_reject_missing_reordered_and_boolean_episode_ids(self):
        for data in [episodes(4, 8), episodes(2, 8)+episodes(1, 1),
                     [dict(episodes(1, 1)[0], episode_id=True)] + episodes(2, 8)]:
            with self.assertRaises(ValueError):
                planning.validate_episode_batch(data, 1, 8)

    def test_stage_scoping_previous_context_and_saved_resume_files(self):
        async def scenario(tmp):
            calls = []
            async def infer(**kwargs):
                params = kwargs['params']
                calls.append(params)
                if 'Original' in params:
                    self.assertEqual(json.loads(params['Original']), episodes(1, 8))
                    kwargs['llm_service'].template.render(**params)
                    if 'Review' in params:
                        return json.loads(params['Original'])
                    return {'total_score': 80, 'overall_assessment': '无需修改', 'issues': [], 'revision_strategy': '保持一致'}
                if 'FullOutline' in params:
                    self.assertIn('专属事件结局', params['FullOutline'])
                    kwargs['llm_service'].template.render(**params)
                    return allocations()
                index = len(calls) - 2
                stage = outline()['framework'][index]
                self.assertEqual(json.loads(params['Outline']), stage['event_list'])
                self.assertNotIn('FullOutline', params)
                self.assertEqual(params['Episode1Script'], '样稿1')
                self.assertEqual(json.loads(params['EpisodeOutline']), episodes(1, index*2))
                prompt = kwargs['llm_service'].template.render(**params)
                self.assertIn('专属事件' + stage['stage'], prompt)
                for other in outline()['framework']:
                    if other != stage:
                        self.assertNotIn('专属事件' + other['stage'], prompt)
                return episodes(int(params['StartEpisodeId']), int(params['EndEpisodeId']))
            common = dict(RoleDescription='编剧', Topic='轻喜', WorldView='架空', StoryOutline='核心创意',
                          Reference='', RoleInfo='人设', Episode1Script='样稿1', Episode2Script='样稿2', Episode3Script='样稿3')
            with patch.object(planning, 'llm_inference_json', side_effect=infer), patch.object(
                planning, 'make_outline_llm', side_effect=lambda text: SimpleNamespace(
                    template=jinja2.Environment(undefined=jinja2.StrictUndefined).from_string(text))):
                result = await planning.generate_stage_episode_outlines(
                    SimpleNamespace(), transfer_outline_to_str(outline()), 8, common, GENERATE_EPISODE_OUTLINE_BY_SCRIPT_PROMPT)
            self.assertEqual(result, episodes(1, 8))
            self.assertEqual(len(calls), 12)
            self.assertEqual(len(list((Path(tmp)/'04_episode_outline').glob('*.json'))), 4)
            self.assertTrue((Path(tmp)/'04_stage_plan/stage_plan.json').exists())
            # Check the existing resume loader accepts the unchanged chunk schema.
            with patch.dict(sys.modules, {'run_test': importlib.import_module('service.drama_by_creativity.tests.run_test')}):
                from service.drama_by_creativity.tests.run_from_episode_outline import load_episode_outlines
                self.assertEqual(load_episode_outlines(tmp), result)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'DRAMA_OUTPUT_DIR': ''}):
            os.environ['DRAMA_OUTPUT_DIR'] = tmp
            asyncio.run(scenario(tmp))

    def test_bad_plan_and_missing_first_episodes_stop(self):
        for fail_plan in [True, False]:
            with self.subTest(fail_plan=fail_plan), tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'DRAMA_OUTPUT_DIR': tmp}):
                async def infer(**kwargs):
                    if 'FullOutline' in kwargs['params']:
                        return [] if fail_plan else allocations()
                    return episodes(4, 4)
                with patch.object(planning, 'llm_inference_json', side_effect=infer) as call, patch.object(
                    planning, 'make_outline_llm', return_value=SimpleNamespace()):
                    with self.assertRaisesRegex(RuntimeError, '停止生成'):
                        asyncio.run(planning.generate_stage_episode_outlines(None, json.dumps(outline(), ensure_ascii=False),
                                                                             8, {}, GENERATE_EPISODE_OUTLINE_BY_SCRIPT_PROMPT))
                self.assertEqual(call.call_count, 3 if fail_plan else 4)
                self.assertFalse(list((Path(tmp)/'04_episode_outline').glob('*.json')))
                self.assertFalse((Path(tmp)/'04_stage_plan/completion.json').exists())
                self.assertTrue(call.call_args.kwargs['params']['RetryFeedback'])
