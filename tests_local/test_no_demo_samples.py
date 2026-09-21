"""The simple pipeline starts from the proposal and never needs sample scripts."""
import asyncio
import importlib
from contextlib import ExitStack
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
import unittest
from jinja2 import Environment, meta
from drama_local import models as pb
from drama_local.runtime import ScriptContentFile
from service.drama_by_creativity.tests import run_test
from service.drama_by_creativity import prompts
from service.drama_by_creativity.prompts.stage_episode_plan import GENERATE_STAGE_EPISODE_PLAN_PROMPT

class NoDemoSamplesTests(unittest.TestCase):
    def test_full_pipeline_order(self):
        calls = []
        names = ['step_1_script_proposal', 'step_3_story_outline', 'step_4_episode_outline',
                 'step_5_drama', 'step_6_regenerate_single_episode_outline']
        values = [NS(), NS(story_outline=NS()), NS(),
                  NS(result=NS(seasons=[NS(episodes=[NS(episode_id=i) for i in range(3)])])), NS()]
        with ExitStack() as stack:
            for name, value in zip(names, values):
                async def fake(*args, _name=name, _value=value, **kwargs):
                    calls.append(_name)
                    return _value
                stack.enter_context(patch.object(run_test, name, side_effect=fake))
            result = asyncio.run(run_test.run_full_pipeline(None, NS(), NS(episode_nums=3)))
        self.assertEqual(calls, names)
        self.assertNotIn('demo_drama', result)

    def test_prompts_have_no_sample_inputs(self):
        for template in [prompts.GENERATE_OUTLINE_BY_SCRIPT_PROMPT,
                         prompts.GENERATE_ALL_EPISODE_OUTLINE_PROMPT,
                         prompts.GENERATE_EPISODE_OUTLINE_BY_SCRIPT_PROMPT,
                         GENERATE_STAGE_EPISODE_PLAN_PROMPT]:
            variables = meta.find_undeclared_variables(Environment().parse(template))
            self.assertFalse(variables & {'Episode1Script', 'Episode2Script', 'Episode3Script'})
            self.assertNotIn('前三集', template)
            self.assertNotIn('样稿', template)

    def test_story_outline_uses_proposal_without_reading_scripts(self):
        module = importlib.import_module('service.drama_by_creativity.generate_story_outline_and_role')
        request = pb.GenerateStoryOutlineByCreativityReq(
            story_info=pb.StoryInfo(plot_type=pb.PlotType.SHORT_CARTOON),
            generate_input=pb.GenerateInputByCreativity(core_story='故事背景'),
            generate_data=pb.GenerateStoryOutline(script_proposal=pb.ScriptProposal()))
        class Captured(Exception): pass
        async def capture(**kwargs):
            self.assertIn('ScriptProposal', kwargs['params'])
            self.assertNotIn('Episode1Script', kwargs['params'])
            raise Captured()
        with patch.object(module, 'llm_inference_json', side_effect=capture), \
             patch.object(ScriptContentFile, 'download', new_callable=AsyncMock) as download:
            with self.assertRaises(Captured):
                asyncio.run(module.generate_story_outline_and_role(None, request))
            download.assert_not_called()

    def test_outline_only_stops_before_body(self):
        names = ['step_1_script_proposal', 'step_3_story_outline', 'step_4_episode_outline']
        with ExitStack() as stack:
            mocks = [stack.enter_context(patch.object(run_test, name, new_callable=AsyncMock)) for name in names]
            body = stack.enter_context(patch.object(run_test, 'step_5_drama', new_callable=AsyncMock))
            regenerate = stack.enter_context(patch.object(run_test, 'step_6_regenerate_single_episode_outline', new_callable=AsyncMock))
            result = asyncio.run(run_test.run_full_pipeline(None, NS(), NS(outline_only=True, episode_nums=60)))
            for mock in mocks:
                mock.assert_awaited_once()
            body.assert_not_called()
            regenerate.assert_not_called()
            self.assertEqual(set(result), {'script_proposal', 'story_outline', 'episode_outline'})
