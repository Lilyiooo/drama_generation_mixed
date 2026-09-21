import asyncio
import copy
import importlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from drama_local.runtime import Context, LocalStore, atomic_json
from service.drama_by_creativity import scene_state_gate as gate
from tools import scene_gate_calibration as calibration


class SceneGateTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.case = calibration.controlled_cases()[0]
        self.instance = calibration.make_gate(self.root, self.case, 'R01')
        self.scenes = self.case['scenes']
        self.fail = {'status': 'fail', 'violations': [{'category': 'life', 'state_ids': ['M0001'],
                     'scene_id': '2-1', 'scene_quote': '李舟当场推开密室门', 'reason': '已死亡仍行动'}], 'reason': '有冲突'}
        self.passed = {'status': 'pass', 'violations': [], 'reason': '没有证实冲突'}
        self.uncertain = {'status': 'uncertain', 'violations': [], 'reason': '状态先后无法判断'}

    def test_evidence_requires_real_state_id_and_quote_in_named_scene(self):
        self.assertEqual(gate.validate_check(json.dumps(self.fail), self.scenes, self.instance.catalog), self.fail)
        invalid = copy.deepcopy(self.fail)
        invalid['violations'][0]['state_ids'] = ['M9999']
        with self.assertRaisesRegex(ValueError, 'existing state_ids'):
            gate.validate_check(json.dumps(invalid), self.scenes, self.instance.catalog)
        invalid = copy.deepcopy(self.fail)
        invalid['violations'][0]['scene_quote'] = '李舟推开门...然后一起离开'
        with self.assertRaisesRegex(ValueError, 'continuous text'):
            gate.validate_check(json.dumps(invalid), self.scenes, self.instance.catalog)
        invalid['violations'][0]['scene_id'] = '2-9'
        with self.assertRaisesRegex(ValueError, 'unknown scene_id'):
            gate.validate_check(json.dumps(invalid), self.scenes, self.instance.catalog)

    def test_enforce_repairs_once_then_blocks_remaining_conflict(self):
        check = AsyncMock(side_effect=[self.fail, self.fail])
        request = AsyncMock(return_value=self.scenes)
        with patch.object(self.instance, 'check', check), patch.object(self.instance, 'request', request):
            with self.assertRaises(gate.SceneGateBlocked):
                asyncio.run(self.instance.apply(Context(), self.scenes))
        self.assertEqual(check.await_count, 2)
        request.assert_awaited_once()
        outcome = gate.read(self.instance.directory / 'outcome.json')
        self.assertFalse(outcome['allowed'])
        self.assertTrue(outcome['repair_attempted'])
        self.assertEqual(self.instance.saved_scenes(), self.scenes)

    def test_uncertain_blocks_without_inventing_a_repair(self):
        with patch.object(self.instance, 'check', AsyncMock(return_value=self.uncertain)), \
             patch.object(self.instance, 'request', AsyncMock()) as request:
            with self.assertRaises(gate.SceneGateBlocked):
                asyncio.run(self.instance.apply(Context(), self.scenes))
        request.assert_not_awaited()

    def test_audit_records_conflict_but_does_not_change_scene(self):
        self.instance.mode = 'audit'
        self.instance.binding['mode'] = 'audit'
        with patch.object(self.instance, 'check', AsyncMock(return_value=self.fail)), \
             patch.object(self.instance, 'request', AsyncMock()) as request:
            result = asyncio.run(self.instance.apply(Context(), self.scenes))
        self.assertEqual(result, self.scenes)
        request.assert_not_awaited()
        self.assertTrue(gate.read(self.instance.directory / 'outcome.json')['allowed'])

    def test_pass_does_not_repair_and_frozen_scenes_reject_drift(self):
        with patch.object(self.instance, 'check', AsyncMock(return_value=self.passed)):
            self.assertEqual(asyncio.run(self.instance.apply(Context(), self.scenes)), self.scenes)
        changed = copy.deepcopy(self.scenes)
        changed[0]['核心功能'] = '另一个任务'
        with self.assertRaisesRegex(ValueError, 'frozen original scenes'):
            asyncio.run(self.instance.apply(Context(), changed))
        self.instance.params = {**self.instance.params, 'Outline': '改了大纲'}
        other = gate.SceneStateGate(self.instance.directory, 2, self.instance.params, self.case['view'])
        with self.assertRaisesRegex(ValueError, 'inputs changed'):
            other.saved_scenes()

    def test_request_retry_feedback_and_cache_keep_original_inputs(self):
        invalid = copy.deepcopy(self.fail)
        invalid['violations'][0]['state_ids'] = ['M9999']
        request = AsyncMock(side_effect=[SimpleNamespace(response=json.dumps(invalid)),
                                        SimpleNamespace(response=json.dumps(self.fail))])
        with patch.object(gate.LLM, 'request', request):
            result = asyncio.run(self.instance.check(Context(), self.scenes))
            again = asyncio.run(self.instance.check(Context(), self.scenes))
        self.assertEqual(result, again)
        self.assertEqual(request.await_count, 2)
        retry = request.await_args_list[1].args[1]['Prompt']
        self.assertIn('existing state_ids', retry)
        self.assertIn('李舟在上一集明确死亡', retry)
        self.assertIn('李舟当场推开密室门', retry)
        self.assertEqual(len(list((self.instance.directory / 'attempts/check_before').glob('*/*.json'))), 2)

    def test_policy_default_off_and_existing_run_cannot_silently_switch(self):
        with patch.dict(os.environ, {'DRAMA_SCENE_STATE_GATE': 'off'}):
            self.assertEqual(gate.gate_mode(), 'off')
        gate.bind_policy(self.root, 'off')
        self.assertFalse((self.root / 'scene_state_gate_policy.json').exists())
        gate.bind_policy(self.root, 'enforce')
        with self.assertRaisesRegex(ValueError, 'mode or implementation changed'):
            gate.bind_policy(self.root, 'off')
        other = self.root / 'old'
        atomic_json(other / '05_drama/episode_01.json', {'content': '旧正文'})
        with self.assertRaisesRegex(ValueError, 'midway'):
            gate.bind_policy(other, 'enforce')

    def test_controlled_probe_labels_are_not_sent_to_model(self):
        cases = calibration.controlled_cases()
        self.assertEqual(len(cases), 12)
        self.assertEqual({case['category'] for case in cases}, {'life', 'object', 'resource', 'knowledge'})
        for case in cases:
            instance = calibration.make_gate(self.root, case, 'R01')
            packet = instance.packet(case['scenes'])
            self.assertNotIn('expected', packet)
            self.assertNotIn('origin', packet)
            self.assertNotIn('case', packet)

    def test_calibration_counts_blocked_as_completed_and_checks_judge_cache(self):
        case = calibration.controlled_cases()[2]
        instance = calibration.make_gate(self.root, case, 'R01')
        request = AsyncMock(return_value=SimpleNamespace(response=json.dumps(self.uncertain)))
        with patch.object(gate.LLM, 'request', request):
            asyncio.run(calibration.run_case(self.root, case, 'R01', False))
            self.assertFalse(calibration.read_outcome(instance)['allowed'])
            asyncio.run(calibration.run_case(self.root, case, 'R01', True))
            asyncio.run(calibration.run_case(self.root, case, 'R01', True))
        self.assertEqual(request.await_count, 3)
        manifest = {'scope': 'offline test', 'cases_sha256': gate.digest([case]),
                    'implementation': calibration.dependencies()}
        atomic_json(self.root / 'manifest.json', manifest)
        atomic_json(self.root / 'cases.json', [case])
        with patch('builtins.print'):
            row = calibration.report(self.root)['results'][0]
        self.assertTrue(row['complete'])
        self.assertTrue(row['scored'])
        self.assertFalse(row['allowed'])
        path = instance.directory / 'judge_after.json'
        cached = gate.read(path)
        cached['request']['model'] = 'wrong-model'
        atomic_json(path, cached)
        with self.assertRaisesRegex(ValueError, 'Judge input changed'):
            calibration.report(self.root)


class SceneGateIntegrationTests(unittest.TestCase):
    def setUp(self):
        from service.drama_by_creativity.tests.test_state_lifecycle_memory import StateLifecycleTest
        self.fixture = StateLifecycleTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.generator = importlib.import_module('service.drama_by_creativity.generate_episode_script')
        from tools import state_hybrid_study
        self.study = state_hybrid_study
        self.experiment = self.fixture.prepare_study()
        self.study.configure(self.experiment / 'hybrid', 'hybrid')
        self.environment = patch.dict(os.environ, {'DRAMA_SCENE_STATE_GATE': 'enforce'})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.scenes = [calibration.scene('李舟当场行动')]
        self.scenes[0]['场次编号'] = '1-1'

    def invoke(self, apply):
        from service.drama_by_creativity.state_lifecycle_memory import StateLifecycleMemory
        request = AsyncMock(return_value=SimpleNamespace(response='第1集正式正文'))
        extraction = AsyncMock(return_value=self.fixture.extraction())
        with patch.object(self.generator, 'scene_outline_inference_json', AsyncMock(return_value=self.scenes)), \
             patch.object(gate.SceneStateGate, 'apply', apply), patch.object(self.generator.LLM, 'request', request), \
             patch.object(self.generator.NarrativeMemory, 'update_from_episode', AsyncMock(return_value={
                 'future_driving_nodes': [], 'future_map_edges': [], 'world_state_updates': []})), \
             patch.object(StateLifecycleMemory, 'extract_initial_state', AsyncMock(return_value=self.fixture.initial_payload())), \
             patch.object(StateLifecycleMemory, 'extract', extraction):
            try:
                asyncio.run(self.generator.generate_episode_script(Context(), self.study.make_request(
                    self.study.load_inputs(self.experiment), 'hybrid', 0), LocalStore()))
            except gate.SceneGateBlocked:
                return request, extraction, True
        return request, extraction, False

    def test_repaired_scenes_reach_whole_episode_writer(self):
        repaired = copy.deepcopy(self.scenes)
        repaired[0]['主要情节'] = ['修复后由周岚展示遗留证据']
        request, extraction, blocked = self.invoke(AsyncMock(return_value=repaired))
        self.assertFalse(blocked)
        self.assertIn('修复后由周岚', request.await_args.args[1]['AllSceneOutlines'])
        extraction.assert_awaited_once()

    def test_blocked_gate_prevents_script_and_memory_update(self):
        request, extraction, blocked = self.invoke(AsyncMock(side_effect=gate.SceneGateBlocked('blocked')))
        self.assertTrue(blocked)
        request.assert_not_awaited()
        extraction.assert_not_awaited()
        self.assertFalse((self.experiment / 'hybrid/05_drama/episode_01.json').exists())


if __name__ == '__main__':
    unittest.main()
