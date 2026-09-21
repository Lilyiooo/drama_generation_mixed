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
from service.drama_by_creativity import conservative_scene_gate as gate
from service.drama_by_creativity import scene_state_gate as original
from service.drama_by_creativity.tests.test_state_lifecycle_memory import StateLifecycleTest
from tools import full_scene_gate_study as study
from tools import scene_gate_decision_study as fixtures


class ConservativeGateTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        case = fixtures.controlled_cases()[0]
        self.scenes = case['scenes']
        self.instance = gate.ConservativeSceneGate(self.root, case['episode'], case['params'], case['view'])
        self.hard = {'status': 'fail', 'checks': [{'classification': 'hard_conflict'}]}
        self.passed = {'status': 'pass', 'checks': []}
        self.uncertain = {'status': 'uncertain', 'checks': []}
        self.changed = copy.deepcopy(self.scenes)
        self.changed[0]['主要情节'] = ['周岚展示遗留证据']

    def test_pass_and_uncertainty_keep_original_without_repair(self):
        for result in (self.passed, self.uncertain):
            with patch.object(self.instance, 'check', AsyncMock(return_value=result)), \
                 patch.object(self.instance, 'request', AsyncMock()) as request:
                self.assertEqual(asyncio.run(self.instance.apply(Context(), self.scenes)), self.scenes)
                request.assert_not_awaited()
            self.assertEqual(original.read(self.root / 'outcome.json')['unresolved'], result == self.uncertain)

    def test_one_repair_accepted_only_on_pass(self):
        for after in (self.passed, self.hard, self.uncertain):
            with patch.object(self.instance, 'check', AsyncMock(side_effect=[self.hard, after])) as check, \
                 patch.object(self.instance, 'request', AsyncMock(return_value=self.changed)) as request:
                selected = asyncio.run(self.instance.apply(Context(), self.scenes))
            self.assertEqual(check.await_count, 2)
            request.assert_awaited_once()
            self.assertEqual(selected, self.changed if after == self.passed else self.scenes)
            outcome = original.read(self.root / 'outcome.json')
            self.assertEqual(outcome['repair_accepted'], after == self.passed)
            self.assertEqual(outcome['fallback_to_original'], after != self.passed)
            self.assertEqual(self.instance.saved_scenes(), self.scenes)

    def test_api_or_schema_error_never_falls_back(self):
        with patch.object(self.instance, 'check', AsyncMock(side_effect=RuntimeError('API failed'))):
            with self.assertRaisesRegex(RuntimeError, 'API failed'):
                asyncio.run(self.instance.apply(Context(), self.scenes))
        self.assertFalse((self.root / 'outcome.json').exists())
        with patch.object(self.instance, 'check', AsyncMock(side_effect=[self.hard, RuntimeError('schema')])), \
             patch.object(self.instance, 'request', AsyncMock(return_value=self.changed)):
            with self.assertRaisesRegex(RuntimeError, 'schema'):
                asyncio.run(self.instance.apply(Context(), self.scenes))
        self.assertFalse((self.root / 'outcome.json').exists())

    def test_frozen_original_rejects_drift(self):
        with patch.object(self.instance, 'check', AsyncMock(return_value=self.passed)):
            asyncio.run(self.instance.apply(Context(), self.scenes))
            with self.assertRaisesRegex(ValueError, 'frozen original'):
                asyncio.run(self.instance.apply(Context(), self.changed))

    def test_real_checker_and_request_cache_resume_without_resampling(self):
        def payload(relation, transition):
            return {'memory_conflicts': [], 'checks': [{
                'scene_id': '2-1', 'domain': 'life', 'state_ids': ['M0001'], 'scene_span_ids': ['S0001'],
                'state_basis': 'consistent', 'scene_relation': relation, 'transition': transition,
                'reason': '根据引用的状态和场内行动判断。'}], 'writing_notes': []}
        responses = [payload('contradicts', 'not_shown'), self.changed, payload('compatible', 'not_applicable')]
        mock = AsyncMock(side_effect=[SimpleNamespace(response=json.dumps(value, ensure_ascii=False)) for value in responses])
        with patch.object(original.LLM, 'request', mock):
            self.assertEqual(asyncio.run(self.instance.apply(Context(), self.scenes)), self.changed)
            self.assertEqual(asyncio.run(self.instance.apply(Context(), self.instance.saved_scenes())), self.changed)
        self.assertEqual(mock.await_count, 3)
        for slot in ('check_before', 'repair', 'check_after'):
            self.assertEqual(original.read(self.root / f'{slot}.json')['request']['model'], 'Qwen3.6-27B')


class FullStudyTests(unittest.TestCase):
    def setUp(self):
        self.fixture = StateLifecycleTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root / 'full'
        previous = self.fixture.prepare_study()
        self.source = previous / 'hybrid'
        inputs = study.study.load_inputs(previous)
        assets = study.study.assets_from_request(study.study.make_request(inputs, 'hybrid', 0, 'source'))
        atomic_json(self.source / 'generation_assets.json', assets)
        memory = study.study.StateLifecycleMemory('source', self.source)
        with patch.object(memory, 'extract_initial_state', AsyncMock(return_value=self.fixture.initial_payload())):
            asyncio.run(memory.initialize(Context(), assets))
        atomic_json(self.source / '05_drama/episode_01.json', {'episode_id': 0, 'content': '旧剧本不得复制'})
        atomic_json(self.source / 'narrative_memory/source/memory_state.json', {'old': True})
        self.hash_patch = patch.object(study, 'code_hashes', return_value={'frozen': 'hash'})
        self.hash_patch.start()
        self.addCleanup(self.hash_patch.stop)
        study.prepare(self.source, self.root, 'source')

    def test_preparation_shares_only_static_assets_and_opening_state(self):
        manifest, bundle = study.load(self.root)
        self.assertEqual(manifest['default_runs'], ['R01'])
        for arm in study.ARMS:
            for run in study.RUNS:
                directory = study.verify_directory(self.root, arm, run, manifest, bundle)
                memory = study.study.StateLifecycleMemory(study.identifier(arm, run), directory)
                self.assertEqual(memory.initial_state, bundle['initial']['initial_state'])
                self.assertEqual(memory.records, [])
                self.assertFalse((directory / '05_drama').exists())
                self.assertFalse((directory / 'narrative_memory').exists())
        study.prepare(self.source, self.root, 'source')

    def test_mutated_assets_or_code_rejected(self):
        with patch.object(study, 'code_hashes', return_value={'frozen': 'changed'}):
            with self.assertRaisesRegex(ValueError, 'implementation changed'):
                study.load(self.root)
        manifest, bundle = study.load(self.root)
        path = self.root / 'control_hybrid/R01/generation_assets.json'
        atomic_json(path, {})
        with self.assertRaisesRegex(ValueError, 'assets changed'):
            study.verify_directory(self.root, 'control_hybrid', 'R01', manifest, bundle)

    def test_control_has_no_hook_candidate_has_hook_in_separate_worker(self):
        with patch.object(study.study, 'generate_directory', AsyncMock()) as generate:
            study.run_worker(self.root, 'control_hybrid', 'R01', 'generate', 2)
            self.assertIsNone(generate.await_args.kwargs['scene_gate_factory'])
            study.run_worker(self.root, 'candidate_gate', 'R01', 'generate', 2)
            self.assertEqual(generate.await_args.kwargs['scene_gate_factory'], gate.ConservativeSceneGate.factory)
        self.assertEqual(os.environ['DRAMA_SCENE_STATE_GATE'], 'off')

    def test_dispatch_uses_subprocesses_and_only_selected_run(self):
        mock = unittest.mock.Mock(return_value=SimpleNamespace(returncode=0))
        with patch.object(study.subprocess, 'run', mock):
            study.dispatch(self.root, ('R01',), 'generate', 2, 2)
        self.assertEqual(mock.call_count, 2)
        for call in mock.call_args_list:
            command = call.args[0]
            self.assertIn('--worker', command)
            self.assertEqual(command[command.index('--run') + 1], 'R01')
            self.assertNotIn('--all-runs', command)

    def test_evaluation_waits_for_complete_scripts_and_memories(self):
        with patch.object(study.subprocess, 'run') as launch:
            with self.assertRaisesRegex(ValueError, 'Finish generation'):
                study.dispatch(self.root, ('R01',), 'evaluate', 60, 2)
            launch.assert_not_called()

    def test_scores_bind_current_script_and_dimensions(self):
        directory = self.root / 'control_hybrid/R01'
        for index in range(60):
            atomic_json(directory / '05_drama' / f'episode_{index + 1:02d}.json',
                        {'episode_id': index, 'content': f'正文{index}'})
        from tools.evaluate_with_drama_evaluator import export_complete_script
        _, checksum = export_complete_script(directory, 60)
        output = directory / 'drama_evaluations_qwen38/full_60_episodes'
        protocol = {'model': 'Qwen3.8-27B', 'temperature': '0', 'enable_thinking': 'false', 'script_sha256': checksum}
        atomic_json(output / 'local_protocol.json', protocol)
        atomic_json(output / 'scores.json', {'model': 'Qwen3.8-27B',
                    'scores': {name: {'fusion_50_50': 80} for name in study.DIMENSIONS}})
        self.assertEqual(study.scores(directory), {name: 80 for name in study.DIMENSIONS})
        atomic_json(directory / '05_drama/episode_01.json', {'episode_id': 0, 'content': '修改正文'})
        with self.assertRaisesRegex(ValueError, 'no longer matches'):
            study.scores(directory)

    def test_injected_gate_reaches_body_and_actual_script_state_update(self):
        generator = importlib.import_module('service.drama_by_creativity.generate_episode_script')
        directory = self.root / 'candidate_gate/R01'
        study.study.configure(directory, 'hybrid')
        manifest, bundle = study.load(self.root)
        scenes = copy.deepcopy(fixtures.controlled_cases()[0]['scenes'])
        scenes[0]['场次编号'] = '1-1'
        repaired = copy.deepcopy(scenes)
        repaired[0]['主要情节'] = ['修订后由周岚展示遗留证据']
        instance = SimpleNamespace(saved_scenes=lambda: scenes, apply=AsyncMock(return_value=repaired))
        factory = unittest.mock.Mock(return_value=instance)
        body = AsyncMock(return_value=SimpleNamespace(response='本集最终正式剧本'))
        extraction = AsyncMock(return_value=self.fixture.extraction())
        with patch.dict(os.environ, {'DRAMA_SCENE_STATE_GATE': 'off'}), \
             patch.object(generator, 'scene_outline_inference_json', AsyncMock()) as scene_request, \
             patch.object(generator.LLM, 'request', body), \
             patch.object(generator.NarrativeMemory, 'update_from_episode', AsyncMock(return_value={
                 'future_driving_nodes': [], 'future_map_edges': [], 'world_state_updates': []})), \
             patch.object(study.study.StateLifecycleMemory, 'extract', extraction):
            asyncio.run(generator.generate_episode_script(Context(), study.study.make_request(
                bundle['inputs'], 'hybrid', 0, study.identifier('candidate_gate', 'R01')), LocalStore(), scene_gate_factory=factory))
        scene_request.assert_not_awaited()
        self.assertIn('修订后由周岚', body.await_args.args[1]['AllSceneOutlines'])
        self.assertIn('本集最终正式剧本', extraction.await_args.args)
        factory.assert_called_once()


if __name__ == '__main__':
    unittest.main()
