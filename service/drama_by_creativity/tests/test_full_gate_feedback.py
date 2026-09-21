import asyncio
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from drama_local.runtime import Context, atomic_json
from service.drama_by_creativity import scene_state_gate as original
from tools import recover_full_scene_gate as recovery
from tools import scene_gate_decision_study as fixtures


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        case = fixtures.controlled_cases()[0]
        self.scenes = case['scenes']
        view = copy.deepcopy(case['view'])
        view['character_state'].append('李舟在同一时点活着。')
        self.gate = recovery.FeedbackGate(self.root, case['episode'], case['params'], view)
        self.valid = {'memory_conflicts': [{'state_ids': ['M0001', 'M0002'], 'reason': '同一人物同一时点生死矛盾。'}],
                      'checks': [{'scene_id': '2-1', 'domain': 'life', 'state_ids': ['M0001', 'M0002'],
                                  'scene_span_ids': ['S0001'], 'state_basis': 'conflicting',
                                  'scene_relation': 'unclear', 'transition': 'not_applicable', 'reason': '历史记忆互斥。'}],
                      'writing_notes': []}
        self.invalid = copy.deepcopy(self.valid)
        self.invalid['checks'][0]['state_ids'] = ['M0001']

    def test_invalid_reference_stays_invalid_and_feedback_is_specific(self):
        raw = json.dumps(self.invalid)
        with self.assertRaises(ValueError) as caught:
            recovery.validate_with_feedback(raw, self.scenes, self.gate.catalog)
        message = str(caught.exception)
        self.assertIn('declared_memory_groups', message)
        self.assertIn('conflicting_checks', message)
        self.assertIn('M0002', message)
        self.assertIn('一股脑', message)
        self.assertEqual(json.loads(raw), self.invalid)

    def test_valid_results_exactly_equal_original_validator(self):
        raw = json.dumps(self.valid)
        self.assertEqual(recovery.validate_with_feedback(raw, self.scenes, self.gate.catalog),
                         recovery.compatible_validate(raw, self.scenes, self.gate.catalog))
        self.assertEqual(recovery.validate_with_feedback(raw, self.scenes, self.gate.catalog)['status'], 'uncertain')

    def test_online_retry_receives_invalid_response_and_group_diagnostic(self):
        mock = AsyncMock(side_effect=[SimpleNamespace(response=json.dumps(value)) for value in (self.invalid, self.valid)])
        with patch.object(original.LLM, 'request', mock):
            result = asyncio.run(self.gate.check(Context(), self.scenes))
        self.assertEqual(result['status'], 'uncertain')
        self.assertEqual(mock.await_count, 2)
        first, second = [call.args[1]['Prompt'] for call in mock.await_args_list]
        self.assertNotIn('previous_invalid_response_excerpt', first)
        self.assertIn('previous_invalid_response_excerpt', second)
        self.assertIn('conflicting_checks', second)
        with patch.object(original.LLM, 'request', AsyncMock()) as request:
            self.assertEqual(asyncio.run(self.gate.check(Context(), self.scenes)), result)
            request.assert_not_awaited()

    def test_exhausted_invalid_checks_still_block_no_outcome(self):
        mock = AsyncMock(return_value=SimpleNamespace(response=json.dumps(self.invalid)))
        with patch.object(original.LLM, 'request', mock):
            with self.assertRaisesRegex(RuntimeError, 'retries exhausted'):
                asyncio.run(self.gate.apply(Context(), self.scenes))
        self.assertEqual(mock.await_count, 3)
        self.assertFalse((self.root / 'outcome.json').exists())

    def test_original_success_cache_is_reused_without_recheck(self):
        old = recovery.ConservativeSceneGate(self.root, self.gate.episode, self.gate.params,
                                             {field: [item['text'] for item in self.gate.catalog.values() if item['field'] == field]
                                              for field in original.FIELDS})
        with patch.object(original.LLM, 'request', AsyncMock(return_value=SimpleNamespace(response=json.dumps(self.valid)))):
            expected = asyncio.run(old.check(Context(), self.scenes))
        checksum = recovery.file_hash(self.root / 'check_before.json')
        with patch.object(original.LLM, 'request', AsyncMock()) as request:
            self.assertEqual(asyncio.run(self.gate.check(Context(), self.scenes)), expected)
            request.assert_not_awaited()
        self.assertEqual(checksum, recovery.file_hash(self.root / 'check_before.json'))

    def test_real_e09_failure_is_seeded_without_rewriting_source(self):
        source = recovery.study.ROOT / 'candidate_gate/R01/scene_state_gate/full-gate-candidate_gate-R01/E09'
        frozen = original.read(source / 'input.json')
        binding = frozen['binding']
        view = {field: [item['text'] for item in binding['catalog'].values() if item['field'] == field]
                for field in original.FIELDS}
        instance = recovery.FeedbackGate(self.root / 'real', 9, binding['params'], view)
        self.assertEqual(instance.binding, binding)
        atomic_json(instance.directory / 'input.json', frozen)
        sources = sorted((source / 'attempts/check_before').glob('*/*.json'))
        self.assertEqual(len(sources), 3)
        for path in sources:
            atomic_json(instance.directory / 'attempts/check_before/original' / path.name, original.read(path))
        retained = {path: recovery.file_hash(path) for path in instance.directory.rglob('*.json')}
        with patch.object(instance, 'request', AsyncMock(return_value={'status': 'uncertain'})) as request:
            asyncio.run(instance.check(Context(), instance.saved_scenes()))
        prompt = request.await_args.args[3]
        self.assertIn('previous_invalid_response_excerpt', prompt)
        self.assertIn('M0107', prompt)
        self.assertEqual(request.await_args.args[1], 'check_before_check_feedback_v1')
        for path, checksum in retained.items():
            self.assertEqual(recovery.file_hash(path), checksum)

    def test_memory_uncertainty_keeps_scene_without_repair(self):
        with patch.object(original.LLM, 'request', AsyncMock(return_value=SimpleNamespace(response=json.dumps(self.valid)))) as request:
            selected = asyncio.run(self.gate.apply(Context(), self.scenes))
        self.assertEqual(selected, self.scenes)
        request.assert_awaited_once()
        outcome = original.read(self.root / 'outcome.json')
        self.assertTrue(outcome['unresolved'])
        self.assertFalse(outcome['repair_attempted'])
        self.assertEqual(outcome['recovery_protocol'], recovery.VERSION)

    def test_bad_json_receives_feedback_but_is_not_accepted(self):
        with self.assertRaisesRegex(ValueError, 'previous_invalid_response_excerpt'):
            recovery.validate_with_feedback('{broken', self.scenes, self.gate.catalog)


class ProfileTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        atomic_json(self.root / 'manifest.json', {'original': True})
        atomic_json(self.root / 'frozen_inputs.json', {'original': True})
        self.rows = [{'arm': 'control_hybrid', 'run': 'R01', 'complete_episodes': 60},
                     {'arm': 'candidate_gate', 'run': 'R01', 'complete_episodes': 8}]
        for arm in recovery.study.ARMS:
            for filename in ('arm_binding.json', 'generation_assets.json', '05_drama/episode_01.json',
                             'narrative_memory/story/memory_state.json'):
                atomic_json(self.root / arm / 'R01' / filename, {'original': True})
        for target, replacement in (
            ('load', Mock(return_value=({}, {}))), ('report', Mock(return_value=self.rows)),
            ('verify_directory', Mock(side_effect=lambda root, arm, run, *unused: root / arm / run))):
            patcher = patch.object(recovery.study, target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        recovery.prepare(self.root, 'R01')

    def test_prepare_is_idempotent_and_retained_edits_rejected(self):
        profile = recovery.load(self.root, 'R01')
        recovery.prepare(self.root, 'R01')
        self.assertEqual(profile, recovery.load(self.root, 'R01'))
        atomic_json(self.root / 'candidate_gate/R01/05_drama/episode_01.json', {'changed': True})
        with self.assertRaisesRegex(ValueError, 'Retained original artifact changed'):
            recovery.load(self.root, 'R01')

    def test_mutable_native_memory_is_snapshotted_not_frozen(self):
        atomic_json(self.root / 'candidate_gate/R01/narrative_memory/story/memory_state.json', {'updated': True})
        recovery.load(self.root, 'R01')
        snapshot = recovery.profile_dir(self.root, 'R01') / 'before/candidate_gate/narrative_memory/story/memory_state.json'
        self.assertEqual(original.read(snapshot), {'original': True})

    def test_dispatch_skips_completed_control_and_calls_new_worker(self):
        finished = [{**row, 'complete_episodes': 60} for row in self.rows]
        with patch.object(recovery.study, 'report', side_effect=[self.rows, finished]), \
             patch.object(recovery.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as request:
            recovery.generate(self.root, 'R01', 60, 2)
        request.assert_called_once()
        command = request.call_args.args[0]
        self.assertIn('recover_full_scene_gate.py', command[1])
        self.assertEqual(command[command.index('--arm') + 1], 'candidate_gate')
        self.assertIn('--worker', command)

    def test_recovery_implementation_change_is_rejected(self):
        with patch.object(recovery, 'dependencies', return_value={'changed': 'implementation'}):
            with self.assertRaisesRegex(ValueError, 'Recovery code'):
                recovery.load(self.root, 'R01')


if __name__ == '__main__':
    unittest.main()
