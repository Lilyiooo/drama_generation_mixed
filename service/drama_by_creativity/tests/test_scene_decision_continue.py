import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from drama_local.runtime import atomic_json
from service.drama_by_creativity import scene_state_gate as gate
from tools import continue_scene_gate_decision as continuation
from tools import scene_gate_decision_study as study


class SceneDecisionContinueTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.case = study.controlled_cases()[0]
        self.value = {'memory_conflicts': [], 'checks': [{'scene_id': '2-1', 'domain': 'life',
            'state_ids': ['M0001'], 'scene_span_ids': ['S0001'], 'state_basis': 'consistent',
            'scene_relation': 'contradicts', 'transition': 'not_shown', 'reason': '死亡与行动矛盾。'}], 'writing_notes': []}

    def prepare_study(self):
        source, r01, root = self.root / 'source', self.root / 'r01', self.root / 'continued'
        atomic_json(source / 'cases.json', [self.case])
        atomic_json(source / 'manifest.json', {'implementation': study.dependencies(),
                    'cases_sha256': gate.digest([self.case]), 'source_files': {}})
        for arm in study.ARMS:
            probe = study.DecisionProbe(source, self.case, 'R01', arm)
            response = {'status': 'pass', 'violations': [], 'reason': '离线结构测试。'} if arm == 'control_indexed' else self.value
            with patch.object(gate.LLM, 'request', AsyncMock(return_value=SimpleNamespace(response=json.dumps(response)))):
                asyncio.run(probe.run())
        with patch('builtins.print'):
            continuation.metadata.recover(source, r01, 'R01')
            continuation.prepare(source, r01, root)
        return source, r01, root

    def test_reuses_original_prompts_and_supports_metadata_in_online_response(self):
        probe = continuation.ContinuedProbe(self.root, self.case, 'R02', 'candidate_factorized')
        original = study.DecisionProbe(self.root, self.case, 'R02', 'candidate_factorized')
        self.assertEqual(probe.protocol()[0], original.protocol()[0])
        self.assertEqual(probe.packet(self.case['scenes']), original.packet(self.case['scenes']))
        self.value['checks'][0]['domain'] = 'character_state'
        result = probe.protocol()[1](json.dumps(self.value))
        self.assertEqual(result['status'], 'fail')
        self.assertEqual(result['checks'][0]['domain'], 'character_state')
        self.assertTrue(result['metadata_compatibility'])

    def test_memory_and_scene_reference_mixing_is_rejected_with_specific_feedback(self):
        self.value['memory_conflicts'] = [{'state_ids': ['M0001', 'S0001'], 'reason': '错误引用。'}]
        with self.assertRaisesRegex(ValueError, '不得引用S编号'):
            continuation.compatible_validate(json.dumps(self.value), self.case['scenes'], gate.state_catalog(self.case['view']))

    def test_r01_generation_is_forbidden_but_independent_review_is_allowed(self):
        with self.assertRaisesRegex(ValueError, 'must not be resampled'):
            continuation.ContinuedProbe(self.root, self.case, 'R01', 'candidate_factorized')
        judge = continuation.ContinuedProbe(self.root, self.case, 'R01', 'judge')
        self.assertEqual(judge.model, 'Qwen3.8-27B')

    def test_only_r02_r03_are_pending_and_r01_files_remain_unchanged(self):
        source, r01, root = self.prepare_study()
        before = {str(path): study.indexed.file_hash(path) for directory in (source, r01) for path in directory.rglob('*.json')}
        with patch('builtins.print'):
            summary = continuation.report(root)
        tasks = continuation.pending_tasks([self.case], summary, False)
        self.assertEqual(len(tasks), 4)
        self.assertEqual({run for case, run, arm in tasks}, {'R02', 'R03'})
        self.value['checks'][0]['domain'] = 'character_state'
        request = AsyncMock(return_value=SimpleNamespace(response=json.dumps(self.value)))
        with patch.object(gate.LLM, 'request', request):
            asyncio.run(continuation.run_one(root, self.case, 'R02', 'candidate_factorized'))
            asyncio.run(continuation.run_one(root, self.case, 'R02', 'candidate_factorized'))
        request.assert_awaited_once()
        with patch('builtins.print'):
            summary = continuation.report(root)
        self.assertEqual(len(continuation.pending_tasks([self.case], summary, False)), 3)
        self.assertEqual(len(continuation.pending_tasks([self.case], summary, True)), 3)
        self.assertEqual(before, {str(path): study.indexed.file_hash(path) for directory in (source, r01) for path in directory.rglob('*.json')})

    def test_refuses_duplicate_original_r02_attempts(self):
        source, r01, root = self.prepare_study()
        atomic_json(source / 'candidate_factorized/life_hard/R02/attempts/decision/test/1.json', {})
        with self.assertRaisesRegex(ValueError, 'already contain attempts'):
            continuation.prepare(source, r01, self.root / 'different')


if __name__ == '__main__':
    unittest.main()
