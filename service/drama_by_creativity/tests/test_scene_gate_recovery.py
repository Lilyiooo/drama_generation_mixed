import asyncio
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from drama_local.runtime import Context, atomic_json
from tools import repair_scene_gate_calibration as recovery
from tools import scene_gate_calibration as original
from service.drama_by_creativity import scene_state_gate as gate


class SceneGateRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.case = original.controlled_cases()[0]
        self.instance = recovery.make_gate(self.root, self.case, 'R01')
        self.fail = {'status': 'fail', 'violations': [{'category': 'life', 'state_ids': ['M0001'],
                     'scene_span_id': 'S0001', 'reason': '已死亡仍行动'}], 'reason': '存在冲突'}
        self.passed = {'status': 'pass', 'violations': [], 'reason': '没有证实冲突'}
        self.uncertain = {'status': 'uncertain', 'violations': [], 'reason': '状态互相矛盾'}

    def prepare_source(self):
        source, root = self.root / 'source', self.root / 'recovery'
        cases = original.controlled_cases()[:2]
        atomic_json(source / 'cases.json', cases)
        atomic_json(source / 'manifest.json', {'cases_sha256': gate.digest(cases), 'implementation': original.dependencies()})
        with patch.object(gate.LLM, 'request', AsyncMock(return_value=SimpleNamespace(response=json.dumps(self.passed)))):
            for run in original.RUNS:
                instance = original.make_gate(source, cases[1], run)
                asyncio.run(instance.apply(Context(), cases[1]['scenes']))
        recovery.prepare(source, root)
        return source, root

    def test_id_resolution_preserves_exact_text_and_original_scene(self):
        scenes = copy.deepcopy(self.case['scenes'])
        scenes.append({**scenes[0], '场次编号': '2-2'})
        spans = recovery.evidence_catalog(scenes)
        span_id = next(key for key, value in spans.items() if value['scene_id'] == '2-2')
        raw = copy.deepcopy(self.fail)
        raw['violations'][0]['scene_span_id'] = span_id
        checked = recovery.validate_indexed(json.dumps(raw), scenes, self.instance.catalog)
        self.assertEqual(checked['violations'][0]['scene_quote'], scenes[1]['情节概要'])
        self.assertEqual(checked['violations'][0]['scene_id'], '2-2')
        self.assertTrue(all(span['field'] not in {'出场人物', '人物动机与目标'} for span in spans.values()))
        packet = self.instance.packet(scenes)
        self.assertNotIn('expected', packet)
        self.assertNotIn('origin', packet)
        self.assertEqual(packet['candidate_scenes'], scenes)

    def test_unknown_ids_and_fabricated_quotes_are_rejected(self):
        for key, value in [('scene_span_id', 'S9999'), ('state_ids', ['M9999']),
                           ('scene_quote', '出场人物：...李舟'), ('scene_id', '2-9')]:
            raw = copy.deepcopy(self.fail)
            raw['violations'][0][key] = value
            with self.assertRaises(ValueError):
                recovery.validate_indexed(json.dumps(raw), self.case['scenes'], self.instance.catalog)
        legacy = copy.deepcopy(self.fail)
        legacy['violations'][0].pop('scene_span_id')
        with self.assertRaisesRegex(ValueError, 'scene_span_id'):
            recovery.validate_indexed(json.dumps(legacy), self.case['scenes'], self.instance.catalog)

    def test_retry_and_cache_use_indexed_protocol_without_changing_verdict(self):
        invalid = copy.deepcopy(self.fail)
        invalid['violations'][0]['scene_span_id'] = 'S9999'
        request = AsyncMock(side_effect=[SimpleNamespace(response=json.dumps(invalid)),
                                        SimpleNamespace(response=json.dumps(self.fail))])
        with patch.object(gate.LLM, 'request', request):
            checked = asyncio.run(self.instance.check(Context(), self.case['scenes']))
            cached = asyncio.run(self.instance.check(Context(), self.case['scenes']))
        self.assertEqual(request.await_count, 2)
        self.assertEqual(checked, cached)
        self.assertEqual(checked['status'], 'fail')
        self.assertIn('scene_span_id', request.await_args_list[1].args[1]['Prompt'])
        self.assertEqual(self.instance.verified_check('check_before', self.case['scenes'], 'Qwen3.6-27B'), checked)

    def test_recovery_retains_successes_and_counts_semantic_blocks_as_complete(self):
        source, root = self.prepare_source()
        before = {str(path): recovery.file_hash(path) for path in source.rglob('*.json')}
        manifest, cases = recovery.load(root)
        pending = [task for task in manifest['tasks'] if not task['source_complete']]
        self.assertEqual(len(pending), 3)
        request = AsyncMock(return_value=SimpleNamespace(response=json.dumps(self.uncertain)))
        with patch.object(gate.LLM, 'request', request):
            for task in pending:
                asyncio.run(recovery.run_task(root, manifest, cases[0], task, False))
            asyncio.run(recovery.run_task(root, manifest, cases[0], pending[0], False))
            self.assertEqual(request.await_count, 3)
            asyncio.run(recovery.run_task(root, manifest, cases[0], pending[0], True))
            asyncio.run(recovery.run_task(root, manifest, cases[0], pending[0], True))
            self.assertEqual(request.await_count, 5)
        with patch('builtins.print'):
            rows = recovery.report(root)['results']
        self.assertEqual(sum(row['complete'] for row in rows), 6)
        self.assertEqual(sum(row['scored'] for row in rows), 1)
        self.assertEqual(sum(row['source_complete'] for row in rows), 3)
        self.assertFalse(rows[0]['allowed'])
        self.assertEqual(before, {str(path): recovery.file_hash(path) for path in source.rglob('*.json')})

    def test_source_result_drift_is_not_silently_accepted(self):
        source, root = self.prepare_source()
        path = next(source.glob('cases/*/*/outcome.json'))
        value = gate.read(path)
        value['allowed'] = False
        atomic_json(path, value)
        with self.assertRaisesRegex(ValueError, 'Frozen original result changed'):
            recovery.load(root)

    def test_repair_is_checked_and_not_resampled_on_resume(self):
        repaired = copy.deepcopy(self.case['scenes'])
        repaired[0]['情节概要'] = '周岚读取李舟生前留下的笔记。'
        repaired[0]['主要情节'] = [repaired[0]['情节概要']]
        request = AsyncMock(side_effect=[SimpleNamespace(response=json.dumps(value))
                                        for value in (self.fail, repaired, self.passed)])
        with patch.object(gate.LLM, 'request', request):
            result = asyncio.run(self.instance.apply(Context(), self.case['scenes']))
            again = asyncio.run(self.instance.apply(Context(), self.case['scenes']))
        self.assertEqual(result, repaired)
        self.assertEqual(result, again)
        self.assertEqual(request.await_count, 3)
        outcome = self.instance.read_outcome()
        self.assertTrue(outcome['allowed'])
        self.assertTrue(outcome['repair_attempted'])
        self.assertEqual(outcome['before']['status'], 'fail')
        self.assertEqual(outcome['after']['status'], 'pass')

    def test_refuses_old_pending_task_completed_after_freeze(self):
        source, root = self.prepare_source()
        atomic_json(source / 'cases/life_conflict/R01/outcome.json', {})
        with self.assertRaisesRegex(ValueError, 'Original pending task was rerun'):
            recovery.load(root)


if __name__ == '__main__':
    unittest.main()
