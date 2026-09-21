import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from drama_local.runtime import atomic_json
from tools import recover_scene_decision_metadata as recovery
from tools import scene_gate_decision_study as study
from service.drama_by_creativity import scene_state_gate as gate


class SceneDecisionMetadataTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.case = study.controlled_cases()[0]
        self.probe = study.DecisionProbe(self.root, self.case, 'R01', 'candidate_factorized')
        self.value = {'memory_conflicts': [], 'checks': [{
            'scene_id': '2-1', 'domain': 'character_state', 'state_ids': ['M0001'],
            'scene_span_ids': ['S0001'], 'state_basis': 'consistent', 'scene_relation': 'contradicts',
            'transition': 'not_shown', 'reason': '死亡与现实行动矛盾。'}], 'writing_notes': []}

    def envelope(self, value=None):
        return {'request': {'model': 'Qwen3.6-27B', 'binding': self.probe.binding,
                            'system': self.probe.protocol()[0],
                            'user': json.dumps(self.probe.packet(self.probe.scenes), ensure_ascii=False),
                            'temperature': 0, 'max_tokens': 4096},
                'raw': json.dumps(value or self.value, ensure_ascii=False),
                'error': 'checks[0] has invalid scene or domain'}

    def test_metadata_compatibility_preserves_fields_and_status(self):
        original = copy.deepcopy(self.value)
        result, changes = recovery.compatible_result(json.dumps(original), self.probe.scenes, self.probe.catalog)
        self.assertEqual(original, self.value)
        self.assertEqual(result['status'], 'fail')
        self.assertEqual(result['checks'][0]['domain'], 'character_state')
        self.assertEqual(result['checks'][0]['domain_namespace'], 'state_field')
        for key, value in original['checks'][0].items():
            self.assertEqual(result['checks'][0][key], value)
        self.assertEqual(len(changes), 1)

    def test_only_known_metadata_names_accepted(self):
        invalid = copy.deepcopy(self.value)
        invalid['checks'][0]['domain'] = 'made_up_domain'
        with self.assertRaises(ValueError):
            recovery.compatible_result(json.dumps(invalid), self.probe.scenes, self.probe.catalog)

    def test_does_not_fix_evidence_or_change_conflicting_decisions(self):
        for field, invalid_value in [('state_ids', ['S0001']), ('scene_span_ids', ['S9999']),
                                      ('transition', 'shown')]:
            invalid = copy.deepcopy(self.value)
            invalid['checks'][0][field] = invalid_value
            with self.assertRaises(ValueError):
                recovery.compatible_result(json.dumps(invalid), self.probe.scenes, self.probe.catalog)
        invalid = copy.deepcopy(self.value)
        invalid['memory_conflicts'] = [{'state_ids': ['M0001', 'S0001'], 'reason': '错误引用。'}]
        with self.assertRaises(ValueError):
            recovery.compatible_result(json.dumps(invalid), self.probe.scenes, self.probe.catalog)

    def test_rejects_wrong_request_and_truncated_response(self):
        envelope = self.envelope()
        envelope['request']['model'] = 'wrong-model'
        with self.assertRaisesRegex(ValueError, 'original request'):
            recovery.validate_attempt(envelope, self.probe)
        envelope = self.envelope()
        envelope['error'] = 'Local LLM output truncated'
        with self.assertRaisesRegex(ValueError, 'cannot recover'):
            recovery.validate_attempt(envelope, self.probe)

    def test_selects_first_valid_attempt_not_most_favorable_verdict(self):
        second = copy.deepcopy(self.value)
        second['checks'][0].update(scene_relation='compatible', transition='not_applicable', reason='第二次判断不同。')
        snapshots = []
        for index, value in enumerate([self.value, second]):
            path = self.root / f'attempt_{index}.json'
            atomic_json(path, self.envelope(value))
            snapshots.append({'path': str(path), 'sha256': study.indexed.file_hash(path)})
        result = recovery.first_recoverable(self.probe, snapshots)
        self.assertEqual(result['result']['status'], 'fail')
        self.assertEqual(result['source_attempt'], snapshots[0]['path'])

    def test_recovery_is_offline_and_original_files_remain_unchanged(self):
        source, root = self.root / 'study', self.root / 'recovery'
        self.probe = study.DecisionProbe(source, self.case, 'R01', 'candidate_factorized')
        atomic_json(source / 'cases.json', [self.case])
        atomic_json(source / 'manifest.json', {'implementation': study.dependencies(),
                                              'cases_sha256': gate.digest([self.case]), 'source_files': {}})
        attempt = self.probe.directory / 'attempts/decision/test/1.json'
        atomic_json(attempt, self.envelope())
        before = {str(path): study.indexed.file_hash(path) for path in source.rglob('*.json')}
        with patch('builtins.print'), patch.object(gate.LLM, 'request', side_effect=AssertionError('No API calls')):
            recovery.recover(source, root, 'R01')
            recovery.recover(source, root, 'R01')
            summary = recovery.report(root)
        self.assertEqual(sum(row['recovered'] for row in summary['results']), 1)
        self.assertEqual(before, {str(path): study.indexed.file_hash(path) for path in source.rglob('*.json')})
        atomic_json(attempt, {})
        with self.assertRaisesRegex(ValueError, 'Original attempt changed'):
            recovery.report(root)


if __name__ == '__main__':
    unittest.main()
