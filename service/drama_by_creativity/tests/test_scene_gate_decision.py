import asyncio
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from drama_local.runtime import atomic_json
from service.drama_by_creativity import scene_gate_decision as decision
from service.drama_by_creativity import scene_state_gate as gate
from tools import scene_gate_decision_study as study
from tools import scene_gate_calibration as original


class SceneGateDecisionTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cases = study.controlled_cases()
        self.case = self.cases[0]
        self.states = gate.state_catalog(self.case['view'])
        self.check = {'scene_id': '2-1', 'domain': 'life', 'state_ids': ['M0001'], 'scene_span_ids': ['S0001'],
                      'state_basis': 'consistent', 'scene_relation': 'contradicts', 'transition': 'not_shown',
                      'reason': '已死亡的人在现实中行动。'}

    def validate(self, checks=None, conflicts=None, scenes=None, states=None, notes=None):
        raw = {'memory_conflicts': conflicts or [], 'checks': checks or [self.check], 'writing_notes': notes or []}
        return decision.validate_decision(json.dumps(raw), scenes or self.case['scenes'], states or self.states)

    def test_aggregate_all_allowed_classifications_without_model_status(self):
        expected = [
            ('consistent', 'contradicts', 'not_shown', 'hard_conflict', 'fail'),
            ('consistent', 'compatible', 'shown', 'legal_transition', 'pass'),
            ('consistent', 'compatible', 'not_applicable', 'no_conflict', 'pass'),
            ('insufficient', 'unclear', 'not_applicable', 'insufficient_evidence', 'uncertain'),
            ('not_needed', 'compatible', 'not_applicable', 'no_conflict', 'pass'),
        ]
        for basis, relation, transition, kind, status in expected:
            check = {**self.check, 'state_basis': basis, 'scene_relation': relation, 'transition': transition}
            result = self.validate([check], notes=['对白可以更紧凑。'])
            self.assertEqual(result['status'], status)
            self.assertEqual(result['checks'][0]['classification'], kind)
            self.assertFalse(result['auto_repair_executed'])

    def test_memory_conflict_requires_both_sources_and_routes_to_review(self):
        states = {**self.states, 'M0002': {'field': 'character_state', 'text': '同一时点李舟活着。'}}
        check = {**self.check, 'state_ids': ['M0001', 'M0002'], 'state_basis': 'conflicting',
                 'scene_relation': 'unclear', 'transition': 'not_applicable'}
        conflict = {'state_ids': ['M0001', 'M0002'], 'reason': '同一时点互相矛盾。'}
        result = self.validate([check], [conflict], states=states)
        self.assertEqual(result['checks'][0]['classification'], 'memory_conflict')
        self.assertEqual(result['status'], 'uncertain')
        self.assertEqual(result['action'], 'review_memory_or_evidence')
        with self.assertRaisesRegex(ValueError, 'complete declared'):
            self.validate([{**check, 'state_ids': ['M0001']}], [conflict], states=states)

    def test_conflicting_memory_cannot_also_be_hard_evidence(self):
        states = {**self.states, 'M0002': {'field': 'character_state', 'text': '李舟活着。'}}
        conflict = {'state_ids': ['M0001', 'M0002'], 'reason': '不一致。'}
        with self.assertRaisesRegex(ValueError, 'consistent evidence'):
            self.validate(conflicts=[conflict], states=states)

    def test_independent_hard_conflict_is_not_hidden_by_other_uncertainty(self):
        states = {**self.states, 'M0002': {'field': 'character_state', 'text': '李舟活着。'},
                  'M0003': {'field': 'resources_and_evidence', 'text': '原件已毁。'}}
        conflict = {'state_ids': ['M0001', 'M0002'], 'reason': '生死矛盾。'}
        memory = {**self.check, 'state_ids': ['M0001', 'M0002'], 'state_basis': 'conflicting',
                  'scene_relation': 'unclear', 'transition': 'not_applicable'}
        hard = {**self.check, 'state_ids': ['M0003'], 'domain': 'object'}
        result = self.validate([memory, hard], [conflict], states=states)
        self.assertEqual(result['status'], 'fail')
        self.assertFalse(result['auto_repair_executed'])

    def test_inconsistent_structured_fields_and_top_level_status_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Inconsistent'):
            self.validate([{**self.check, 'transition': 'shown'}])
        raw = {'memory_conflicts': [], 'checks': [self.check], 'writing_notes': [], 'status': 'pass'}
        with self.assertRaisesRegex(ValueError, 'do not output status'):
            decision.validate_decision(json.dumps(raw), self.case['scenes'], self.states)

    def test_unknown_ids_cross_scene_citations_and_missing_coverage_fail(self):
        for field, value in [('state_ids', ['M9999']), ('scene_span_ids', ['S9999'])]:
            with self.assertRaises(ValueError):
                self.validate([{**self.check, field: value}])
        scenes = self.case['scenes'] + [{**self.case['scenes'][0], '场次编号': '2-2'}]
        with self.assertRaisesRegex(ValueError, 'cover all scenes'):
            self.validate(scenes=scenes)
        with self.assertRaisesRegex(ValueError, 'named scene'):
            self.validate([{**self.check, 'scene_id': '2-2'}], scenes=scenes)

    def test_short_reasons_and_relevant_memory_conflicts_required(self):
        with self.assertRaisesRegex(ValueError, '300 characters'):
            self.validate([{**self.check, 'reason': '字' * 301}])
        states = {**self.states, 'M0002': {'field': 'resources_and_evidence', 'text': '钱袋0。'},
                  'M0003': {'field': 'resources_and_evidence', 'text': '同一钱袋10。'}}
        with self.assertRaisesRegex(ValueError, 'relevant scene'):
            self.validate(conflicts=[{'state_ids': ['M0002', 'M0003'], 'reason': '余额矛盾。'}], states=states)

    def test_revised_fixtures_do_not_leak_labels_and_remove_identified_confounders(self):
        self.assertEqual(len(self.cases), 16)
        for case in self.cases:
            probe = study.DecisionProbe(self.root, case, 'R01', 'candidate_factorized')
            packet = probe.packet(case['scenes'])
            self.assertFalse({'expected_status', 'expected_kind', 'origin', 'id'} & set(packet))
            control = study.DecisionProbe(self.root, case, 'R01', 'control_indexed')
            self.assertEqual(packet, control.packet(case['scenes']))
        lookup = {case['id']: case for case in self.cases}
        self.assertNotIn('李舟', lookup['life_legal']['scenes'][0]['出场人物'])
        self.assertIn('铺主', lookup['resource_legal']['scenes'][0]['出场人物'])
        self.assertIn('柜台', lookup['resource_legal']['scenes'][0]['时间与地点'])
        self.assertIn('门外', lookup['knowledge_legal']['scenes'][0]['时间与地点'])

    def test_requests_resume_and_judge_is_blind_to_generation_results(self):
        raw = {'memory_conflicts': [], 'checks': [self.check], 'writing_notes': []}
        probe = study.DecisionProbe(self.root, self.case, 'R01', 'candidate_factorized')
        request = AsyncMock(return_value=SimpleNamespace(response=json.dumps(raw)))
        with patch.object(gate.LLM, 'request', request), patch.object(gate.SceneStateGate, 'apply', AsyncMock()) as apply:
            asyncio.run(probe.run())
            asyncio.run(probe.run())
            apply.assert_not_awaited()
        request.assert_awaited_once()
        self.assertEqual(probe.read_result()['status'], 'fail')
        judge = study.DecisionProbe(self.root, self.case, 'R01', 'judge')
        self.assertEqual(probe.packet(self.case['scenes']), judge.packet(self.case['scenes']))
        self.assertEqual(judge.model, 'Qwen3.8-27B')
        self.assertFalse((probe.directory / 'repair.json').exists())

    def test_prepare_preserves_real_scenes_and_drops_unjustified_global_labels(self):
        source, root = self.root / 'source', self.root / 'study'
        real = [{**copy.deepcopy(self.case), 'id': name, 'origin': 'real_saved_generation',
                 'category': 'life', 'expected': expected} for name, expected in [('real_bad', 'fail'), ('real_clean', 'pass')]]
        atomic_json(source / 'cases.json', real)
        atomic_json(source / 'manifest.json', {'cases_sha256': gate.digest(real), 'implementation': original.dependencies()})
        study.prepare(source, root)
        _, cases = study.load(root)
        self.assertEqual(len(cases), 18)
        for before, after in zip(real, cases[-2:]):
            self.assertEqual(before['scenes'], after['scenes'])
            self.assertEqual(before['view'], after['view'])
            self.assertIsNone(after['expected_status'])
        with patch('builtins.print'):
            self.assertEqual(len(study.report(root)['results']), 108)
        altered = gate.read(root / 'cases.json')
        altered[0]['expected_status'] = 'pass'
        atomic_json(root / 'cases.json', altered)
        with self.assertRaisesRegex(ValueError, 'Frozen study'):
            study.load(root)


if __name__ == '__main__':
    unittest.main()
