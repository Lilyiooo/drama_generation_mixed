import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parents[3]
os.environ['SCRIPTPIPELINE_REPO'] = str(REPO)
path = Path(os.environ.get('BRANCH_MODULE_UNDER_TEST', REPO / 'tools/state_constraint_branch.py'))
spec = importlib.util.spec_from_file_location('branch_test_target', path)
branch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(branch)


class ConstraintBranchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        branch.prepare(branch.SOURCE, cls.root / 'fixture')
        cls.manifest, cls.inputs = branch.load(cls.root / 'fixture')

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def verdict(self, status='pass'):
        return {'status': status, 'violations': [] if status == 'pass' else [
            {'constraint_id': 'C01', 'evidence_quote': '白无相跃上高台', 'reason': '当下现实直接行动'}],
            'reason': '检查结论'}

    def test_real_prepare_preserves_recorded_inputs_and_excludes_old_scenes(self):
        self.assertNotIn('AllSceneOutlines', self.inputs['calls']['script']['parameters'])
        self.assertIn(branch.CONSTRAINT_TEXT, self.inputs['calls']['scene']['parameters']['NarrativeMemory'])
        self.assertEqual(len(self.manifest['blind_map']), 10)
        self.assertEqual(self.inputs['episode'], 45)
        self.assertIn(branch.DEATH_QUOTE, self.inputs['historical_script'])
        before = (self.root / 'fixture/inputs.json').read_bytes()
        branch.prepare(branch.SOURCE, self.root / 'fixture')
        self.assertEqual((self.root / 'fixture/inputs.json').read_bytes(), before)

    def test_constraint_is_only_added_to_control_prompt_and_new_scenes_are_used(self):
        for stage, scenes in [('scene', None), ('script', [{'新分场标记': '本分支新场次'}])]:
            control = branch.generation_payload(self.inputs, stage, scenes, False, 4501)
            guarded = branch.generation_payload(self.inputs, stage, scenes, True, 4501)
            self.assertEqual(control['messages'][0], guarded['messages'][0])
            self.assertEqual(guarded['messages'][1]['content'], control['messages'][1]['content'] + branch.constraint_block(self.inputs))
            for key in ['model', 'temperature', 'top_p', 'top_k', 'seed']:
                self.assertEqual(control[key], guarded[key])
            if stage == 'script':
                self.assertIn('本分支新场次', control['messages'][1]['content'])

    def test_judge_requires_real_candidate_quote_and_consistent_status(self):
        raw = json.dumps(self.verdict('fail'), ensure_ascii=False)
        self.assertEqual(branch.validate_check(raw, '白无相跃上高台，开始讲话。')['status'], 'fail')
        with self.assertRaisesRegex(ValueError, 'exact quote'):
            branch.validate_check(raw, '沈惊蛰拿出白无相留下的笔记。')
        invalid = self.verdict('fail')
        invalid['status'] = 'pass'
        with self.assertRaisesRegex(ValueError, 'pass requires'):
            branch.validate_check(json.dumps(invalid), '白无相跃上高台')
        payload = branch.judge_payload(self.inputs, '白无相留下的笔记', 'Qwen3.8-27B', True)
        self.assertNotIn('blind_map', payload['messages'][1]['content'])
        self.assertNotIn('original_script', payload['messages'][1]['content'])
        self.assertIn('笔记画外音', payload['messages'][0]['content'])

    def test_cached_request_reuses_response_and_rejects_changed_input(self):
        payload = {'model': 'Qwen3.6-27B', 'messages': [{'role': 'user', 'content': 'test'}]}
        response = Mock(text='response')
        response.json.return_value = {'choices': [{'finish_reason': 'stop', 'message': {'content': '完整正文'}}]}
        session = Mock()
        session.__enter__ = Mock(return_value=session)
        session.__exit__ = Mock(return_value=False)
        session.post.return_value = response
        with patch.object(branch.requests, 'Session', return_value=session):
            first = branch.request_cached(self.directory, 'draft', payload, 'http://test/v1', branch.plain_text, 'dep')
            second = branch.request_cached(self.directory, 'draft', payload, 'http://test/v1', branch.plain_text, 'dep')
        self.assertEqual(first, second)
        session.post.assert_called_once()
        with self.assertRaisesRegex(ValueError, 'inputs changed'):
            branch.request_cached(self.directory, 'draft', payload, 'http://test/v1', branch.plain_text, 'changed')

    def test_checked_arm_keeps_exact_draft_if_check_passes(self):
        check = self.verdict()
        calls = []
        def response(directory, name, *args):
            calls.append(name)
            return {'scene': [{'scene': 'new'}], 'draft': '原始分支正文', 'check_before': check}[name]
        with patch.object(branch, 'load', return_value=(self.manifest, self.inputs)), \
             patch.object(branch, 'request_cached', side_effect=response):
            branch.generate_pair(self.directory, 'constraint', 'R01', 'http://test/v1')
        raw = branch.read(self.directory / 'variants/constraint/R01.json')
        checked = branch.read(self.directory / 'variants/checked/R01.json')
        self.assertEqual(raw['script'], checked['script'])
        self.assertFalse(checked['rewritten'])
        self.assertEqual(calls, ['scene', 'draft', 'check_before'])

    def test_checked_arm_rewrites_once_and_retains_unresolved_failure(self):
        calls = []
        def response(directory, name, *args):
            calls.append(name)
            return {'scene': [{'scene': 'new'}], 'draft': '白无相跃上高台', 'check_before': self.verdict('fail'),
                    'rewrite': '白无相跃上高台并说话', 'check_after': self.verdict('fail')}[name]
        with patch.object(branch, 'load', return_value=(self.manifest, self.inputs)), \
             patch.object(branch, 'request_cached', side_effect=response):
            branch.generate_pair(self.directory, 'constraint', 'R01', 'http://test/v1')
        self.assertEqual(calls.count('rewrite'), 1)
        checked = branch.read(self.directory / 'variants/checked/R01.json')
        self.assertEqual(checked['final_check']['status'], 'fail')
        self.assertEqual(branch.read(self.directory / 'variants/constraint/R01.json')['script'], '白无相跃上高台')

    def test_input_and_candidate_tampering_are_rejected(self):
        branch.atomic_json(self.directory / 'inputs.json', self.inputs)
        branch.atomic_json(self.directory / 'manifest.json', {**self.manifest, 'inputs_sha256': 'changed'})
        with self.assertRaisesRegex(ValueError, 'Frozen inputs changed'):
            branch.load(self.directory)
        branch.write_variant(self.directory / 'variants/control/R01.json', '正文', {'inputs_sha256': branch.digest(self.inputs)})
        value = branch.read(self.directory / 'variants/control/R01.json')
        value['script'] = '篡改正文'
        branch.atomic_json(self.directory / 'variants/control/R01.json', value)
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            branch.candidate(self.directory, {'arm': 'control', 'run': 'R01'}, self.inputs)


if __name__ == '__main__':
    unittest.main()
