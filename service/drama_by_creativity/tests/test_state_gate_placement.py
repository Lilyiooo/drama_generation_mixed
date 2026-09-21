import copy
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
os.environ['SCRIPTPIPELINE_REPO'] = str(REPO)
module_path = Path(os.environ.get('GATE_MODULE_UNDER_TEST', REPO / 'tools/state_gate_placement.py'))
spec = importlib.util.spec_from_file_location('gate_test_target', module_path)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class GatePlacementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = TemporaryDirectory()
        cls.fixture_root = Path(cls.fixture.name) / 'study'
        gate.prepare(gate.SOURCE, cls.fixture_root)
        cls.manifest, cls.inputs = gate.load(cls.fixture_root)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def verdict(self, status):
        return {'status': status, 'violations': [] if status == 'pass' else [
            {'constraint_id': 'C01', 'evidence_quote': '白无相突然从人群中跳上高台', 'reason': '当下行动'}],
            'reason': '状态判断'}

    def test_real_preparation_freezes_known_error_and_correct_controls(self):
        self.assertEqual(len(self.manifest['blind_map']), 14)
        self.assertIn(gate.SCENE_ERROR_QUOTE, gate.scene_text(self.inputs['original_scenes']))
        self.assertEqual(set(self.inputs['clean_controls']), set(gate.RUNS))
        for run in gate.RUNS:
            self.assertNotIn('白无相', self.inputs['clean_controls'][run]['script'])
        original = (self.fixture_root / 'inputs.json').read_bytes()
        gate.prepare(gate.SOURCE, self.fixture_root)
        self.assertEqual((self.fixture_root / 'inputs.json').read_bytes(), original)

    def test_detection_prompt_hides_expected_labels_and_discriminates_artifact_type(self):
        for name, (kind, text) in gate.examples(self.inputs, 'R01').items():
            payload = gate.check_payload(self.inputs['context'], text, kind, 'Qwen3.6-27B', 4601)
            packet = json.loads(payload['messages'][1]['content'])
            self.assertNotIn('ground_truth', packet)
            self.assertNotIn('expected', packet)
            self.assertNotIn('clean_controls', packet)
            self.assertNotIn(name, packet)
            field = 'candidate_scene_outline' if kind == 'scene' else 'candidate_script'
            self.assertEqual(packet[field], text)
            self.assertEqual(payload['model'], 'Qwen3.6-27B')
        scene = gate.check_payload(self.inputs['context'], '场次文本', 'scene', 'Qwen3.8-27B')
        self.assertNotIn('outline_alignment', scene['messages'][0]['content'])

    def test_repaired_scenes_cannot_drop_or_reorder_original_scenes(self):
        original = self.inputs['original_scenes']
        self.assertGreater(len(original), 1)
        with self.assertRaisesRegex(ValueError, '改变了场次数量'):
            gate.validate_scene_repair(json.dumps(original[:1], ensure_ascii=False), original)
        self.assertEqual(gate.validate_scene_repair(gate.scene_text(original), original), original)

    def test_missed_detection_preserves_error_and_does_not_silently_repair(self):
        calls = []
        def response(directory, name, *args):
            calls.append(name)
            return '原错误场次生成的正文' if name == 'raw_scene_script' else self.verdict('pass')
        with patch.object(gate, 'load', return_value=(self.manifest, self.inputs)), \
             patch.object(gate.base, 'request_cached', side_effect=response):
            gate.generate_run(self.root, 'R01', 'http://test/v1')
        self.assertEqual(set(calls), set(gate.CHECKS) | {'raw_scene_script'})
        front = gate.base.read(self.root / 'variants/scene_gate/R01.json')
        back = gate.base.read(self.root / 'variants/script_gate/R01.json')
        self.assertFalse(front['repair_attempted'])
        self.assertEqual(front['script'], '原错误场次生成的正文')
        self.assertFalse(back['repair_attempted'])
        self.assertEqual(back['script'], self.inputs['context']['original_script'])

    def test_each_gate_repairs_once_and_retains_remaining_failure(self):
        modified = copy.deepcopy(self.inputs['original_scenes'])
        modified[0]['核心功能'] += ' 修订版'
        calls, payloads = [], {}
        def response(directory, name, payload, *args):
            calls.append(name)
            payloads[name] = payload
            if name in gate.CHECKS or name in {'scene_after', 'script_after'}:
                return self.verdict('fail' if name.startswith('error') or name.endswith('after') else 'pass')
            return {'raw_scene_script': '对照正文', 'scene_repair': modified,
                    'scene_gate_script': '前置修复后正文', 'script_repair': '后置修复后正文'}[name]
        with patch.object(gate, 'load', return_value=(self.manifest, self.inputs)), \
             patch.object(gate.base, 'request_cached', side_effect=response):
            gate.generate_run(self.root, 'R02', 'http://test/v1')
        self.assertEqual(calls.count('scene_repair'), 1)
        self.assertEqual(calls.count('script_repair'), 1)
        self.assertIn('修订版', payloads['scene_gate_script']['messages'][1]['content'])
        self.assertEqual(len(payloads['scene_gate_script']['messages']), 2)
        self.assertEqual(payloads['script_repair']['messages'][-2]['content'], self.inputs['context']['original_script'])
        self.assertNotIn('后置修复后正文', payloads['scene_gate_script']['messages'][1]['content'])
        for arm in ('scene_gate', 'script_gate'):
            record = gate.base.read(self.root / 'variants' / arm / 'R02.json')
            self.assertTrue(record['repair_attempted'])
            self.assertEqual(record['final_check']['status'], 'fail')

    def test_judge_requires_candidate_evidence_and_reports_quality_only_for_scripts(self):
        response = json.dumps(self.verdict('fail'), ensure_ascii=False)
        with self.assertRaisesRegex(ValueError, 'exact quote'):
            gate.parse_check(response, '沈惊蛰拿着白无相留下的笔记', 'scene', 'Qwen3.6-27B')
        check = self.verdict('pass')
        raw = json.dumps(check)
        self.assertEqual(gate.parse_check(raw, '合法场次', 'scene', 'Qwen3.8-27B')['status'], 'pass')
        with self.assertRaisesRegex(ValueError, 'outline_alignment'):
            gate.parse_check(raw, '合法正文', 'script', 'Qwen3.8-27B')

    def test_frozen_inputs_and_candidate_checksums_reject_tampering(self):
        gate.base.atomic_json(self.root / 'manifest.json', self.manifest)
        gate.base.atomic_json(self.root / 'inputs.json', {**self.inputs, 'episode': 46})
        with self.assertRaisesRegex(ValueError, 'Frozen inputs changed'):
            gate.load(self.root)
        gate.base.atomic_json(self.root / 'scenes/scene_gate/R01.json', {
            'scenes': [], 'scene_sha256': 'wrong', 'inputs_sha256': gate.base.digest(self.inputs)})
        with self.assertRaisesRegex(ValueError, 'Scene candidate changed'):
            gate.candidate(self.root, {'kind': 'scene', 'arm': 'scene_gate', 'run': 'R01'}, self.inputs)

    def test_reformatted_json_evidence_is_rejected_with_specific_feedback(self):
        text = gate.scene_text(self.inputs['original_scenes'])
        value = self.verdict('fail')
        value['violations'].insert(0, {
            'constraint_id': 'C01', 'evidence_quote': '出场人物: ["白无相"]', 'reason': '名单'})
        with self.assertRaisesRegex(ValueError, r'violations\[0\].evidence_quote is not an exact quote'):
            gate.parse_check(json.dumps(value), text, 'scene', 'Qwen3.6-27B')
        # Valid later evidence must not silently mask an invalid earlier citation.
        value['violations'].pop(0)
        self.assertEqual(gate.parse_check(json.dumps(value), text, 'scene', 'Qwen3.6-27B'), value)

    def test_reason_requires_nonempty_string(self):
        for reason in (' ', 1, ['explanation']):
            value = self.verdict('fail')
            value['violations'][0]['reason'] = reason
            with self.assertRaisesRegex(ValueError, 'reason must be a nonempty string'):
                gate.parse_check(json.dumps(value), gate.scene_text(self.inputs['original_scenes']),
                                 'scene', 'Qwen3.6-27B')

    def test_schema_retry_reports_bad_quote_without_forcing_verdict(self):
        from unittest.mock import MagicMock
        text = gate.scene_text(self.inputs['original_scenes'])
        bad = self.verdict('fail')
        bad['violations'][0]['evidence_quote'] = '出场人物: ["白无相"]'
        good = self.verdict('fail')
        replies = []
        for value in (bad, good):
            data = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]}
            reply = MagicMock()
            reply.text = json.dumps(data)
            reply.json.return_value = data
            replies.append(reply)
        with patch.object(gate.base.requests, 'Session') as session:
            post = session.return_value.__enter__.return_value.post
            post.side_effect = replies
            result = gate.checked_request(self.root, 'check', self.inputs['context'], text,
                                          'scene', 'Qwen3.6-27B', 4601, 'http://test/v1')
            self.assertEqual(result, good)
            self.assertEqual(post.call_count, 2)
            retry = post.call_args_list[1].kwargs['json']['messages'][-1]['content']
            self.assertIn('violations[0].evidence_quote', retry)
            self.assertNotIn('expected', retry)
            gate.checked_request(self.root, 'check', self.inputs['context'], text,
                                 'scene', 'Qwen3.6-27B', 4601, 'http://test/v1')
            self.assertEqual(post.call_count, 2)


if __name__ == '__main__':
    unittest.main()
