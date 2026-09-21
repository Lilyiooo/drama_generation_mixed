import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from drama_local.runtime import Context, atomic_json
from tools import recover_full_scene_gate_state as recovery


class FakeEngine:
    @staticmethod
    def extract_json_value(value):
        return json.loads(value)


class FakeMemory:
    def __init__(self, root):
        self.root = root
        self.state = {'character_state': ['上一状态']}
        self.engine = FakeEngine()

    @staticmethod
    def validate_extraction(value):
        if set(value) != {'state_delta', 'resolved_goals', 'resolved_unknown', 'retracted_facts'}:
            raise ValueError('invalid extraction')


def valid_result():
    return {'state_delta': {}, 'resolved_goals': [], 'resolved_unknown': [], 'retracted_facts': []}


class StateExtractionRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.memory = FakeMemory(self.root)
        self.script = '第49集已经保存的正式正文'

    def add_prior_truncations(self):
        for attempt in range(1, 4):
            atomic_json(self.root / 'failures' / f'E49_{attempt}.json', {
                'error': recovery.TRUNCATED, 'response': '截断输出',
                'script_sha256': recovery.digest(self.script)})

    def test_existing_three_truncations_start_at_16384_and_keep_binding(self):
        self.add_prior_truncations()
        limits = []

        async def request(client, ctx, params, request_id):
            limits.append(client.max_tokens)
            self.assertIn('此前对同一剧本', params['Prompt'])
            self.assertIn('第49集已经保存的正式正文', params['Prompt'])
            return SimpleNamespace(response=json.dumps(valid_result(), ensure_ascii=False))

        with patch.object(recovery.LLM, 'request', request):
            result = asyncio.run(recovery.extraction_with_extended_limit(
                self.memory, Context(), 48, self.script))
        self.assertEqual(result, valid_result())
        self.assertEqual(limits, [16384])
        binding = next((self.root / 'state_extraction_recovery/E49').glob('*/binding.json'))
        record = json.loads(binding.read_text())
        self.assertEqual(record['limits'], [16384, 24576])
        self.assertEqual(len(record['previous_truncated_attempts']), 3)
        self.assertEqual(record['script_sha256'], recovery.digest(self.script))

    def test_new_extraction_preserves_original_limits_then_adds_16384(self):
        limits = []

        async def request(client, ctx, params, request_id):
            limits.append(client.max_tokens)
            if client.max_tokens < 16384:
                client.last_output = '截断'
                raise ValueError(recovery.TRUNCATED)
            return SimpleNamespace(response=json.dumps(valid_result()))

        with patch.object(recovery.LLM, 'request', request):
            result = asyncio.run(recovery.extraction_with_extended_limit(
                self.memory, Context(), 49, self.script))
        self.assertEqual(result, valid_result())
        self.assertEqual(limits, [4096, 8192, 16384])

    def test_24576_final_failure_is_not_silently_accepted(self):
        self.add_prior_truncations()
        limits = []

        async def request(client, ctx, params, request_id):
            limits.append(client.max_tokens)
            client.last_output = '仍然截断'
            raise ValueError(recovery.TRUNCATED)

        with patch.object(recovery.LLM, 'request', request):
            with self.assertRaisesRegex(RuntimeError, 'recovery failed'):
                asyncio.run(recovery.extraction_with_extended_limit(
                    self.memory, Context(), 48, self.script))
        self.assertEqual(limits, [16384, 24576])
        attempts = list((self.root / 'state_extraction_recovery/E49').glob('*/attempt_*.json'))
        self.assertEqual(len(attempts), 2)

    def test_other_script_does_not_reuse_old_failure(self):
        self.add_prior_truncations()
        self.assertEqual(recovery.previous_truncation(self.memory, 48, '不同正文'), [])


if __name__ == '__main__':
    unittest.main()
