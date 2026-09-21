import asyncio
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from drama_local.runtime import Context
from service.drama_by_creativity.process_coherence import polish_plot


class PolishSplitTests(unittest.TestCase):
    def test_all_trailing_scenes_survive_real_failure_responses(self):
        root = Path(__file__).resolve().parents[1] / 'reports/polish_loss_20260912'
        samples = [('synthetic', '上一集结尾\n####\n本集第一场\n####\n本集第二场\n####\n本集第三场')]
        samples.extend((str(p), p.read_text()) for p in sorted(root.glob('*_polish_raw.txt')))
        for name, raw in samples:
            with self.subTest(name=name), patch(
                'service.drama_by_creativity.process_coherence.LLM'
            ) as client:
                async def request(*args):
                    return SimpleNamespace(response=raw)
                client.return_value.request = request
                prev, current = asyncio.run(polish_plot(Context(), 'old', 'new'))
                self.assertEqual(prev + '\n####\n' + current, raw)
                self.assertEqual(current, raw.partition('\n####\n')[2])
