import asyncio
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from drama_local import models as pb
from service.drama_by_creativity.summary_memory import SummaryMemory, summary_baseline_enabled, memory_prompt
from service.drama_by_creativity.prompts import GENERATE_WHOLE_EPISODE_PROMPT


class SummaryTests(unittest.TestCase):
    def test_opt_in_and_original_prompt_unchanged(self):
        with patch.dict(os.environ,{'DRAMA_SUMMARY_MEMORY_BASELINE':'0'}):
            self.assertFalse(summary_baseline_enabled(pb.GenerateInputByCreativity()))
            self.assertTrue(summary_baseline_enabled(pb.GenerateInputByCreativity(summary_memory_baseline=True)))
        self.assertEqual(memory_prompt(GENERATE_WHOLE_EPISODE_PROMPT,False),GENERATE_WHOLE_EPISODE_PROMPT)
        prompt=memory_prompt(GENERATE_WHOLE_EPISODE_PROMPT,True)
        self.assertIn('前面各集的剧情梗概',prompt)
        self.assertNotIn('## 前五集摘要',prompt)

    def test_restart_rewrite_and_missing_summary(self):
        async def fake(*args):return SimpleNamespace(response='本集实际发生的情节')
        with tempfile.TemporaryDirectory() as output, patch.dict(os.environ,{'DRAMA_OUTPUT_DIR':output}), \
             patch('service.drama_by_creativity.summary_memory.LLM') as client:
            client.return_value.request=fake
            mem=SummaryMemory('test')
            asyncio.run(mem.update_from_episode(None,0,'最终正文'))
            mem=SummaryMemory('test')
            self.assertIn('第1集梗概',mem.retrieve(1))
            with self.assertRaises(ValueError):mem.retrieve(2)
            asyncio.run(mem.update_from_episode(None,1,'第二集正文'))
            asyncio.run(mem.update_from_episode(None,0,'重新生成第一集'))
            state=json.loads(mem.path.read_text())
            self.assertEqual(len(state['episode_summaries']),1)
            self.assertNotIn('第2集',mem.retrieve(1))
            self.assertFalse((Path(output)/'narrative_memory').exists())
