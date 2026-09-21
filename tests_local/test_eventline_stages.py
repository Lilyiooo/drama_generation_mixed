import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from service.drama_by_creativity import eventline_refinement as refinement


class StageSelectionTests(unittest.TestCase):
    def test_stage_parser_and_default(self):
        self.assertEqual(refinement.parse_refinement_stages('开端，发展 高潮,结局'),
                         ('开端', '发展', '高潮', '结局'))
        with patch.dict('os.environ', {}, clear=True):
            self.assertEqual(refinement._load_config_from_env().stages, ('开端', '发展', '高潮', '结局'))
            self.assertEqual(refinement._load_config_from_env().parallel_workers, 32)
        with patch.dict('os.environ', {'DRAMA_EVENTLINE_PARALLEL_WORKERS':'12'}):
            self.assertEqual(refinement._load_config_from_env().parallel_workers, 12)
        for value in ['', '发展,发展']:
            with self.assertRaises(ValueError):
                refinement.parse_refinement_stages(value)

    def test_ordered_outline_and_cumulative_trope_handoff(self):
        original = {'framework': [{'stage': s, 'event_list': [{'event': '原始' + s}]}
                                  for s in ['开端', '发展', '高潮', '结局']]}
        for selected in [('开端', '发展', '高潮', '结局'), ('发展', '结局')]:
            with self.subTest(selected=selected), tempfile.TemporaryDirectory() as tmp:
                received = []
                expected = copy.deepcopy(original)

                def generate(**kwargs):
                    stage = kwargs['stage_name']
                    self.assertEqual(stage, selected[len(received)])
                    self.assertEqual(json.loads(kwargs['outline_path'].read_text()), expected)
                    previous_bank = [{'stage': s} for s in received]
                    self.assertEqual(kwargs['initial_trope_bank'], previous_bank or None)
                    received.append(stage)
                    refinement._replace_stage_events(expected, stage, ['细化' + stage])
                    return {'final_event_line': ['细化' + stage],
                            'trope_bank': previous_bank + [{'stage': stage}]}

                with patch.object(refinement, 'generate_stage_eventline_with_local_tree_v9', side_effect=generate):
                    result = refinement._refine_story_outline_sync(
                        original, refinement.EventlineRefinementConfig(stages=selected), Path(tmp))
                self.assertEqual(result, expected)
                self.assertTrue(all(s['event_list'][0]['event'].startswith('原始') for s in original['framework']))
                manifest = json.loads((Path(tmp) / 'manifest.json').read_text())
                self.assertEqual([s['stage'] for s in manifest['stages']], list(selected))
                for i, record in enumerate(manifest['stages']):
                    self.assertEqual(json.loads(Path(record['trope_bank_path']).read_text()),
                                     [{'stage': s} for s in selected[:i + 1]])

    def test_missing_stage_fails_before_model_call(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            refinement, 'generate_stage_eventline_with_local_tree_v9'
        ) as generate:
            with self.assertRaisesRegex(ValueError, '结局'):
                refinement._refine_story_outline_sync(
                    {'framework': [{'stage': '发展'}]},
                    refinement.EventlineRefinementConfig(stages=('发展', '结局')), Path(tmp))
            generate.assert_not_called()
