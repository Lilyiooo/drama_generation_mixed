import asyncio
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from drama_local import models as pb
from drama_local.runtime import Context, LocalStore
from service.drama_by_creativity import DramaByCreativity
from service.drama_by_creativity.llm_inference import llm_inference_json
from service.drama_by_creativity.local_llm import LLM


def records(root, kind):
    return list(Path(root).glob(f'diagnostics/*/{kind}/*/record.json'))


class DiagnosticTests(unittest.TestCase):
    def test_all_failed_json_generations_and_repairs_are_retained(self):
        class Original:
            def __init__(self): self.counter=0
            async def request(self, *args):
                self.counter+=1
                return SimpleNamespace(response=f'bad original {self.counter}')
        class Repair:
            def __init__(self, **kw): self.counter=0
            async def request(self, *args):
                self.counter+=1
                return SimpleNamespace(response=f'bad repair {self.counter}')
        with tempfile.TemporaryDirectory() as output, patch.dict(os.environ,{'DRAMA_OUTPUT_DIR':output}), \
                patch('service.drama_by_creativity.llm_inference.LLM',Repair):
            service=Original()
            result=asyncio.run(llm_inference_json({},service,'test failure',Context(),retry_times=2,check_times=3))
            self.assertEqual(result,{})
            files=records(output,'json_attempts')
            self.assertEqual(len(files),8)  # Two originals, six repair outputs.
            texts={p.with_name('output.txt').read_text() for p in files}
            self.assertIn('bad original 1',texts)
            self.assertIn('bad original 2',texts)
            self.assertIn('bad repair 6',texts)  # Includes the unvalidated final repair.
            self.assertEqual(len(records(output,'json_validation')),6)
            self.assertEqual(service.last_output,'bad repair 6')

    def test_wrong_scene_shape_saves_skip_reason_and_exact_text(self):
        raw='{"scenes": [{"场次编号": "1-1"}]}'
        async def request(*args): return SimpleNamespace(response=raw)
        req=pb.GenerateDramaByCreativityReq(
            story_info=pb.StoryInfo(story_id='test',plot_type=pb.PlotType.SHORT_CARTOON),
            generate_input=pb.GenerateInputByCreativity(common=pb.GenerateInputCommon(
                episode_nums=1,min_word_count_per_episode=1000,max_word_count_per_episode=1400)),
            generate_data=pb.GenerateDrama(
                story_outline=pb.StoryOutline(story_outline=['总纲'],role_info=['人设'],world_building=['世界观']),
                episode_outline=pb.EpisodeOutline(seasons=[pb.EpisodeOutline.Season(
                    episodes=[pb.EpisodeOutline.Episode(content='第一集大纲')])]),
                select_range=[pb.GenerateDrama.SelectRange(episode_ids=[0])]))
        with tempfile.TemporaryDirectory() as output, patch.dict(os.environ,{'DRAMA_OUTPUT_DIR':output}), \
                patch.object(LLM,'request',request):
            response=asyncio.run(DramaByCreativity(LocalStore(output)).GenerateDrama(Context(),req))
            self.assertEqual(response.result.seasons[0].episodes,[])
            files=records(output,'skipped_episodes')
            self.assertEqual(len(files),1)
            data=json.loads(files[0].read_text())
            self.assertEqual(data['episode_number'],1)
            self.assertIn('非空对象列表',data['reason'])
            self.assertEqual(files[0].with_name('output.txt').read_text(),raw)
            self.assertTrue(Path(data['last_call']).exists())

    def test_transport_keeps_bad_http_and_truncated_text_without_api_key(self):
        import jinja2
        with tempfile.TemporaryDirectory() as output, patch.dict(os.environ,{
            'DRAMA_OUTPUT_DIR':output,'DRAMA_LLM_API_KEY':'private-test-key'}):
            service=LLM(model='qwen-local',system_prompt='',temperature=1,top_p=1,top_k=50,
                        max_length=100,max_new_tokens=20,template=jinja2.Template('{{Text}}'))
            responses=[
                SimpleNamespace(ok=False,status_code=500,text='server failure'),
                SimpleNamespace(ok=True,status_code=200,text='raw truncated HTTP',
                                json=lambda:{'choices':[{'message':{'content':'unfinished {'},'finish_reason':'length'}]}),
                SimpleNamespace(ok=True,status_code=200,text='empty response',
                                json=lambda:{'choices':[{'message':{'content':''},'finish_reason':'stop'}]}),
            ]
            with patch('requests.Session') as session:
                session.return_value.__enter__.return_value.post.side_effect=responses
                for _ in responses:
                    with self.assertRaises((RuntimeError,ValueError)):
                        asyncio.run(service.request(Context(),{'Text':'test prompt'},'same-id'))
            calls=records(output,'model_calls')
            self.assertEqual(len(calls),3)
            contents=[p.with_name('http_response.txt').read_text() for p in calls]
            self.assertIn('server failure',contents)
            self.assertIn('raw truncated HTTP',contents)
            output_texts=[p.with_name('output.txt').read_text() for p in calls if p.with_name('output.txt').exists()]
            self.assertIn('unfinished {',output_texts)
            self.assertIn('',output_texts)
            for p in calls:
                self.assertEqual(json.loads(p.with_name('status.json').read_text())['status'],'error')
            for p in Path(output).rglob('*'):
                if p.is_file(): self.assertNotIn('private-test-key',p.read_text())
