import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import jinja2

from drama_local.runtime import Context
from service.drama_by_creativity.scene_outline_inference import (
    scene_outline_inference_json, validate_scene_outline, validate_repair_preserves_scenes,
)
from service.drama_by_creativity.prompts.scene_repair import REPAIR_SCENE_OUTLINE_JSON_PROMPT


def scene(index=1):
    return {'场次编号':f'1-{index}','时间与地点':'夜，庭院','出场人物':['甲'],
            '核心功能':'推进','情节概要':'原有完整剧情','人物动机与目标':[{'角色':'甲','动机与目标':'追债'}],
            '冲突与张力':['债主受阻'],'主要情节':['甲说：“一字不能少。”'],'悬念钩子':'乙出现'}


class RetryTests(unittest.TestCase):
    def run_case(self, originals, repairs):
        calls=[]
        repair_params=[]
        class Original:
            async def request(self, ctx, params, request_id):
                calls.append('generate')
                if params != {'EpisodeIdx':'1'}: raise AssertionError('Original parameters changed')
                return SimpleNamespace(response=originals.pop(0))
        class Repair:
            def __init__(self, **kwargs):
                self.template=kwargs['template']
            async def request(self, ctx, params, request_id):
                calls.append('repair');repair_params.append(dict(params))
                return SimpleNamespace(response=repairs.pop(0))
        with patch('service.drama_by_creativity.scene_outline_inference.LLM',Repair):
            result=asyncio.run(scene_outline_inference_json({'EpisodeIdx':'1'},Original(),'scene test',Context()))
        return result,calls,repair_params

    def test_successful_regeneration_never_calls_repair(self):
        good=json.dumps([scene()],ensure_ascii=False)
        result,calls,_=self.run_case(['bad JSON',good],[])
        self.assertEqual(result,[scene()]);self.assertEqual(calls,['generate','generate'])

    def test_three_original_attempts_then_repair(self):
        good=json.dumps([scene()],ensure_ascii=False)
        result,calls,params=self.run_case(['bad1','bad2','bad3'],[good])
        self.assertEqual(calls,['generate']*3+['repair'])
        self.assertEqual(params[0]['Ret'],'bad3');self.assertEqual(result,[scene()])

    def test_each_repair_including_last_is_validated_against_original(self):
        good=json.dumps([scene(),scene(2)],ensure_ascii=False)
        damaged=good[:-1]
        one_scene=json.dumps([scene()],ensure_ascii=False)
        result,calls,params=self.run_case([damaged]*3,[one_scene,'bad repair',good])
        self.assertEqual(calls,['generate']*3+['repair']*3)
        self.assertEqual(result,[scene(),scene(2)])
        self.assertTrue(all(p['Ret']==damaged for p in params))
        self.assertIn('改变了场次数量',params[1]['ValidationError'])

    def test_wrong_shape_and_missing_fields_cannot_escape_validation(self):
        wrong=json.dumps(scene(),ensure_ascii=False)
        partial=json.dumps([{'主要情节':['残缺内容'],'悬念钩子':'后续'}],ensure_ascii=False)
        result,calls,_=self.run_case([wrong,partial,'[]'],[wrong,partial,'[]'])
        self.assertEqual(result,{})
        self.assertEqual(calls,['generate']*3+['repair']*3)

    def test_complete_prompt_template_is_valid_json(self):
        rendered=jinja2.Template(REPAIR_SCENE_OUTLINE_JSON_PROMPT).render(EpisodeIdx='1',Ret='ERROR',ValidationError='test')
        template=rendered.split('## 完整格式模板\n',1)[1].split('## 校验错误',1)[0].strip()
        self.assertEqual(len(validate_scene_outline(template,1)),1)
        self.assertIn('不允许删减任何有效文字信息',rendered)
