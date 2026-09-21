import json
import threading
import time
from unittest.mock import Mock, patch

import pytest
from tools import stage_eventline_local_tree_v9 as v9


def common():
    return dict(title='测试', background='背景', highlights='卖点', summary='摘要',
                stage_name='结局', seed_events=['目标'], fixed_prefix=['男主已完全恢复记忆', '反派仍被关押'],
                target_anchor='反派认罪', future_anchors=['退隐'], trope_bank=[], anchor_remaining_budget=4)


def chain(events, reached=False):
    return dict(events=events, path=[{'event':e,'progress_to_anchor':50} for e in events],
                anchor_reached=reached, final_progress_to_anchor=50, terminal_reason='depth_limit')


def score(coherence=80, novelty=70, budget_fit=60):
    return dict(coherence=coherence, novelty=novelty, budget_fit=budget_fit,
                consistency_review='前序反派被关押，本链在牢中讯问，状态衔接成立。',
                novelty_review='采用交易获得口供。', budget_review='尚需一条认罪事件，额度足够。', reason='可自然推进。')


def test_prompt_one_chain_full_history_and_actual_budget():
    kwargs=common()
    prompt=v9.build_local_chain_score_messages(**kwargs,chain=chain(['审问', '交易']))[0]['content']
    assert json.dumps(kwargs['fixed_prefix'],ensure_ascii=False,indent=2) in prompt
    assert '局部事件链是目前设计的' in prompt
    assert '本轮开始时到该锚点的剩余事件预算' not in prompt
    assert '剩余事件预算：2 条事件' in prompt
    assert '包括但不限于' in prompt
    assert '候选链 A' not in prompt
    assert '剩余事件预算：0 条事件' in v9.build_local_chain_score_messages(**kwargs,chain=chain(['a']*4))[0]['content']
    with pytest.raises(ValueError):
        v9.build_local_chain_score_messages(**kwargs,chain=chain(['a']*5))


def test_schema_and_exact_weights_selection():
    assert v9.parse_local_chain_score(score())==score()
    assert v9.parse_local_chain_score(score(0,100,55.5))['budget_fit']==55.5
    for bad in [dict(scores=[score()]), score(coherence=True), score(budget_fit=101), score(novelty=float('nan')), dict(score(),consistency_review='')]:
        with pytest.raises(ValueError):v9.parse_local_chain_score(bad)
    assert v9.compute_local_chain_score(score(100,10,10))==46
    a=dict(score(10,5,5),score=7,anchor_reached=False)
    b=dict(score(7,7,7),score=7,anchor_reached=True)
    assert v9.choose_best_local_chain([b,a]) is a  # no reached bonus/tiebreak


def test_parallel_one_request_per_chain_and_ordered_results():
    gate=threading.Barrier(2)
    seen=[]
    def query(messages,**kwargs):
        text=messages[0]['content']; first='独有甲' in text
        assert first != ('独有乙' in text)
        assert '反派仍被关押' in text
        seen.append(text)
        gate.wait(timeout=5)  # succeeds only if requests actually overlap
        if first:time.sleep(0.02)
        result=score(9 if first else 3)
        return result,json.dumps(result,ensure_ascii=False)
    with patch.object(v9,'query_json',side_effect=query):
        scores,raw,records=v9.score_local_chains_in_parallel(
            **common(),terminal_chains=[chain(['独有甲']),chain(['独有乙','追加'])],
            model_name='test',max_new_tokens=100,parallel_workers=2,
            progress_logger=Mock(),anchor_index=0,total_anchors=1,round_index=1)
    assert len(seen)==2
    assert [s['coherence'] for s in scores]==[9,3]
    assert [r['remaining_budget_after_chain'] for r in records]==[3,2]
    assert [r['candidate_count'] for r in records]==[1,1]


def test_invalid_score_retries_instead_of_default(tmp_path):
    with patch.object(v9,'query_json',side_effect=[({},'{}'),(score(),'valid')]) as q, patch.object(v9,'write_llm_failure_dump',return_value=tmp_path/'failure.json'), patch.object(v9.time,'sleep'):
        result,raw=v9.score_local_chain_once(**common(),chain=chain(['审问']),
            model_name='test',max_new_tokens=100,progress_logger=Mock(),
            anchor_index=0,total_anchors=1,round_index=1,chain_index=1,chain_count=1)
    assert result==score() and q.call_count==2


def test_stage_settlement_and_budget_across_rounds(tmp_path):
    outline=tmp_path/'outline.json'
    outline.write_text(json.dumps({'title':'t','framework':[
        {'stage':'开端','event_list':[{'event':'前序全部保留'}]},
        {'stage':'结局','event_list':[{'event':'认罪'}]}]},ensure_ascii=False))
    seen=[]
    def collect(_):
        return [chain(['搜证']),chain(['闲聊'])] if not seen else [chain(['认罪'],True)]
    def judge(**kwargs):
        seen.append(kwargs)
        scores=[score(10,10,10),score(1,1,1)] if len(seen)==1 else [score(9,9,9)]
        return scores,'raw', [{'candidate_count':1} for s in scores]
    tree=dict(tree={},layer_summaries=[],completed_paths=[],incomplete_paths=[])
    out=tmp_path/'result.json'
    with patch.object(v9,'build_tree_with_parallel_generation',return_value=tree), patch.object(v9,'collect_terminal_chains',side_effect=collect), patch.object(v9,'score_local_chains_in_parallel',side_effect=judge), patch.object(v9,'abstract_micro_motifs_once',return_value=[]):
        result=v9.generate_stage_eventline_with_local_tree_v9(
            outline_path=outline,stage_name='结局',branch_factor=2,keep_top_k=1,
            local_max_depth=1,max_events_to_anchor=4,max_local_chains_per_anchor=4,
            parallel_workers=2,model_name='test',temperature=0.9,max_new_tokens=100,
            output_path=out,progress_logger=Mock())
    assert [k['anchor_remaining_budget'] for k in seen]==[4,3]
    assert seen[1]['fixed_prefix']==['前序全部保留','搜证']
    assert result['final_event_line']==['搜证','认罪']
    saved=json.loads(out.read_text())
    rounds=saved['segments'][0]['local_chain_rounds']
    assert rounds[0]['selected_local_chain']['score']==10
    assert rounds[0]['selected_local_chain']['remaining_budget_after_chain']==3
    assert 'terminal_chain_score_records' in rounds[0]
