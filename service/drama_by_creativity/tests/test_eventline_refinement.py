import asyncio
import json

from drama_local import runtime as context

from service.drama_by_creativity import eventline_refinement


def _outline():
    return {
        "title": "测试剧",
        "background": "背景",
        "framework": [
            {"stage": "开端", "event_list": [{"event": "开端锚点"}]},
            {"stage": "发展", "event_list": [{"event": "发展锚点"}]},
            {"stage": "高潮", "event_list": [{"event": "高潮锚点"}]},
            {"stage": "结局", "event_list": [{"event": "结局锚点"}]},
        ],
        "highlights": "亮点",
        "summary": "总结",
    }


def test_refinement_replaces_configured_stages_and_reuses_trope_bank(monkeypatch, tmp_path):
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        stage = kwargs["stage_name"]
        result = {
            "final_event_line": [f"{stage}细化事件1", f"{stage}细化事件2"],
            "trope_bank": [{"cause": stage, "method": "方式", "outcome": "结果", "response": "反应"}],
        }
        with open(kwargs["output_path"], "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False)
        return result

    monkeypatch.setenv("DRAMA_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(eventline_refinement, "generate_stage_eventline_with_local_tree_v9", fake_generate)

    original = _outline()
    refined = asyncio.run(eventline_refinement.refine_story_outline_eventlines(context.Context(), original))

    assert original["framework"][1]["event_list"] == [{"event": "发展锚点"}]
    assert refined["framework"][0]["event_list"] == [{"event": "开端锚点"}]
    assert refined["framework"][1]["event_list"] == [
        {"event": "发展细化事件1"},
        {"event": "发展细化事件2"},
    ]
    assert refined["framework"][2]["event_list"] == [
        {"event": "高潮细化事件1"},
        {"event": "高潮细化事件2"},
    ]
    assert refined["framework"][3]["event_list"] == [{"event": "结局锚点"}]
    assert len(calls) == 2
    assert calls[0]["max_events_to_anchor"] == 4
    assert calls[0]["max_local_chains_per_anchor"] == 4
    assert calls[0]["initial_trope_bank"] is None
    assert calls[1]["initial_trope_bank"] == [
        {"cause": "发展", "method": "方式", "outcome": "结果", "response": "反应"}
    ]
    assert (tmp_path / "03_story_outline" / "eventline_refinement_v9" / "outline_refined_final.json").exists()
    assert (tmp_path / "03_story_outline" / "eventline_refinement_v9" / "manifest.json").exists()


def test_refinement_can_be_explicitly_disabled(monkeypatch):
    monkeypatch.setenv("DRAMA_EVENTLINE_REFINEMENT_ENABLED", "0")
    original = _outline()
    assert asyncio.run(
        eventline_refinement.refine_story_outline_eventlines(context.Context(), original)
    ) is original
