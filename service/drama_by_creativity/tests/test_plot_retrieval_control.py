import asyncio
import importlib
from types import SimpleNamespace

import jinja2

from service.drama_by_creativity.plot_retrieval_control import (
    is_plot_retrieval_disabled,
    select_prompt_reference,
)
from service.drama_by_creativity.prompts.episode_outline import (
    GENERATE_ALL_EPISODE_OUTLINE_PROMPT,
    GENERATE_EPISODE_OUTLINE_BY_SCRIPT_PROMPT,
)
from service.drama_by_creativity.prompts.script import (
    GENERATE_SCENE_OUTLINE_PROMPT,
    GENERATE_SCENE_PLOT_PROMPT,
    GENERATE_WHOLE_EPISODE_PROMPT,
)


def test_switch_defaults_to_false(monkeypatch):
    monkeypatch.delenv("DRAMA_DISABLE_PLOT_RETRIEVAL", raising=False)
    generate_input = SimpleNamespace(reference="")
    assert is_plot_retrieval_disabled(generate_input) is False
    assert select_prompt_reference(generate_input, "召回结果") == "召回结果"


def test_direct_boolean_field_disables_retrieval():
    generate_input = SimpleNamespace(reference="", disable_plot_retrieval=True)
    assert is_plot_retrieval_disabled(generate_input) is True
    assert select_prompt_reference(generate_input, "召回结果") == ""


def test_user_reference_is_preserved_when_retrieval_is_disabled():
    generate_input = SimpleNamespace(reference="用户参考", disable_plot_retrieval=True)
    assert select_prompt_reference(generate_input, "召回结果") == "用户参考"


def test_local_input_disables_retrieval_by_default():
    from drama_local import models as pb

    generate_input = pb.GenerateInputByCreativity()
    assert is_plot_retrieval_disabled(generate_input) is True


def test_empty_reference_removes_reference_sections_from_prompts():
    templates = [
        GENERATE_ALL_EPISODE_OUTLINE_PROMPT,
        GENERATE_EPISODE_OUTLINE_BY_SCRIPT_PROMPT,
        GENERATE_SCENE_OUTLINE_PROMPT,
        GENERATE_SCENE_PLOT_PROMPT,
        GENERATE_WHOLE_EPISODE_PROMPT,
    ]
    for source in templates:
        rendered = jinja2.Template(source).render(Reference="", PlotReference="")
        assert "创作参考桥段" not in rendered
        assert "创作参考套路" not in rendered


def test_nonempty_reference_keeps_reference_sections():
    outline_rendered = jinja2.Template(GENERATE_ALL_EPISODE_OUTLINE_PROMPT).render(
        Reference="用户参考"
    )
    script_rendered = jinja2.Template(GENERATE_WHOLE_EPISODE_PROMPT).render(
        PlotReference="用户参考"
    )
    assert "创作参考套路" in outline_rendered
    assert "创作参考桥段" in script_rendered


def test_prev_three_outline_skips_search_and_omits_prompt_section(monkeypatch):
    from drama_local import runtime as context
    from drama_local import models as pb

    module = importlib.import_module(
        "service.drama_by_creativity.generate_prev_three_episode_outline"
    )
    search_called = False
    captured_params = {}

    async def fake_search(*args, **kwargs):
        nonlocal search_called
        search_called = True
        return [{"content": "不应被使用"}]

    async def fake_inference(*, params, **kwargs):
        captured_params.update(params)
        return [
            {
                "episode_id": episode_id,
                "title": f"第{episode_id}集",
                "core_plot": "核心剧情",
            }
            for episode_id in range(1, 4)
        ]

    monkeypatch.setenv("DRAMA_DISABLE_PLOT_RETRIEVAL", "1")
    monkeypatch.setattr(module, "search_plot_from_library", fake_search)
    monkeypatch.setattr(module, "llm_inference_json", fake_inference)
    monkeypatch.setattr(module, "LLM", lambda **kwargs: object())
    monkeypatch.setattr(module, "save_realtime", lambda *args, **kwargs: None)

    request = pb.GenerateEpisodeOutlineByCreativityReq(
        story_info=pb.StoryInfo(plot_type=pb.PlotType.SHORT_CARTOON),
        generate_input=pb.GenerateInputByCreativity(
            core_story="核心故事",
            common=pb.GenerateInputCommon(episode_nums=60),
        ),
    )
    response = asyncio.run(
        module.generate_prev_three_episode_outline(context.Context(), request)
    )

    assert search_called is False
    assert captured_params["Reference"] == ""
    assert len(response.episode_outline.seasons[0].episodes) == 3
