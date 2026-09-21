import asyncio
import importlib
from types import SimpleNamespace

from drama_local import runtime as context

generate_episode_script = importlib.import_module(
    "service.drama_by_creativity.generate_episode_script"
)


def test_adjustment_model_is_called_at_most_once(monkeypatch):
    calls = []

    class FakeLLM:
        def __init__(self, **_kwargs):
            pass

        async def request(self, _ctx, _params, _request_id):
            calls.append(1)
            return SimpleNamespace(response="调整后仍不达标")

    ratio_result = {
        "episode_ratio": {
            "dialogue_ratio": 0.2,
            "narration_ratio": 0.8,
            "is_ratio_ok": False,
        },
        "scenes_not_ok": [],
        "all_scenes_ok": False,
    }
    monkeypatch.setattr(generate_episode_script, "LLM", FakeLLM)
    monkeypatch.setattr(
        generate_episode_script,
        "calc_dialogue_ratio_per_scene",
        lambda _script: ratio_result,
    )

    script_config = {
        "model": "fake",
        "system_prompt": "",
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
        "max_length": 100,
        "max_new_tokens": 100,
    }
    asyncio.run(
        generate_episode_script.check_and_adjust_word_count(
            context.Context(),
            "字数不足",
            "1000-1400",
            script_config,
        )
    )

    assert len(calls) == 1
