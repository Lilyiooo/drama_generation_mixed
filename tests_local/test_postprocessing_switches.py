import asyncio
import importlib
from unittest.mock import patch
import pytest
from service.drama_by_creativity import script_postprocessing as switches
coherence = importlib.import_module('service.drama_by_creativity.process_coherence')
script = importlib.import_module('service.drama_by_creativity.generate_episode_script')


def test_off_preserves_text_without_model_or_validation(monkeypatch):
    monkeypatch.delenv('DRAMA_ADJUST_WORD_COUNT', raising=False)
    monkeypatch.delenv('DRAMA_ADJUST_COHERENCE', raising=False)
    text = '  原文\n####\n第二场\n'
    with patch.object(coherence, 'LLM', side_effect=AssertionError('不应调用')), patch.object(script, 'LLM', side_effect=AssertionError('不应调用')):
        assert asyncio.run(coherence.polish_plot(None, '上一集', text)) == ('上一集', text)
        assert asyncio.run(script.check_and_adjust_word_count(None, text, '不应解析', {})) == text


def test_defaults_and_independence(monkeypatch):
    for key in ('DRAMA_ADJUST_WORD_COUNT', 'DRAMA_ADJUST_COHERENCE'):
        monkeypatch.delenv(key, raising=False)
    assert not switches.word_count_adjustment_enabled() and not switches.coherence_adjustment_enabled()
    monkeypatch.setenv('DRAMA_ADJUST_WORD_COUNT', 'true')
    assert switches.word_count_adjustment_enabled() and not switches.coherence_adjustment_enabled()
    monkeypatch.setenv('DRAMA_ADJUST_COHERENCE', 'true')
    assert switches.word_count_adjustment_enabled() and switches.coherence_adjustment_enabled()
    monkeypatch.setenv('DRAMA_ADJUST_COHERENCE', 'typo')
    with pytest.raises(ValueError): switches.coherence_adjustment_enabled()
