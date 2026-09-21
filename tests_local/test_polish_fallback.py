import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import pytest
from drama_local.runtime import Context
from service.drama_by_creativity import process_coherence as module


def test_first_boundary_preserves_remaining_scenes():
    assert module.split_polished_scripts('上一集\n####\n当前场1\n####\n当前场2') == ('上一集', '当前场1\n####\n当前场2')
    assert module.split_polished_scripts('上一集\n\\n####\\n\n当前场1\n\\n####\\n\n当前场2') == ('上一集', '当前场1\n\\n####\\n\n当前场2')
    with pytest.raises(ValueError):
        module.split_polished_scripts('上一集\n####\n')


def run_responses(responses):
    request = AsyncMock(side_effect=[SimpleNamespace(response=r) for r in responses])
    with patch.object(module, 'LLM', return_value=SimpleNamespace(request=request)), patch.object(module.asyncio, 'sleep', new_callable=AsyncMock), patch.object(module.logger, 'error_context'):
        result = asyncio.run(module.polish_plot(Context(), '原上一集', '原当前集'))
    assert request.await_count == 4
    return result


def test_final_retry_is_checked():
    assert run_responses(['坏格式'] * 3 + ['润色上一集\n####\n润色当前集']) == ('润色上一集', '润色当前集')


def test_exhaustion_returns_original_without_crashing():
    assert run_responses(['坏格式'] * 4) == ('原上一集', '原当前集')
