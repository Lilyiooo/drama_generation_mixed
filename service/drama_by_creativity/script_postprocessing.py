"""Independent switches for script postprocessing; both adjustments are disabled by default."""
import os


def enabled(name):
    value = os.environ.get(name, 'false').strip().lower()
    if value in ('1', 'true', 'yes', 'on'):
        return True
    if value in ('0', 'false', 'no', 'off'):
        return False
    raise ValueError(f'{name} 必须为 true 或 false，实际为 {value!r}')


def word_count_adjustment_enabled():
    return enabled('DRAMA_ADJUST_WORD_COUNT')


def coherence_adjustment_enabled():
    return enabled('DRAMA_ADJUST_COHERENCE')
