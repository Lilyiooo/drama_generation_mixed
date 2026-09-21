"""八字段对照实验：直接使用原始详细大纲，分阶段逐集生成后进行三轮评审重写。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import demo_polish_story_to_episode_outline as base
from service.drama_by_creativity.eight_field_episode_outline import (
    FIELDS, EXAMPLE, EPISODE_PROMPT, EIGHT_REVIEW_PROMPT, EIGHT_REWRITE_PROMPT,
    validate_eight_fields, review_eight_fields, generate_eight_fields,
)


async def generate(out, original, background, total):
    for name, prompt in [('episode', EPISODE_PROMPT), ('review', EIGHT_REVIEW_PROMPT), ('rewrite', EIGHT_REWRITE_PROMPT)]:
        (out/f'{name}_prompt_template.txt').write_text(prompt, encoding='utf-8')
    common = dict(background, RoleDescription='你是一位擅长叙事一致性与短剧节奏的专业编剧。', DramaType='短剧漫剧')
    common.setdefault('Reference', '')
    base.atomic_json(out/'03_story_outline/outline.json', original)
    base.atomic_json(out/'generation_background.json', common)
    print('直接使用原始详细大纲，开始阶段规划、八字段逐集生成与三轮审校择优。', flush=True)
    result = await generate_eight_fields(base.Context(), base.transfer_outline_to_str(original), total, common, EPISODE_PROMPT)
    base.atomic_json(out/'episode_outlines.json', result)
    return result


if __name__ == '__main__':
    base.main(generate_fn=generate, output_prefix='qwen36_27b_eight_fields_demo_', polish=False)
