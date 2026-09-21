"""四字段对照实验：直接使用原始详细大纲，分阶段逐集生成后进行三轮评审重写。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import demo_polish_story_to_episode_outline as base
from service.drama_by_creativity.stage_episode_planning import generate_stage_episode_outlines
from service.drama_by_creativity.episode_outline_review import review_episode_outlines
from service.drama_by_creativity.prompts.episode_outline import GENERATE_ALL_EPISODE_OUTLINE_PROMPT
from service.drama_by_creativity.prompts.episode_outline_full_review import REVIEW_PROMPT, REWRITE_PROMPT

FIELDS = {'episode_id', 'title', 'core_plot', 'roles'}
EXAMPLE = '''[
  {"episode_id": 1, "title": "本集标题", "core_plot": "本集核心剧情，约200字，写清具体行动与结果", "roles": ["人物姓名"]}
]'''
start = GENERATE_ALL_EPISODE_OUTLINE_PROMPT.index('请以JSON格式输出最终的结果')
EPISODE_PROMPT = GENERATE_ALL_EPISODE_OUTLINE_PROMPT[:start] + '''仅输出JSON数组。每集必须且只能包含episode_id、title、core_plot、roles四个字段。集号使用当前阶段规定的全剧集号，从{{StartEpisodeId}}连续到{{EndEpisodeId}}。以下episode_id仅为示例：
''' + EXAMPLE
FOUR_REVIEW_PROMPT = REVIEW_PROMPT.replace('核对角色成长、关系变化和结尾钩子等字段是否与核心剧情一致。', '核对各集核心剧情中的角色成长、关系变化与前后衔接是否一致；每集仅包含episode_id、title、core_plot、roles四项。')
start = REWRITE_PROMPT.index('只输出一个JSON数组：')
FOUR_REWRITE_PROMPT = REWRITE_PROMPT[:start] + '只输出完整JSON数组。每集必须且只能包含episode_id、title、core_plot、roles四个字段，集号从1到{{Total}}连续：\n' + EXAMPLE


def validate_four_fields(episodes, start, end):
    if not isinstance(episodes, list) or len(episodes) != end - start + 1:
        raise ValueError(f'必须完整输出第{start}—{end}集')
    for expected, episode in zip(range(start, end + 1), episodes):
        if not isinstance(episode, dict) or set(episode) != FIELDS:
            raise ValueError(f'第{expected}集必须且只能包含episode_id、title、core_plot、roles')
        if type(episode['episode_id']) is not int or episode['episode_id'] != expected:
            raise ValueError(f'集号必须连续，当前应为第{expected}集')
        for key in ('title', 'core_plot'):
            if not isinstance(episode[key], str) or not episode[key].strip():
                raise ValueError(f'第{expected}集{key}必须为非空文字')
        roles = episode['roles']
        if not isinstance(roles, list) or not roles or any(not isinstance(r, str) or not r.strip() for r in roles):
            raise ValueError(f'第{expected}集roles必须为非空人物姓名列表')


async def review_four_fields(ctx, episodes, story, plan, common):
    return await review_episode_outlines(ctx, episodes, story, plan, common,
        batch_validator=validate_four_fields, review_prompt=FOUR_REVIEW_PROMPT,
        rewrite_prompt=FOUR_REWRITE_PROMPT)


async def generate_four_fields(ctx, story, total, common, unused_prompt):
    return await generate_stage_episode_outlines(ctx, story, total, common, EPISODE_PROMPT,
        batch_validator=validate_four_fields, review_callback=review_four_fields)


async def generate(out, original, background, total):
    for name, prompt in [('episode', EPISODE_PROMPT), ('review', FOUR_REVIEW_PROMPT), ('rewrite', FOUR_REWRITE_PROMPT)]:
        (out/f'{name}_prompt_template.txt').write_text(prompt, encoding='utf-8')
    common = dict(background, RoleDescription='你是一位擅长叙事一致性与短剧节奏的专业编剧。', DramaType='短剧漫剧')
    common.setdefault('Reference', '')
    base.atomic_json(out/'03_story_outline/outline.json', original)
    base.atomic_json(out/'generation_background.json', common)
    print('直接使用原始详细大纲，开始阶段规划、四字段逐集生成与三轮审校择优。', flush=True)
    result = await generate_four_fields(base.Context(), base.transfer_outline_to_str(original), total, common, EPISODE_PROMPT)
    base.atomic_json(out/'episode_outlines.json', result)
    return result


if __name__ == '__main__':
    base.main(generate_fn=generate, output_prefix='qwen36_27b_four_fields_demo_', polish=False)
