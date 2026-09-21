REVIEW_EPISODE_OUTLINES_PROMPT = """
{{RoleDescription}}
你是全剧分集大纲的审校编辑。请通读全部集大纲，检查情节逻辑冲突、叙事一致性与相邻集衔接，并给出需要替换的完整集大纲条目。

题材：{{Topic}}
世界观：{{WorldView}}
核心故事：{{StoryOutline}}
人物设定：{{RoleInfo}}
{% if Reference %}创作参考：{{Reference}}
{% endif %}全剧总集数：{{TotalEpisodeNums}}

待检查的完整逐集大纲（episode_id为从1开始的全剧集号）：
{{EpisodeOutlinesForReview}}

审校要求：
1. 对照全剧检查人物身份、人物或世界状态与关系的连续性，动机与知情状态、事件起因与结果、时间地点、重要道具和前序事件结果的承接。检查事项包括但不限于上述内容，重点发现具体矛盾、重复发生的事件及不自然的衔接。
2. 结合故事背景、核心故事、人物设定与全部逐集大纲判断问题，作最小必要修正并说明依据，保留核心故事、主要事件与人物弧光。
3. 仅替换解决问题所必需的集大纲；有跨集影响时一并修正关联集，使修改后的全剧自洽。保留有效剧情信息与重要线索，对关键铺垫和后续结果检查对应关系，兼顾整体节奏。
4. 总集数和episode_id保持不变。每个替换条目提供完整字段，保持中文和原字段结构。无需修改的集不要输出。
5. 无需修改时，has_changes为false，replacements为空数组，并简要给出review_summary；需要修改时has_changes为true，逐条提供问题依据、关联集数和完整替换条目。
{% if RetryFeedback %}上次结构校验未通过，请纠正：{{RetryFeedback}}{% endif %}

输出一个JSON对象，格式如下（数值和文字为示意，实际填写）：
{
  "has_changes": true,
  "review_summary": "全剧检查结论与主要修改说明",
  "replacements": [
    {
      "episode_id": 1,
      "related_episode_ids": [1, 2],
      "reason": "指出原大纲的具体矛盾、涉及的事件及修正方式",
      "replacement": {
        "episode_id": 1,
        "title": "本集标题",
        "core_plot": "本集完整核心剧情，200字左右",
        "roles": ["人物姓名"],
        "highlights": "本集看点和名场面",
        "character_growth": "人物成长变化",
        "relationship_changes": "人物关系变化",
        "main_storyline_progression": "主线推进",
        "ending_hook": "本集结尾钩子"
      }
    }
  ]
}
"""
