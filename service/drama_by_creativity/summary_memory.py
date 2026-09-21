"""Opt-in baseline: one synopsis per completed episode, no structured memory."""
import json
import os
from pathlib import Path
import uuid

import jinja2
from drama_local.runtime import LocalStore, atomic_json
from .local_llm import LLM
from .utils import get_model_config

SUMMARIZE_EPISODE_PROMPT = """请为下面已经完成的第 {{EpisodeNumber}} 集剧本写一份剧情梗概，供后续集创作时作为情节记忆。
按事件发生顺序概括实际发生的剧情，保留人物行为及动机、关键因果、人物关系变化、重要物品与线索、结尾悬念。
只依据本集最终剧本，不预测未来，不编造剧情，不建立需求树或世界状态表。
建议300～500字，关键情节较多时可适当增加。只输出中文梗概正文，不输出JSON或其他说明。

## 本集最终剧本
{{EpisodeScript}}
"""


def summary_baseline_enabled(generate_input=None):
    return bool(getattr(generate_input, 'summary_memory_baseline', False)) or os.getenv(
        'DRAMA_SUMMARY_MEMORY_BASELINE','').strip().lower() in ('1','true','yes','on')


def memory_prompt(prompt, baseline):
    if not baseline:
        return prompt
    prompt = prompt.replace('## 前五集摘要\n{{RecentEpisodeSummaries}}\n\n---\n\n', '')
    return prompt.replace('## 与当前集内容具有叙事相关性的已发生事件、细节、伏笔、线索',
                          '## 前面各集的剧情梗概（情节记忆）')


class SummaryMemory:
    def __init__(self, story_id):
        store = LocalStore()
        store.path(story_id, 'check')  # Validate the story ID.
        self.story_dir = store.root / 'summary_memory' / story_id
        self.path = self.story_dir / 'memory_state.json'

    def _load(self):
        if not self.path.exists():
            return {'last_updated_episode':-1,'episode_summaries':[]}
        return json.loads(self.path.read_text(encoding='utf-8'))

    def retrieve(self, episode_id):
        summaries = sorted((x for x in self._load()['episode_summaries'] if x['episode_id'] < episode_id),
                           key=lambda x:x['episode_id'])
        if [x['episode_id'] for x in summaries] != list(range(episode_id)):
            raise ValueError(f'梗概记忆不连续，无法生成第{episode_id+1}集；请先补齐前面的剧本和梗概')
        return '\n\n'.join(f"第{x['episode_id']+1}集梗概：\n{x['summary']}" for x in summaries) or '暂无（第一集）'

    async def update_from_episode(self, ctx, episode_id, final_script):
        config = get_model_config('auxiliary_model')
        client = LLM(**config, template=jinja2.Template(SUMMARIZE_EPISODE_PROMPT))
        # Failure stops baseline progression; never generate later episodes with missing memory.
        summary = (await client.request(ctx, {'EpisodeNumber':str(episode_id+1),
                   'EpisodeScript':final_script}, uuid.uuid4().hex)).response
        if not isinstance(summary,str) or not summary.strip():
            raise ValueError(f'第{episode_id+1}集梗概为空')
        entries = [x for x in self._load()['episode_summaries'] if x['episode_id'] < episode_id]
        entries.append({'episode_id':episode_id,'summary':summary.strip()})
        atomic_json(self.path, {'mode':'summary_memory_baseline','last_updated_episode':episode_id,
                               'episode_summaries':entries})
        atomic_json(self.story_dir / f'episode_{episode_id+1:03d}.json', entries[-1])
        return summary
