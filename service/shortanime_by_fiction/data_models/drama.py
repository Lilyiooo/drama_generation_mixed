"""
models/drama.py

剧本生产相关数据模型：
  - Act             — 单幕结构描述
  - ActOutline      — 幕大纲（COS 模型）
  - StoryOutline    — 季故事大纲（COS 模型）
  - EpisodeOutline  — 单集分集大纲（COS 模型）
  - EpisodeScript   — 单集剧本（COS 模型）
  - EvalReport      — 质量评估报告（COS 模型）
"""

import re

from dataclasses import dataclass, field
from typing import List

from dataclasses_json import dataclass_json

from service.shortanime_by_fiction.configs import BaseCosModel
from service.shortanime_by_fiction.data_models.fiction import StoryInfo


def _plot_type_to_dir(plot_type: str) -> str:
    if plot_type == "SHORT":
        return "drama_operation/short_drama"
    elif plot_type == "SHORT_CARTOON":
        return "drama_operation/short_anime"
    else:
        raise ValueError(f"Invalid plot_type: {plot_type}")


@dataclass_json
@dataclass
class Act:
    act_id: int = 0
    act_title: str = ""
    episode_start: int = 0
    episode_end: int = 0
    description: str = ""
    plots: List[dict] = field(default_factory=list)
    adaptation_logic: str = ""
    ending: str = ""


@dataclass_json
@dataclass
class ActOutline(BaseCosModel):
    acts: List[Act] = field(default_factory=list)

    @classmethod
    def get_path(cls, story_info: StoryInfo, season_id) -> str:
        base_dir = _plot_type_to_dir(story_info.plot_type)
        season_id = int(season_id) + 1
        return f"{base_dir}/{story_info.story_id}/SE{season_id}/act_outline.json"


@dataclass_json
@dataclass
class StoryOutline(BaseCosModel):
    content: str = ""

    @classmethod
    def get_path(cls, story_info: StoryInfo, season_id) -> str:
        base_dir = _plot_type_to_dir(story_info.plot_type)
        season_id = int(season_id) + 1
        return f"{base_dir}/{story_info.story_id}/SE{season_id}/story_outline.json"


@dataclass_json
@dataclass
class EpisodeOutline(BaseCosModel):
    season_id: int = 0
    act_id: int = 0
    episode_id: int = 0
    ch_ranges: List[int] = field(default_factory=list)
    chapter_from: int = 0
    chapter_to: int = 0
    chapter_range: List[str] = field(default_factory=list)
    content: str = ""
    episode_title: str = ""
    events: List[str] = field(default_factory=list)
    ending_hook: str = ""

    @classmethod
    def get_path(cls, story_info: StoryInfo, season_id, episode_id) -> str:
        base_dir = _plot_type_to_dir(story_info.plot_type)
        season_id = int(season_id) + 1
        episode_id = int(episode_id) + 1
        return f"{base_dir}/{story_info.story_id}/SE{season_id}/episode_outline_EP{episode_id}.json"

    @classmethod
    def to_text(cls, episode_id, chapter_range, synopsis, events, ending_hook):
        text = f"第{episode_id+1}集:"
        text += f"\n【章节范围】: {chapter_range}"
        text += f"\n【剧情梗概】{synopsis}"
        text += f"\n【事件列表】{events}"
        text += f"\n【结尾钩子】{ending_hook}"
        return text

    def sysnopsis_to_text(self):
        synopsis = re.search(r"【剧情梗概】(.*?)\n【事件列表】", self.content, re.DOTALL)
        if synopsis:
            return f"第{self.episode_id+1}集:" + synopsis.group(1).strip()
        return self.content


@dataclass_json
@dataclass
class EpisodeScript(BaseCosModel):
    season_id: int = 0
    episode_id: int = 0
    script_content: str = "暂无"
    word_tier: str = ""      # 字数档位，如"标准模式"/"加强模式"/"丰富模式"
    tier_reason: str = ""    # 选择该档位的理由

    @classmethod
    def get_path(cls, story_info: StoryInfo, season_id, episode_id) -> str:
        base_dir = _plot_type_to_dir(story_info.plot_type)
        season_id = int(season_id) + 1
        episode_id = int(episode_id) + 1
        return f"{base_dir}/{story_info.story_id}/SE{season_id}/episode_script_EP{episode_id}.json"


@dataclass_json
@dataclass
class EvalReport(BaseCosModel):
    report: str = ""
    suggestion: str = ""
    score: float = 0.0

    @classmethod
    def get_path(cls, story_info: StoryInfo, season_id, save_name) -> str:
        base_dir = _plot_type_to_dir(story_info.plot_type)
        season_id = int(season_id) + 1
        return f"{base_dir}/{story_info.story_id}/SE{season_id}/eval_{save_name}.json"
