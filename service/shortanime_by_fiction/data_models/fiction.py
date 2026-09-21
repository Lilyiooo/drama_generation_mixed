"""
models/fiction.py

小说原著相关数据模型：
  - StoryInfo       — 故事基本信息（story_id, story_name, plot_type）
  - GenerateInput   — 生成任务输入参数
  - NovelStructed   — 结构化小说数据（COS 模型）
  - NovelArcs       — 小说弧线数据（COS 模型）
  - StoryArcs       — 故事弧线数据（COS 模型）
"""

import os
import json

from dataclasses import dataclass, field
from typing import List, Dict, Union
from typing_extensions import Self

from dataclasses_json import dataclass_json

from service.shortanime_by_fiction.configs import BaseCosModel


@dataclass
class StoryInfo:
    story_id: str = ""
    story_name: str = ""
    plot_type: str = ""


@dataclass
class GenerateInput:
    novel_id: str = ""
    season_nums: str = ""
    episode_nums: str = ""
    min_word_count_per_episode: int = 0
    max_word_count_per_episode: int = 0
    instruction: str = ""


@dataclass_json
@dataclass
class NovelStructed(BaseCosModel):
    data: List[dict] = field(default_factory=list)

    @classmethod
    def get_path(cls, novel_id, novel_dir) -> str:
        return f"{novel_dir}/{novel_id}_structured.json"

    @classmethod
    def unmarshal(cls, data: bytes) -> Self:
        rsp = cls()
        data = json.loads(data)
        rsp.data = data
        return rsp

    def save_to_file(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fp:
            json.dump(self.data, fp, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load_from_file(cls, path: str):
        rsp = cls()
        with open(path) as fp:
            data = json.load(fp)
            rsp.data = data
        return rsp


@dataclass_json
@dataclass
class NovelArcs(BaseCosModel):
    data: Dict[str, Union[str, List[dict]]] = field(default_factory=dict)

    @classmethod
    def get_path(cls, novel_id, novel_dir) -> str:
        return f"{novel_dir}/{novel_id}_arc.json"

    @classmethod
    def unmarshal(cls, data: bytes) -> Self:
        rsp = cls()
        data = json.loads(data)
        rsp.data = data
        return rsp

    def save_to_file(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fp:
            json.dump(self.data, fp, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load_from_file(cls, path: str):
        rsp = cls()
        with open(path) as fp:
            data = json.load(fp)
            rsp.data = data
        return rsp


@dataclass_json
@dataclass
class StoryArcs(BaseCosModel):
    story_arcs: List[dict] = field(default_factory=list)

    @classmethod
    def get_path(cls, novel_id) -> str:
        return f"drama_operation/short_anime/novel/{novel_id}_arc.json"

    def save_to_file(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fp:
            json.dump(self.story_arcs, fp, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load_from_file(cls, path: str):
        rsp = cls()
        with open(path) as fp:
            rsp.story_arcs = json.load(fp)
        return rsp
