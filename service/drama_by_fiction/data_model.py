import os
import json

from dataclasses import dataclass, field
from typing import Dict, List, Union
from typing_extensions import Self

import yaml
import cos_orm
from dataclasses_json import dataclass_json


class BaseCosModel(cos_orm.BaseCosModel):

    def marshal(self) -> bytes:
        if hasattr(self, "to_json"):
            rsp = self.to_json(ensure_ascii=False, indent=2)
            if isinstance(rsp, str):
                rsp = rsp.encode("utf-8")
            return rsp
        else:
            raise NotImplementedError()


class FictionConfig(BaseCosModel):
    model_name: str = "gemini-2.5-pro"
    model_type: str = "venus"

    # story_arc参数
    chapter_block_cnt: int = 50  # 每个区块的章节数
    overlap_chapter_cnt: int = 10  # 区块间的重叠章节数
    arc_range: str = "5～10章"  # 每个故事弧的目标章节范围
    min_arc_len: int = 4
    max_arc_len: int = 12
    fallback_len: int = 8

    max_workers: int = 4
    max_word_cnt: int = 80000
    turncat_word_cnt: int = 160000
    retry_cnt: int = 3
    novel_dirs: List[str] = ["ip_gpt/formal/novel_emb/1.0", "ip_gpt/test/novel_emb/1.0"]

    def __init__(self, yaml_path: str = None):
        super().__init__()
        self._update_from_yaml(yaml_path)

    def _update_from_yaml(self, yaml_path: str | None):
        """从yaml文件加载配置，使用hasattr动态更新字段"""
        if yaml_path is None:
            yaml_path = os.path.join(
                os.path.dirname(__file__), "../../trpc_python.yaml"
            )
        if not os.path.exists(yaml_path):
            return

        with open(yaml_path, "r", encoding="utf-8") as file:
            config_data = yaml.safe_load(file)

        if not config_data or "drama_fiction" not in config_data:
            return

        fiction_config = config_data["drama_fiction"]

        # 使用hasattr动态更新所有存在的字段
        for key, value in fiction_config.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass
class StoryInfo:
    story_id: str = ""
    story_name: str = ""
    plot_type: str = ""


@dataclass
class GenerateInput:
    novel_id: str = ""
    season_nums: int = 0
    episode_nums: int = 0
    duration: int = 20
    min_word_count_per_episode: int = 1000
    max_word_count_per_episode: int = 10000
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
    data: Dict[str, Union[str, List[dict]]] = field(default_factory=dict)

    @classmethod
    def get_path(cls, novel_id) -> str:
        return f"drama_operation/cartoon/novel/{novel_id}_arc.json"

    @classmethod
    def unmarshal(cls, data: bytes) -> Self:
        rsp = cls()
        data = json.loads(data)
        rsp.data = data
        return rsp

    def marshal(self) -> bytes:
        if hasattr(self, "to_json"):
            rsp = json.dumps(self.data, ensure_ascii=False, indent=2)
            if isinstance(rsp, str):
                rsp = rsp.encode("utf-8")
            return rsp
        else:
            raise NotImplementedError()

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
class Adaptation(BaseCosModel):
    content: str = "暂无"

    def __str__(self) -> str:
        return self.content

    @classmethod
    def get_path(cls, story_id: str) -> str:
        return f"drama_operation/cartoon/{story_id}/adaptation.json"


@dataclass_json
@dataclass
class RoleInfo(BaseCosModel):
    content: str = "暂无"

    def __str__(self) -> str:
        return self.content

    @classmethod
    def get_path(cls, story_id) -> str:
        return f"drama_operation/cartoon/{story_id}/role_info.json"


@dataclass_json
@dataclass
class WorldBuilding(BaseCosModel):
    content: str = "暂无"

    def __str__(self) -> str:
        return self.content

    @classmethod
    def get_path(cls, story_id) -> str:
        return f"drama_operation/cartoon/{story_id}/world_building.json"


@dataclass_json
@dataclass
class Chapter(BaseCosModel):
    chapter_num: int = 0
    content: str = ""


@dataclass_json
@dataclass
class Event(BaseCosModel):
    event_id: str = ""
    chapter_from: int = 0
    chapter_to: int = 0
    title: str = ""
    description: str = ""
    chapters: List[Chapter] = field(default_factory=list)


@dataclass_json
@dataclass
class Plot:
    plot_id: int = 0
    chapter_from: int = 0
    chapter_to: int = 0
    title: str = ""
    description: str = ""
    status: str = "保留"
    chapters: List[Chapter] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)

    def __str__(self) -> str:
        return f"PLOT_{self.plot_id} (§{self.chapter_from}-§{self.chapter_to}) [{self.title}]\n{self.description}"


@dataclass_json
@dataclass
class Point:
    point_id: int = 0
    point_name: str = ""
    start_plot_id: int = 0
    end_plot_id: int = 0


@dataclass_json
@dataclass
class DetailedStoryline:
    start_chapter_id: int = 0
    end_chapter_id: int = 0
    title: str = ""
    content: str = ""


@dataclass_json
@dataclass
class PointPlanning:
    points: List[Point] = field(default_factory=list)
    description: str = ""


@dataclass_json
@dataclass
class StoryLine:
    name: str = ""
    description: str = ""
    status: str = ""
    detailed_story_lines: List[DetailedStoryline] = field(default_factory=list)


@dataclass_json
@dataclass
class EventProposal(BaseCosModel):
    event_id: str = ""
    chapters: List[int] = field(default_factory=list)


@dataclass_json
@dataclass
class PlotProposal:
    plot_id: int = 0
    events: List[EventProposal] = field(default_factory=list)


@dataclass_json
@dataclass
class EpisodeProposal:
    episode_id: int = 0
    episode_title: str = ""
    plots: List[PlotProposal] = field(default_factory=list)

    def get_chapters(self) -> List[int]:
        """从分集规划中提取去重且排序后的章节列表"""
        chapters = []
        for plot in self.plots:
            for event in plot.events:
                chapters.extend(event.chapters)
        return sorted(set(chapters))


@dataclass_json
@dataclass
class SeasonProposal(BaseCosModel):
    title: str = ""
    season_id: int = -1
    plot_planning: List[Plot] = field(default_factory=list)
    point_planning: PointPlanning = field(default_factory=PointPlanning)
    episode_planning: List[EpisodeProposal] = field(default_factory=list)
    storylines: List[StoryLine] = field(default_factory=list)
    other_adaptation_proposal: str = ""

    @classmethod
    def get_path(cls, story_id: str, season_id=None) -> str:
        if season_id is None:
            return f"drama_operation/cartoon/{story_id}/season_proposal.json"
        else:
            se_id = int(season_id) + 1
            return f"drama_operation/cartoon/{story_id}/SE{se_id}/season_proposal.json"

    def get_plot_text(self) -> str:
        text = ""
        for plot in self.plot_planning:
            text += f"\n\nPLOT_{plot.plot_id} (§{plot.chapter_from}-§{plot.chapter_to}) [{plot.title}]\n{plot.description}"
        return text.strip()

    def to_proposal_text(self) -> str:
        delete_plot_text = "\n".join(
            f"- {plot.title} (第{plot.chapter_from}-{plot.chapter_to}章)"
            for plot in self.plot_planning
            if plot.status == "删除"
        )
        storylines_text = "\n".join(
            f"- **{s.status}**故事线 `{s.name}`: {s.description}"
            for s in self.storylines
        )

        return f"""
### 情节规划
#### 需要删除的情节:
{delete_plot_text}
### 卡点规划
{self.point_planning.description}
### 故事线规划
{storylines_text}
### 其他改编规划
{self.other_adaptation_proposal}""".strip()


@dataclass_json
@dataclass
class Scene(BaseCosModel):
    title: str = ""
    content: str = ""


@dataclass_json
@dataclass
class EpisodeOutline(BaseCosModel):
    season_id: int = 0
    episode_id: int = 0
    episode_title: str = ""
    ch_ranges: List[int] = field(default_factory=list)
    chapter_from: int = 0
    chapter_to: int = 0
    chapter_range: List[str] = field(default_factory=list)
    content: str = "暂无"
    scenes: List[Scene] = field(default_factory=list)

    @classmethod
    def get_path(cls, story_id, season_id, episode_id) -> str:
        season_id = int(season_id) + 1
        episode_id = int(episode_id) + 1
        return f"drama_operation/cartoon/{story_id}/SE{season_id}/outline_EP{episode_id:02d}.json"

    @classmethod
    def to_text(cls, episode_id, chapter_range, synopsis, events, ending_hook):
        text = f"第{episode_id+1}集:"
        text += f"\n【章节范围】: {chapter_range}"
        text += f"\n【剧情梗概】{synopsis}"
        text += f"\n【事件列表】{events}"
        text += f"\n【结尾钩子】{ending_hook}"
        return text


@dataclass_json
@dataclass
class SceneScript(BaseCosModel):
    scene_id: int = 0
    scene_title: str = ""
    script_content: str = ""


@dataclass_json
@dataclass
class EpisodeScript(BaseCosModel):
    season_id: int = 0
    episode_id: int = 0
    script_content: str = "暂无"
    scenes: List[SceneScript] = field(default_factory=list)

    @classmethod
    def get_path(cls, story_id, season_id, episode_id) -> str:
        season_id = int(season_id) + 1
        episode_id = int(episode_id) + 1
        return f"drama_operation/cartoon/{story_id}/SE{season_id}/script_EP{episode_id:02d}.json"


@dataclass_json
@dataclass
class StoryOutline(BaseCosModel):
    content: str = ""

    def __str__(self) -> str:
        return self.content

    @classmethod
    def get_path(cls, story_id: str, season_id) -> str:
        season_id = int(season_id) + 1
        return f"drama_operation/cartoon/{story_id}/SE{season_id}/story_outline.json"


###########################################################################


@dataclass_json
@dataclass
class SeasonOutline(BaseCosModel):
    season_title: str = ""
    chapter_range: List[int] = field(default_factory=list)
    core_content: str = ""
    theme: str = ""
    emotional_curve: str = ""
    ending_cliffhanger: str = ""

    @classmethod
    def get_path(cls, story_id, season_id) -> str:
        season_id = int(season_id) + 1
        return f"drama_operation/cartoon/{story_id}/story_outline_SE{season_id}.json"


@dataclass_json
@dataclass
class DramaOutline(BaseCosModel):
    total_seasons: int = 0
    episodes_per_season: int = 0
    division_basis: str = ""
    season_details: List[SeasonOutline] = field(default_factory=list)

    @classmethod
    def get_path(cls, story_id) -> str:
        return f"drama_operation/cartoon/{story_id}/drama_outline.json"
