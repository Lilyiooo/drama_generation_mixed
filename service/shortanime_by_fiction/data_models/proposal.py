"""
models/proposal.py

改编方案相关数据模型：
  - FictionConfig      — 配置（从 configs 重新导出）
  - Adaptation         — 整体改编方案（COS 模型）
  - Plot               — 情节单元
  - Point              — 卡点
  - PointPlanning      — 卡点规划
  - DetailedStoryline  — 详细故事线
  - StoryLine          — 故事线
  - SeasonProposal     — 分季/季改编方案（COS 模型）
  - RoleInfo           — 角色信息（COS 模型）
  - WorldBuilding      — 世界观设定（COS 模型）
"""

from dataclasses import dataclass, field
from typing import List

from dataclasses_json import dataclass_json

from service.shortanime_by_fiction.configs import BaseCosModel, FictionConfig
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
class Adaptation(BaseCosModel):
    content: str = "暂无"

    @classmethod
    def get_path(cls, story_info: StoryInfo) -> str:
        base_dir = _plot_type_to_dir(story_info.plot_type)
        return f"{base_dir}/{story_info.story_id}/adaptation.json"


@dataclass_json
@dataclass
class Plot:
    plot_id: int = 0
    chapter_from: int = 0
    chapter_to: int = 0
    title: str = ""
    description: str = ""
    status: str = ""


@dataclass_json
@dataclass
class Point:
    point_id: int = 0
    point_name: str = ""
    start_plot_id: int = 0
    end_plot_id: int = 0


@dataclass_json
@dataclass
class PointPlanning:
    points: List[Point] = field(default_factory=list)
    description: str = ""


@dataclass_json
@dataclass
class DetailedStoryline:
    start_chapter_id: int = 0
    end_chapter_id: int = 0
    title: str = ""
    content: str = ""


@dataclass_json
@dataclass
class StoryLine:
    name: str = ""
    description: str = ""
    status: str = ""
    detailed_story_lines: List[DetailedStoryline] = field(default_factory=list)


@dataclass_json
@dataclass
class SeasonProposal(BaseCosModel):
    title: str = ""
    season_id: int = -1
    plot_planning: List[Plot] = field(default_factory=list)
    point_planning: PointPlanning = field(default_factory=PointPlanning)

    storylines: List[StoryLine] = field(default_factory=list)
    other_adaptation_proposal: str = ""

    @classmethod
    def get_path(cls, story_info: StoryInfo, season_id=None) -> str:
        base_dir = _plot_type_to_dir(story_info.plot_type)
        if season_id is None:
            return f"{base_dir}/{story_info.story_id}/season_proposal.json"
        else:
            season_id = int(season_id) + 1
            return f"{base_dir}/{story_info.story_id}/SE{season_id}/season_proposal.json"

    def get_plot_text(self) -> str:
        text = ""
        for plot in self.plot_planning:
            text += "\n\n" + f"PLOT_{plot.plot_id} (§{plot.chapter_from}-§{plot.chapter_to}) [{plot.title}]\n{plot.description}"
        return text.strip()

    @staticmethod
    def get_plots_and_content(season_division: "SeasonProposal", start_plot_id: int, end_plot_id: int) -> tuple[list[dict], str]:
        plots = [p for p in season_division.plot_planning if start_plot_id <= p.plot_id <= end_plot_id]
        plot_list = [dict(p.to_dict(), plot_id=i) for i, p in enumerate(plots)]
        content = "\n\n".join(f"PLOT_{i} (§{p.chapter_from}-§{p.chapter_to}) [{p.title}]\n{p.description}" for i, p in enumerate(plots))
        return plot_list, content

    def to_proposal_text(self) -> str:
        text = ""
        # 卡点规划
        if self.point_planning.description != "":
            text += f"\n\n### 卡点规划\n" + self.point_planning.description

        # 故事线规划
        data_text = ""
        for storyline in self.storylines:
            data_text += f'\n- **{storyline.status}**故事线"{storyline.name}": {storyline.description}'
        if data_text != "":
            text += "\n\n### 故事线规划" + data_text

        # 其他改编规划
        if self.other_adaptation_proposal != "":
            text += f"\n\n### 其他改编规划\n" + self.other_adaptation_proposal
        return text


@dataclass_json
@dataclass
class RoleInfo(BaseCosModel):
    content: str = "暂无"

    @classmethod
    def get_path(cls, story_info: StoryInfo) -> str:
        base_dir = _plot_type_to_dir(story_info.plot_type)
        return f"{base_dir}/{story_info.story_id}/role_info.json"


@dataclass_json
@dataclass
class WorldBuilding(BaseCosModel):
    content: str = "暂无"

    @classmethod
    def get_path(cls, story_info: StoryInfo) -> str:
        base_dir = _plot_type_to_dir(story_info.plot_type)
        return f"{base_dir}/{story_info.story_id}/world_building.json"


# Re-export FictionConfig for convenience
__all__ = [
    "FictionConfig",
    "Adaptation",
    "Plot",
    "Point",
    "PointPlanning",
    "DetailedStoryline",
    "StoryLine",
    "SeasonProposal",
    "RoleInfo",
    "WorldBuilding",
]
