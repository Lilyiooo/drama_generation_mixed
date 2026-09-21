"""Local schema for creation inputs/results, independent of Tencent protobuf.

Only fields used by this pipeline are defined. Unknown constructor fields fail
explicitly. Nested values and lists are copied on construction, like the original
message containers, so a downstream step cannot mutate a previous result.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from enum import IntEnum
import json
from typing import get_args, get_origin, get_type_hints


class PlotType(IntEnum):
    SHORT = 1
    CARTOON = 2
    LONG = 3
    SHORT_CARTOON = 4


class Model:
    def __post_init__(self):
        for item in fields(self):
            setattr(self, item.name, deepcopy(getattr(self, item.name)))

    def CopyFrom(self, other):
        if type(self) is not type(other):
            raise TypeError(f"Cannot copy {type(other).__name__} into {type(self).__name__}")
        for item in fields(self):
            setattr(self, item.name, deepcopy(getattr(other, item.name)))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        hints = get_type_hints(cls)
        def convert(kind, value):
            if get_origin(kind) is list:
                return [convert(get_args(kind)[0], item) for item in value]
            if isinstance(kind, type) and issubclass(kind, Model):
                return kind.from_dict(value)
            return value
        return cls(**{key: convert(hints[key], value) for key, value in data.items()})

    def __str__(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class StoryInfo(Model):
    story_id: str = ""
    plot_type: int = 0

@dataclass
class GenerateInputCommon(Model):
    episode_nums: int = 0
    season_nums: int = 0
    min_word_count_per_episode: int = 0
    max_word_count_per_episode: int = 0

@dataclass
class GenerateInputByCreativity(Model):
    summary_memory_baseline: bool = False
    disable_plot_retrieval: bool = True
    common: GenerateInputCommon = field(default_factory=GenerateInputCommon)
    core_story: str = ""
    reference: str = ""
    role_setting: str = ""
    topic: str = ""
    world_view: str = ""

@dataclass
class StoryOutline(Model):
    role_info: list[str] = field(default_factory=list)
    story_outline: list[str] = field(default_factory=list)
    world_building: list[str] = field(default_factory=list)

@dataclass
class Chapter(Model):
    chapter_num: int = 0
    content: str = ""

@dataclass
class PlotPlanning(Model):
    description: str = ""
    chapter_from: int = 0
    chapter_to: int = 0
    chapters: list[Chapter] = field(default_factory=list)
    plot_id: int = 0
    title: str = ""

@dataclass
class Point(Model):
    end_plot_id: int = 0
    point_id: int = 0
    point_name: str = ""
    start_plot_id: int = 0

@dataclass
class PointPlanning(Model):
    description: str = ""
    points: list[Point] = field(default_factory=list)

@dataclass
class DetailedStoryLine(Model):
    start_chapter_id: int = 0
    end_chapter_id: int = 0
    title: str = ""
    content: str = ""


@dataclass
class Storyline(Model):
    name: str = ""
    description: str = ""
    detailed_story_lines: list[DetailedStoryLine] = field(default_factory=list)


@dataclass
class SingleScriptProposal(Model):
    other_adaptation_proposal: str = ""
    plot_planning: list[PlotPlanning] = field(default_factory=list)
    point_planning: PointPlanning = field(default_factory=PointPlanning)
    season_id: int = 0
    storylines: list[Storyline] = field(default_factory=list)
    title: str = ""

@dataclass
class ScriptProposal(Model):
    script_proposals: list[SingleScriptProposal] = field(default_factory=list)

@dataclass
class OutlineEpisode(Model):
    chapter_from: int = 0
    chapter_to: int = 0
    content: str = ""
    episode_id: int = 0
    season_id: int = 0

@dataclass
class OutlineSeason(Model):
    episodes: list[OutlineEpisode] = field(default_factory=list)
    season_id: int = 0

@dataclass
class EpisodeOutline(Model):
    seasons: list[OutlineSeason] = field(default_factory=list)

@dataclass
class DramaEpisode(Model):
    episode_id: int = 0
    script_content: str = ""
    season_id: int = 0

@dataclass
class DramaSeason(Model):
    episodes: list[DramaEpisode] = field(default_factory=list)
    season_id: int = 0

@dataclass
class Drama(Model):
    seasons: list[DramaSeason] = field(default_factory=list)

@dataclass
class SelectRange(Model):
    episode_ids: list[int] = field(default_factory=list)
    season_id: int = 0

@dataclass
class GenerateDrama(Model):
    episode_outline: EpisodeOutline = field(default_factory=EpisodeOutline)
    prev_episode_content: str = ""
    script_proposal: ScriptProposal = field(default_factory=ScriptProposal)
    select_range: list[SelectRange] = field(default_factory=list)
    story_outline: StoryOutline = field(default_factory=StoryOutline)
    suggestion: str = ""

@dataclass
class GeneratedData(Model):
    episode_outline: EpisodeOutline = field(default_factory=EpisodeOutline)
    script_proposal: ScriptProposal = field(default_factory=ScriptProposal)
    story_outline: StoryOutline = field(default_factory=StoryOutline)

@dataclass
class GenerateStoryOutline(Model):
    script_proposal: ScriptProposal = field(default_factory=ScriptProposal)

@dataclass
class RegenerateData(Model):
    generate_data: GeneratedData = field(default_factory=GeneratedData)
    story_outline: StoryOutline = field(default_factory=StoryOutline)
    script_proposal: ScriptProposal = field(default_factory=ScriptProposal)
    suggestion: str = ""

@dataclass
class GenerateScriptProposalByCreativityReq(Model):
    generate_input: GenerateInputByCreativity = field(default_factory=GenerateInputByCreativity)
    story_info: StoryInfo = field(default_factory=StoryInfo)

@dataclass
class GenerateStoryOutlineByCreativityReq(Model):
    generate_data: GenerateStoryOutline = field(default_factory=GenerateStoryOutline)
    generate_input: GenerateInputByCreativity = field(default_factory=GenerateInputByCreativity)
    story_info: StoryInfo = field(default_factory=StoryInfo)

@dataclass
class GenerateEpisodeOutlineByCreativityReq(Model):
    generate_input: GenerateInputByCreativity = field(default_factory=GenerateInputByCreativity)
    generated_data: GeneratedData = field(default_factory=GeneratedData)
    story_info: StoryInfo = field(default_factory=StoryInfo)

@dataclass
class GenerateDramaByCreativityReq(Model):
    generate_data: GenerateDrama = field(default_factory=GenerateDrama)
    generate_input: GenerateInputByCreativity = field(default_factory=GenerateInputByCreativity)
    story_info: StoryInfo = field(default_factory=StoryInfo)

@dataclass
class RegenerateEpisodeOutlineSingleByCreativityReq(Model):
    episode_id: int = 0
    episode_outline: str = ""
    generate_input: GenerateInputByCreativity = field(default_factory=GenerateInputByCreativity)
    generated_data: GeneratedData = field(default_factory=GeneratedData)
    story_info: StoryInfo = field(default_factory=StoryInfo)
    suggestion: str = ""

@dataclass
class RegenerateStoryOutlineByCreativityReq(Model):
    generate_input: GenerateInputByCreativity = field(default_factory=GenerateInputByCreativity)
    story_info: StoryInfo = field(default_factory=StoryInfo)
    regenerate_data: RegenerateData = field(default_factory=RegenerateData)

@dataclass
class RegenerateScriptProposalByCreativityReq(Model):
    generate_input: GenerateInputByCreativity = field(default_factory=GenerateInputByCreativity)
    story_info: StoryInfo = field(default_factory=StoryInfo)
    regenerate_data: RegenerateData = field(default_factory=RegenerateData)

@dataclass
class RegenerateEpisodeOutlineAllByCreativityReq(Model):
    generate_input: GenerateInputByCreativity = field(default_factory=GenerateInputByCreativity)
    story_info: StoryInfo = field(default_factory=StoryInfo)
    regenerate_data: RegenerateData = field(default_factory=RegenerateData)
    generated_data: GeneratedData = field(default_factory=GeneratedData)

@dataclass
class GenerateScriptProposalRsp(Model):
    script_proposal: ScriptProposal = field(default_factory=ScriptProposal)
    story_outline: StoryOutline = field(default_factory=StoryOutline)

@dataclass
class GenerateStoryOutlineRsp(Model):
    story_outline: StoryOutline = field(default_factory=StoryOutline)

@dataclass
class GenerateEpisodeOutlineRsp(Model):
    episode_outline: EpisodeOutline = field(default_factory=EpisodeOutline)

@dataclass
class GenerateDramaRsp(Model):
    result: Drama = field(default_factory=Drama)
    episode_outline: EpisodeOutline = field(default_factory=EpisodeOutline)

@dataclass
class RegenerateEpisodeOutlineSingleRsp(Model):
    episode_outline: str = ""

@dataclass
class RegenerateStoryOutlineRsp(Model):
    result: str = ""

# Preserve the nested names used by the existing algorithm.
SingleScriptProposal.PlotPlanning = PlotPlanning
SingleScriptProposal.PointPlanning = PointPlanning
PlotPlanning.Chapter = Chapter
PointPlanning.Point = Point
EpisodeOutline.Episode = OutlineEpisode
EpisodeOutline.Season = OutlineSeason
Drama.Episode = DramaEpisode
Drama.Season = DramaSeason
GenerateDrama.SelectRange = SelectRange
