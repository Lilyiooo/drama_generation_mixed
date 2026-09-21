import os
import json

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
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
    chapter_block_cnt: int = 200
    overlap_chapter_cnt: int = 10
    arc_range: str = "1~200章"
    max_workers: int = 1
    max_word_cnt: int = 80000
    retry_cnt: int = 3
    text_limit: int = 50000

    def __init__(self, yaml_path: str = None):
        super().__init__()
        self._update_from_yaml(yaml_path)

    def _update_from_yaml(self, yaml_path: str):
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

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """类方法：从yaml文件创建配置实例"""
        instance = cls()
        instance._update_from_yaml(yaml_path)
        return instance


@dataclass_json
@dataclass
class NovelStructed(BaseCosModel):
    data: List[dict] = field(default_factory=list)

    @classmethod
    def get_path(cls, novel_id) -> str:
        return f"ip_gpt/formal/novel_emb/1.0/{novel_id}_structured.json"

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
    block_arcs: List[str] = field(default_factory=list)
    story_arc: str = ""

    @classmethod
    def get_path(cls, novel_id, ver_id="1") -> str:
        return f"drama_operation/short/{novel_id}/story_outline_v{ver_id}.json"


@dataclass_json
@dataclass
class ShortDramaGenData(BaseCosModel):
    season_title: str = ""
    chapter_range: List[int] = field(default_factory=list)
    core_content: str = ""
    theme: str = ""
    emotional_curve: str = ""
    ending_cliffhanger: str = ""

    @classmethod
    def get_path(cls, story_id, season_id, ver_id="1") -> str:
        return f"drama_operation/short/{story_id}/ShortDramaGen_SE{season_id}_v{ver_id}.json"



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
    def get_path(cls, story_id, season_id, ver_id="1") -> str:
        return f"drama_operation/short/{story_id}/story_outline_SE{season_id}_v{ver_id}.json"

    def format_str(self) -> str:
        if len(self.chapter_range) == 2:
            chapter_str = f"§{self.chapter_range[0]} - §{self.chapter_range[1]}"
        else:
            chapter_str = "未指定"

        result = f"""
**- 标题：** {self.season_title}
**- 章节范围：** {chapter_str}
**- 核心内容：**
  {self.core_content}

**- 本季主题：** {self.theme}
**- 情绪曲线：** {self.emotional_curve}
**- 结尾悬念：**
  {self.ending_cliffhanger}
"""
        return result.strip()


@dataclass_json
@dataclass
class ShortDramaGenPartScript(BaseCosModel):
    episodes_script_raw: dict = field(default_factory=dict)
    episodes_script_fix1: dict = field(default_factory=dict)
    episodes_script_fix2: dict = field(default_factory=dict)
    episodes_script: dict = field(default_factory=dict)

    @classmethod
    def get_path(cls, story_id, partname, season_id=0, ver_id="1") -> str:
        return f"drama_operation/short/{story_id}/shortdrama_genscript{partname}_SE{season_id}_v{ver_id}.json"


@dataclass_json
@dataclass
class ShortDramaGenScripts(BaseCosModel):
    # not used
    scripts: List[ShortDramaGenPartScript] = field(default_factory=list)

    @classmethod
    def get_path(cls, story_id, season_id=0, ver_id="1") -> str:
        return f"drama_operation/short/{story_id}/shortdrama_genscripts_SE{season_id}_v{ver_id}.json"


@dataclass_json
@dataclass
class ShortDramaPartPlan(BaseCosModel):
    content: str = "暂无"
    #dict = field(default_factory=dict)
    # expand_summary: str = ""
    # part_summary: List[str] = field(default_fatory=list)
    # episode_ids: List[str] = field(default_fatory=list)

    @classmethod
    def get_path(cls, story_id, season_id=0, ver_id="1") -> str:
        return f"drama_operation/short/{story_id}/shortdrama_partplan_SE{season_id}_v{ver_id}.json"


@dataclass_json
@dataclass
class ShortDramaPartPlan0(BaseCosModel):
    # not used!
    content: str = ""

    @classmethod
    def get_path(cls, story_id, partname, ver_id="1") -> str:
        return f"drama_operation/short/{story_id}/shortdrama_partplan0_{partname}_v{ver_id}.json"


@dataclass_json
@dataclass
class ShortDramaPartsPlan(BaseCosModel):
    # not used!
    partplans: List[ShortDramaPartPlan] = field(default_factory=list)

    @classmethod
    def get_path(cls, story_id, season_id=0, ver_id="1") -> str:
        return f"drama_operation/short/{story_id}/shortdrama_partsplan_SE{season_id}_v{ver_id}.json"

@dataclass_json
@dataclass
class OnePartInfo:
    """part info each."""
    
    chaptersid: List[str] = field(default_factory=list)    
    expand_summary_list: List[str] = field(default_factory=list)
    refine_summary_list: List[str] = field(default_factory=list)
    episodes_ids: List[str] = field(default_factory=list)    
    episodes_script: List[str] = field(default_factory=list) 
    expand_summary: str = ""
    refine_summary: str = ""
    info: str = ""
    # epsid
    @classmethod
    def get_path(cls, story_id, partname, ver_id="1") -> str:
        return f"drama_operation/short/{story_id}/shortdrama_onepartinfo_{partname}_v{ver_id}.json"

@dataclass_json
@dataclass
class PartInfo:
    #part info dict.

    start: OnePartInfo = field(default_factory=OnePartInfo)
    develop: OnePartInfo = field(default_factory=OnePartInfo)
    hit: OnePartInfo = field(default_factory=OnePartInfo)
    end: OnePartInfo = field(default_factory=OnePartInfo)
    @classmethod
    def get_path(cls, story_id, ver_id="1") -> str:
        return f"drama_operation/short/{story_id}/shortdrama_allpartinfo_v{ver_id}.json"


@dataclass_json
@dataclass
class PartInfoFlat:
    #part info dict.

    start: str = "" 
    develop: str = ""
    hit: str = ""
    end: str = ""
    #cls_str: str = "暂无"

@dataclass_json
@dataclass
class ShortDramaInfo(BaseCosModel):
    fname: str = ""
    outline: PartInfo = field(default_factory=PartInfo) 
    heros_str: str = ""
    world_str: str = ""
    outline_str: str = ""
    outline_str_dict: dict = field(default_factory=dict)
    chapters: List[str] = field(default_factory=list)
    chapters_summary: List[str] = field(default_factory=list) 
    #text_len
    mainline: str = ""
    subline: str = ""
    full_story: str = ""
    outline_plan_str: str = ""
    outline_plan: PartInfoFlat = field(default_factory=PartInfoFlat)
    num_episode:   int = 0
    #text5w: str = "暂无"
    

    # {"起:{"环境规则"：x,"主角登场":x, 激励事件:x, 外部目标:x, 走出舒适区:x ,"集数规划":{"总集数建议":"xx","分集内容描述":xx}},  承：{初识红利:x, 伙伴爱情:x, 起始对手:x, 阻力对抗:x, 故事拐点:x,"集数规划":x}, 转：{调整目标:x, 确认对手:x, 冲突升级:x, 第一次决战:x, 遭受重挫:x,"集数规划":x}, 合：{疗伤反省:x, 解决方案:x, 牺牲成长:x, 最终决战:x, 回到原点:x,"集数规划":x}}

    @classmethod
    def get_path(cls, story_id, season_id=0, ver_id="1") -> str:
        print(f"drama_operation/short/{story_id}/shortdrama_info_SE{season_id}_v{ver_id}.json")
        return f"drama_operation/short/{story_id}/shortdrama_info_SE{season_id}_v{ver_id}.json"


@dataclass_json
@dataclass
class Heros(BaseCosModel):
    content: str = ""
    #content: dict = field(default_factory=dict)

    @classmethod
    def get_path(cls, story_id, season_id=0, ver_id="1") -> str:
        return f"drama_operation/short/{story_id}/heros_SE{season_id}_v{ver_id}.json"


@dataclass_json
@dataclass
class RoleInfo(BaseCosModel):
    content: str = ""

    @classmethod
    def get_path(cls, story_id, season_id=0, ver_id="1") -> str:
        return (
            f"drama_operation/short/{story_id}/role_info_SE{season_id}_v{ver_id}.json"
        )


@dataclass_json
@dataclass
class WorldBuilding(BaseCosModel):
    content: str = ""

    @classmethod
    def get_path(cls, story_id, season_id=0, ver_id="1") -> str:
        return f"drama_operation/short/{story_id}/world_building_SE{season_id}_v{ver_id}.json"


# =================================================================
# Enums
# =================================================================


class PlotType(Enum):
    """
    Corresponds to the PlotType enum in the Protobuf schema.
    """

    UNKNOWN = 0
    CARTOON = 1
    SHORT = 2
    LONG = 3


# =================================================================
# Core Data Structures
# =================================================================


@dataclass
class StoryInfo:
    """项目信息 (Project Information)"""

    story_id: Optional[str] = None
    story_name: Optional[str] = None
    plot_type: Optional[PlotType] = None


@dataclass
class StoryOutline:
    """故事大纲结构 (Story Outline Structure)"""

    # 每一季的故事大纲 (Story outline for each season)
    story_outline: List[str] = field(default_factory=list)
    # 每一季的人设 (Character info for each season)
    role_info: List[str] = field(default_factory=list)


@dataclass
class EpisodeOutline_Episode:
    """A single episode's outline details."""

    season_id: Optional[int] = None
    episode_id: Optional[int] = None
    chapter_from: Optional[int] = None
    chapter_to: Optional[int] = None
    content: Optional[str] = None

@dataclass_json
@dataclass
class EpisodeOutline_Season(BaseCosModel):
    """A season containing multiple episode outlines."""

    episodes: List[EpisodeOutline_Episode] = field(default_factory=list)
    season_id: Optional[int] = None
    
    @classmethod
    def get_path(cls, story_id, season_id, v_id='1') -> str:
        #season_id = int(season_id) + 1
        return f"drama_operation/short/{story_id}/episode_outline_SE{season_id}_v{v_id}.json"

@dataclass
class EpisodeOutline:
    """分集大纲结构 (Episode Outline Structure)"""

    seasons: List[EpisodeOutline_Season] = field(default_factory=list)


@dataclass
class Drama_Episode:
    """A single episode's script content."""

    season_id: Optional[int] = None
    episode_id: Optional[int] = None
    script_content: Optional[str] = None


@dataclass
class Drama_Season:
    """A season containing multiple full drama episodes."""

    episodes: List[Drama_Episode] = field(default_factory=list)
    season_id: Optional[int] = None


@dataclass
class Drama:
    """剧本结构 (Drama Script Structure)"""

    seasons: List[Drama_Season] = field(default_factory=list)


@dataclass
class GenerateInputByCreativity:
    """从创意生成剧本输入信息 (Input for generating a script from creativity)"""

    # 创意想法 (Core story idea)
    core_story: Optional[str] = None
    # 题材类型 (Topic/Genre)
    topic: Optional[str] = None
    # 世界观 (Worldview)
    world_view: Optional[str] = None
    # 人物设定 (Character settings)
    role_setting: Optional[str] = None
    # 参考套路 (Reference tropes/patterns)
    reference: Optional[str] = None


@dataclass
class GenerateInputByFiction:
    """从小说生成剧本输入信息 (Input for generating a script from a novel)"""

    # 小说id (Novel ID)
    novel_id: Optional[str] = None
    # 预计季数 (Projected number of seasons)
    season_nums: Optional[int] = None
    # 每季集数 (Number of episodes per season)
    episode_nums: Optional[int] = None
    # 分钟时长 (Duration in minutes)
    duration: Optional[int] = None


# =================================================================
# Request and Response Data Classes
# =================================================================


@dataclass
class GenerateStoryOutlineByCreativityReq:
    story_info: Optional[StoryInfo] = None
    generate_input: Optional[GenerateInputByCreativity] = None


@dataclass
class GenerateStoryOutlineByFictionReq:
    story_info: Optional[StoryInfo] = None
    generate_input: Optional[GenerateInputByFiction] = None


@dataclass
class GenerateStoryOutlineRsp:
    generate_input: Optional[StoryOutline] = None


@dataclass
class RegenerateStoryOutline:  # Note: Typo 'Regenerate' matches the Protobuf schema
    story_outline: Optional[StoryOutline] = None
    suggestion: Optional[str] = None


@dataclass
class RegenerateStoryOutlineByCreativityReq:
    story_info: Optional[StoryInfo] = None
    generate_input: Optional[GenerateInputByCreativity] = None
    regenerate_data: Optional[RegenerateStoryOutline] = None


@dataclass
class RegenerateStoryOutlineByFictionReq:
    story_info: Optional[StoryInfo] = None
    generate_input: Optional[GenerateInputByFiction] = None
    regenerate_data: Optional[RegenerateStoryOutline] = None
    season_id: List[int] = field(default_factory=list)


@dataclass
class RegenerateStoryOutlineRsp:
    result: Optional[str] = None


@dataclass
class GenerateEpisodeOutlineByCreativityReq:
    story_info: Optional[StoryInfo] = None
    generate_input: Optional[GenerateInputByCreativity] = None
    story_outline: Optional[StoryOutline] = None


@dataclass
class GenerateEpisodeOutlineByFictionReq:
    story_info: Optional[StoryInfo] = None
    generate_input: Optional[GenerateInputByFiction] = None
    story_outline: Optional[StoryOutline] = None


@dataclass
class GenerateEpisodeOutlineRsp:
    episode_outline: Optional[EpisodeOutline] = None


@dataclass
class RegenerateEpisodeOutline:
    story_outline: Optional[StoryOutline] = None
    episode_outline: Optional[EpisodeOutline] = None
    suggestion: Optional[str] = None


@dataclass
class RegenerateEpisodeOutlineAllByCreativityReq:
    story_info: Optional[StoryInfo] = None
    generate_input: Optional[GenerateInputByCreativity] = None
    regenerate_data: Optional[RegenerateEpisodeOutline] = None


@dataclass
class RegenerateEpisodeOutlineAllByFictionReq:
    story_info: Optional[StoryInfo] = None
    generate_input: Optional[GenerateInputByFiction] = None
    regenerate_data: Optional[RegenerateEpisodeOutline] = None


@dataclass
class RegenerateEpisodeOutlineSingleByCreativityReq:
    story_info: Optional[StoryInfo] = None
    generate_input: Optional[GenerateInputByCreativity] = None
    episode_outline: Optional[str] = None
    episode_id: Optional[int] = None
    suggestion: Optional[str] = None


@dataclass
class RegenerateEpisodeOutlineSingleByFictionReq:
    story_info: Optional[StoryInfo] = None
    generate_input: Optional[GenerateInputByFiction] = None
    episode_outline: Optional[str] = None
    season_id: Optional[int] = None
    episode_id: Optional[int] = None
    suggestion: Optional[str] = None


@dataclass
class RegenerateEpisodeOutlineSingleRsp:
    episode_outline: Optional[str] = None


@dataclass
class GenerateDrama_SelectRange:
    """Specifies a range of episodes to select."""

    season_id: Optional[int] = None
    episode_ids: List[int] = field(default_factory=list)


@dataclass
class GenerateDrama:
    story_outline: Optional[StoryOutline] = None
    # Note: Typo 'episdode' matches the Protobuf schema
    episode_outline: Optional[EpisodeOutline] = None
    prev_episode_content: Optional[str] = None
    suggestion: Optional[str] = None
    select_range: List[GenerateDrama_SelectRange] = field(default_factory=list)


@dataclass
class GenerateDramaByCreativityReq:
    story_info: Optional[StoryInfo] = None
    generate_input: Optional[GenerateInputByCreativity] = None
    generate_data: Optional[GenerateDrama] = None


@dataclass
class GenerateDramaByFictionReq:
    story_info: Optional[StoryInfo] = None
    generate_input: Optional[GenerateInputByFiction] = None
    generate_data: Optional[GenerateDrama] = None


@dataclass
class GenerateDramaRsp:
    result: Optional[Drama] = None
