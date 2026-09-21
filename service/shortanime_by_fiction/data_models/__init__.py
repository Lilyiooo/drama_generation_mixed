"""
models/__init__.py

统一导出所有模型，保持向后兼容性。
所有原来从 data_model 导入的类均可从此处导入。
"""

# fiction.py
from service.shortanime_by_fiction.data_models.fiction import (
    StoryInfo,
    GenerateInput,
    NovelStructed,
    NovelArcs,
    StoryArcs,
)

# drama.py
from service.shortanime_by_fiction.data_models.drama import (
    Act,
    ActOutline,
    StoryOutline,
    EpisodeOutline,
    EpisodeScript,
    EvalReport,
)

# proposal.py
from service.shortanime_by_fiction.data_models.proposal import (
    Adaptation,
    Plot,
    Point,
    PointPlanning,
    DetailedStoryline,
    StoryLine,
    SeasonProposal,
    RoleInfo,
    WorldBuilding,
)

# cpg.py
# from service.shortanime_by_fiction.data_models.cpg import (
#     CausalEvent,
#     CausalEdge,
#     CausalPlotGraph,
# )

# Re-export FictionConfig for convenience (originally lived in configs/)
from service.shortanime_by_fiction.configs import FictionConfig

__all__ = [
    # fiction
    "StoryInfo",
    "GenerateInput",
    "NovelStructed",
    "NovelArcs",
    "StoryArcs",
    # drama
    "Act",
    "ActOutline",
    "StoryOutline",
    "EpisodeOutline",
    "EpisodeScript",
    "EvalReport",
    # proposal
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
    # cpg
    "CausalEvent",
    "CausalEdge",
    "CausalPlotGraph",
]
