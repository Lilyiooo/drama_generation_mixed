"""
短剧创作Prompt模板集合

按类型拆分为以下模块：
- outline: 故事大纲相关prompt
- role: 角色设定相关prompt
- episode_outline: 集大纲相关prompt
- script: 剧本创作相关prompt
- proposal: 策划相关prompt（世界观、情节规划等）
- utils: 工具类prompt（JSON修正、润色、摘要等）
"""

from .outline import (
    GENERATE_OUTLINE_PROMPT,
    REGENERATE_OUTLINE_PROMPT,
    GENERATE_OUTLINE_BY_SCRIPT_PROMPT,
    REGENERATE_OUTLINE_BY_SCRIPT_PROMPT,
)

from .role import (
    GENERATE_ROLE_PROMPT,
    REGENERATE_ROLE_PROMPT,
    GENERATE_ROLE_SETTING_PROMPT,
    REGENERATE_ROLE_SETTING_PROMPT,
)

from .episode_outline import (
    GENERATE_ALL_EPISODE_OUTLINE_PROMPT,
    REGENERATE_ALL_EPISODE_OUTLINE_PROMPT,
    REGENERATE_SINGLE_EPISODE_OUTLINE_PROMPT,
    GENERATE_EPISODE_OUTLINE_BY_SCRIPT_PROMPT,
)

from .script import (
    GENERATE_SCENE_OUTLINE_PROMPT,
    GENERATE_SCENE_PLOT_PROMPT,
    GENERATE_WHOLE_EPISODE_PROMPT,
    ADJUST_WORD_COUNT_PROMPT,
)

from .script_format import (
    SCRIPT_FORMAT_PROMPT,
)

from .proposal import (
    GENERATE_WORLD_VIEW_PROMPT,
    REGENERATE_WORLD_VIEW_PROMPT,
    GENERATE_PLOT_POINT_PROMPT,
    REGENERATE_PLOT_POINT_PROMPT,
)

from .utils import (
    ADJUST_JSON_FORMAT_PROMPT,
    POLISH_PLOT_PROMPT,
    EXTRACT_ABS_PROMPT,
)
