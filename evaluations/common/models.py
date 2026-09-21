from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Scene:
    scene_id: str
    episode_id: int
    scene_number: int
    heading: str
    text: str
    source_file: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SceneAnnotation:
    scene_id: str
    episode_id: int
    scene_number: int
    goal: str
    obstacle: str
    action: str
    turn: str
    outcome: str
    hook: str
    state_change: str
    narrative_functions: list[str]
    annotation_method: str
    source_text: str = ""

    @property
    def structural_summary(self) -> str:
        return "；".join(
            [
                f"目标：{self.goal}",
                f"阻碍：{self.obstacle}",
                f"行动：{self.action}",
                f"转折：{self.turn}",
                f"结果：{self.outcome}",
                f"状态变化：{self.state_change}",
                f"钩子：{self.hook}",
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["structural_summary"] = self.structural_summary
        return result
