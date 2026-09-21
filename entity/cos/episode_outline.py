import cos_orm
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from trpc import client as tclient
from trpc import context
from trpc_cos import new_client as Client

@dataclass_json
@dataclass
class EpisodeOutlineFile(cos_orm.BaseCosModel):
    episodeOutline: str  # 集大纲内容

    @classmethod
    def get_path(cls, project_id, episode_id) -> str:
        return f"drama_operation/creativity/{project_id}/episode_outline/episode_{episode_id}.json"
