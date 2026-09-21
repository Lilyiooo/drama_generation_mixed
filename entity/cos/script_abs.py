import cos_orm
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from trpc import client as tclient
from trpc import context
from trpc_cos import new_client as Client

@dataclass_json
@dataclass
class ScriptAbsFile(cos_orm.BaseCosModel):
    episodeAbs: list[str]

    @classmethod
    def get_path(cls, project_id) -> str:
        return f"drama_operation/creativity/{project_id}/script_abs.json"
