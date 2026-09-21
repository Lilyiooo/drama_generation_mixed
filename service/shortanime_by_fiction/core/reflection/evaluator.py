
from service.shortanime_by_fiction.data_manager import DataManager

from service.shortanime_by_fiction.llms import (
    query_llm,
    post_extract_json,
    Task
)
from service.shortanime_by_fiction.data_models import (
    FictionConfig,
    EvalReport
)

class Evaluator:

    def __init__(self, ctx):
        self.ctx = ctx
        self.data_manager = DataManager(ctx)
        self.cfg = FictionConfig()
    
    def evaluate_script(self, script):
        pass
    
    def _process_eval_act_outline(self, act_outline):
        pass

    async def evaluate_episode_outlines(
        self,
        story_info,
        role_info,
        world_building,
        season_proposal,
        episode_outlines,
        generate_input=None,
    ):
        
        episode_outlines_text = '\n\n'.join([outline.content for outline in episode_outlines])
        variables = {
            "EpisodeOutlines": episode_outlines_text,
            "RoleInfo": role_info,
            "WorldBuilding": world_building,
            "SeasonProposal": season_proposal.to_proposal_text()
        }

        task = Task(
            stage="evaluation",
            prompt_name="evaluate_episode_outlines",
            domain=story_info.plot_type,
            post_process_func=post_extract_json,
            variables=variables,
            debug_info="evaluate episode outlines",
            extra={"story_info": story_info.__dict__, "generate_input": generate_input.__dict__ if generate_input is not None else {}},
        )

        result = query_llm(
            ctx=self.ctx,
            task=task,
            model_name=self.cfg.model_name,
            max_retries=self.cfg.retry_cnt,
        )

        eval_result = EvalReport.from_dict(result)

        return eval_result

    async def evaluate_act_outline(
            self,
            story_info,
            role_info,
            world_building,
            season_proposal,
            act_outline,
            generate_input=None,
        ):

        variables = {
            "ActOutline": act_outline,
            "RoleInfo": role_info,
            "WorldBuilding": world_building,
            "SeasonProposal": season_proposal.to_proposal_text()
        }

        task = Task(
            stage="evaluation",
            prompt_name="evaluate_act_outline",
            domain=story_info.plot_type,
            post_process_func=post_extract_json,
            variables=variables,
            debug_info="evaluate act outline",
            extra={"story_info": story_info.__dict__, "generate_input": generate_input.__dict__ if generate_input is not None else {}},
        )

        result = query_llm(
            ctx=self.ctx,
            task=task,
            model_name=self.cfg.model_name,
            max_retries=self.cfg.retry_cnt,
        )

        eval_result = EvalReport.from_dict(result)

        return eval_result
