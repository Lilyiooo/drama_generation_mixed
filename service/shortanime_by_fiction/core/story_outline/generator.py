from concurrent.futures import ThreadPoolExecutor

import re
import json_repair
from trpc.log import logger

from service.shortanime_by_fiction.data_models.drama import (
    ActOutline, StoryOutline,
    EpisodeOutline,
)
from service.shortanime_by_fiction.data_models.fiction import StoryInfo
from service.shortanime_by_fiction.configs import FictionConfig, model_cfg
from service.shortanime_by_fiction.data_manager import DataManager
from service.shortanime_by_fiction.llms import (
    query_llm,
    Task
)


class StoryOutlineGenerator:
    def __init__(self, ctx):
        self.ctx = ctx
        self.data_manager = DataManager(ctx)
        self.cfg = FictionConfig()

    def _process_story_outline(self, response):
        json_data = json_repair.loads(response)
        story_outline = StoryOutline(content=json_data['outline'])
        return story_outline

    async def generate_story_outline(
        self,
        story_info,
        generate_input,
        season_id,
        world_building,
        role_info,
        chapter_from,
        chapter_to,
        season_proposal,
        demo_drama,
        suggestion='暂无',
        ctype="summary",
    ):
        try:
            chapter_content = await self.data_manager.load_contents_adaptive(
                generate_input.novel_id, chapter_from, chapter_to, max_length=model_cfg.content_max_word_cnt
            )
            completed_script_content = "\n\n".join([f"第{script['episode_id']+1}集\n" + script['script_content'] for script in demo_drama])
        except Exception as e:
            logger.error_context(self.ctx, f"Error in loading chapter content: {e}")
            raise

        variables = {
            "WorldBuilding": world_building,
            "RoleInfos": role_info,
            "SeasonIndex": season_id + 1,
            "SeasonCnt": generate_input.season_nums,
            "EpisodesCnt": generate_input.episode_nums,
            "ChapterRange": f"{chapter_from}~{chapter_to}章",
            "ChapterAbstract": chapter_content,
            "SeasonProposal": season_proposal.to_proposal_text(),
            "CompletedScript": completed_script_content,
            "Suggestion": suggestion,
            "Instruction": generate_input.instruction
        }

        try:

            story_outline = query_llm(
                ctx=self.ctx,
                task=Task(
                    stage="plan",
                    prompt_name="story_outline",
                    variables=variables,
                    domain=story_info.plot_type,
                    post_process_func=self._process_story_outline,
                    debug_info="generate story outline",
                    extra={"story_info": story_info.__dict__, "generate_input": generate_input.__dict__},
                ),
                model_name=self.cfg.model_name,
                max_retries=self.cfg.retry_cnt,
            )
            story_outline_url = await self.data_manager.upload(
                story_outline, story_info=story_info, season_id=season_id
            )
            logger.info_context(self.ctx, f"Story outline uploaded to COS: {story_outline_url}")

        except Exception as e:
            story_outline = StoryOutline(content="")
            logger.error_context(self.ctx, f"Error: {e}")

        return story_outline