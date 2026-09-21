import json_repair
from trpc.log import logger

from service.drama_by_fiction.data_model import (
    FictionConfig,
    StoryInfo,
    GenerateInput,
    SeasonProposal,
    StoryOutline,
)
from service.drama_by_fiction.core.data_manager import DataManager
from service.drama_by_fiction.utils import (
    _query_llm_with_postprocess,
    Task,
)


class OutlineGenerator:
    def __init__(self, ctx):
        self.ctx = ctx
        self.data_manager = DataManager(ctx)
        self.cfg = FictionConfig()

    def _process_story_outline(self, response):
        json_data = json_repair.loads(response)
        story_outline = StoryOutline(content=json_data["outline"])
        return story_outline

    async def generate_story_outline(
        self,
        story_info: StoryInfo,
        generate_input: GenerateInput,
        season_id,
        world_building,
        role_info,
        chapter_from,
        chapter_to,
        season_proposal: SeasonProposal,
        suggestion="暂无",
    ):
        try:
            chapter_content = await self.data_manager.load_contents_adaptive(
                generate_input.novel_id, chapter_from, chapter_to, self.cfg.max_word_cnt
            )
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
            "Suggestion": suggestion,
            "Instruction": generate_input.instruction,
        }

        try:

            story_outline = _query_llm_with_postprocess(
                ctx=self.ctx,
                task=Task(
                    stage="plan",
                    prompt_name="story_outline",
                    variables=variables,
                    post_process_func=self._process_story_outline,
                    debug_info="generate story outline",
                ),
                model_name=self.cfg.model_name,
                max_retries=self.cfg.retry_cnt,
            )
            story_outline_url = await self.data_manager.upload(
                story_outline, story_id=story_info.story_id, season_id=season_id
            )
            logger.info_context(
                self.ctx, f"Story outline uploaded to COS: {story_outline_url}"
            )
        except Exception as e:
            # 生成失败返回空大纲
            story_outline = StoryOutline(content="")
            logger.error_context(self.ctx, f"Error: {e}")

        return story_outline

