import json_repair
from trpc.log import logger

from service.drama_by_fiction.data_model import FictionConfig, EpisodeOutline
from service.drama_by_fiction.core.data_manager import DataManager
from service.drama_by_fiction.utils import (
    _query_llm_with_postprocess,
    Task,
)


class EpisodeOutlineGenerator:
    """专注于「分集大纲」的生成，职责单一。"""

    def __init__(self, ctx):
        self.ctx = ctx
        self.data_manager = DataManager(ctx)
        self.cfg = FictionConfig()

    async def upload_episode_outlines(self, story_info, episode_outlines):
        for epi_outline in episode_outlines:
            await self.data_manager.upload(
                epi_outline,
                story_id=story_info.story_id,
                season_id=epi_outline.season_id,
                episode_id=epi_outline.episode_id,
            )

    def _get_ch_range(self, chapter_range):
        ch_ranges = []
        for cr in chapter_range:
            try:
                cr = str(cr)
                if cr == "0":
                    continue
                if "-" in cr:
                    start, end = cr.split("-")
                    ch_ranges.extend(range(int(start), int(end) + 1))
                else:
                    ch_ranges.append(int(cr))
            except Exception:  # pylint: disable=broad-except
                continue
        ch_ranges = sorted(ch_ranges)

        if not ch_ranges:
            return 0, 0, []
        start, end = ch_ranges[0], ch_ranges[-1]
        return start, end, ch_ranges

    def _process_episode_plan(self, response, season_id):
        json_data = json_repair.loads(response)

        episode_outlines = []
        completed_episodes = []
        for outline in json_data:
            chapter_range = outline["chapter_range"]
            start, end, ch_ranges = self._get_ch_range(chapter_range)
            episode_id = outline["episode_number"] - 1
            completed_episodes.append(outline["episode_number"])

            content = EpisodeOutline.to_text(
                episode_id,
                chapter_range,
                outline["synopsis"],
                outline["events"],
                outline["ending_hook"],
            )
            episode_outline = {
                "season_id": season_id,
                "episode_id": episode_id,
                "episode_title": outline["episode_title"],
                "ch_ranges": ch_ranges,
                "chapter_from": start,
                "chapter_to": end,
                "chapter_range": chapter_range,
                "content": content,
            }
            episode_outline = EpisodeOutline.from_dict(episode_outline)
            episode_outlines.append(episode_outline)
        return episode_outlines

    def do_generate_episode_plans(
        self,
        generate_input,
        season_id,
        world_building,
        role_info,
        chapter_from,
        chapter_to,
        season_proposal,
        chapter_content,
        story_outline,
        prev_summary="暂无",
        suggestion="暂无",
    ):
        """分集大纲生成核心逻辑（同步），仅包含 LLM 调用。

        chapter_content 由外部预加载，upload 由外部统一执行。
        """
        variables = {
            "NovelInfo": "暂无",
            "WorldBuilding": world_building,
            "RoleInfos": role_info,
            "SeasonIndex": season_id + 1,
            "SeasonCnt": generate_input.season_nums,
            "EpisodesCnt": generate_input.episode_nums,
            "ChapterRange": f"{chapter_from}~{chapter_to}章",
            "StoryOutline": story_outline,
            "ChapterAbstrAct": chapter_content,
            "Proposal": season_proposal.to_proposal_text(),
            "PrevSummary": prev_summary,
            "Suggestion": suggestion,
        }

        task = Task(
            stage="plan",
            prompt_name="episode_plan",
            variables=variables,
            debug_info="generate episode outline",
            post_process_func=self._process_episode_plan,
            post_process_kwargs={"season_id": season_id},
        )

        episode_outlines = _query_llm_with_postprocess(
            ctx=self.ctx,
            task=task,
            model_name=self.cfg.model_name,
            max_retries=self.cfg.retry_cnt,
        )
        return episode_outlines

    async def generate_episode_plans(
        self,
        story_info,
        generate_input,
        season_id,
        world_building,
        role_info,
        chapter_from,
        chapter_to,
        season_proposal,
        story_outline,
        prev_summary="暂无",
        suggestion="暂无",
    ):
        try:
            chapter_content = await self.data_manager.load_contents_adaptive(
                generate_input.novel_id, chapter_from, chapter_to, self.cfg.max_word_cnt
            )
        except Exception as e:
            logger.error_context(
                self.ctx, f"Error loading chapter content for season {season_id}: {e}"
            )
            raise

        try:
            episode_outlines = self.do_generate_episode_plans(
                generate_input=generate_input,
                season_id=season_id,
                world_building=world_building,
                role_info=role_info,
                chapter_from=chapter_from,
                chapter_to=chapter_to,
                season_proposal=season_proposal,
                chapter_content=chapter_content,
                story_outline=story_outline,
                prev_summary=prev_summary,
                suggestion=suggestion,
            )
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Error generating episode outlines with LLM for season {season_id}: {e}",
            )
            raise

        try:
            await self.upload_episode_outlines(story_info, episode_outlines)
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Error uploading episode outlines for season {season_id}: {e}",
            )
            raise

        return episode_outlines
