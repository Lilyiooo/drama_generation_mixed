import cn2an
import json_repair
from trpc.log import logger

from service.shortanime_by_fiction.data_models.proposal import FictionConfig, SeasonProposal, PointPlanning
from service.shortanime_by_fiction.data_manager import DataManager
from service.shortanime_by_fiction.llms import (
    query_llm,
    Task
)

class AnimeScriptDivider:
    def __init__(self, ctx):
        self.ctx = ctx
        self.data_loader = DataManager(ctx)
        self.cfg = FictionConfig()

    def _process_points(self, json_data, season_nums, max_plot_id):
        """
            处理卡点规划
        """
        assert season_nums == len(json_data['season_details']), f"Length of season details: {len(json_data['season_details'])} is not equal to season count: {season_nums}"

        points = []
        description = "### 划分依据\n" + json_data['division_basis']
        prev_end_plot_id = -1

        for season_id, season in enumerate(json_data['season_details']):
            point_name = f"第{cn2an.an2cn(season_id + 1)}季"
            description += f"\n\n### {point_name} [{season['season_title']}]"
            description += f"\n{season['content']}"

            # 每一季的start为前一季结尾+1，首季开始为0
            start_plot_id = 0 if season_id == 0 else prev_end_plot_id + 1
            end_plot_id = season['end_plot_id']

            # 检查是否超过max_plot_id
            if start_plot_id > max_plot_id:
                raise ValueError(f"Season {season_id + 1} start_plot_id ({start_plot_id}) exceeds max_plot_id ({max_plot_id})")
            if end_plot_id > max_plot_id:
                raise ValueError(f"Season {season_id + 1} end_plot_id ({end_plot_id}) exceeds max_plot_id ({max_plot_id})")

            points.append({
                "point_id": season_id,
                "point_name": point_name,
                "start_plot_id": start_plot_id,
                "end_plot_id": end_plot_id
            })

            prev_end_plot_id = end_plot_id

        point_planning = {
            "points": points,
            "description": description
        }
        return point_planning

    def _process_season_division(self, response, plot_list, season_nums):
        json_data = json_repair.loads(response)
        plot_planning = {}
        for plot in plot_list:
            plot_planning[plot['plot_id']] = {
                "plot_id": plot['plot_id'],
                "title": plot['title'],
                "description": plot['description'],
                "status": "保留",
                "chapter_from": plot['chapter_from'],
                "chapter_to": plot['chapter_to'],
                "chapters": []
            }
        plot_planning = list(plot_planning.values())
        max_plot_id = plot_planning[-1]['plot_id'] if plot_planning else 0
        point_planning = self._process_points(json_data, season_nums, max_plot_id)

        result = {
            "title": "整体卡点规划",
            "season_id": -1,
            "plot_planning": plot_planning,
            "point_planning": point_planning
        }
        return SeasonProposal.from_dict(result)

    async def generate_season_division(
        self,
        story_info,
        generate_input,
        adaptation,
        plot_list,
        chapter_content,
        suggestion='暂无'
    ):

        chapter_cnt = await self.data_loader.get_max_chapter_idx(generate_input.novel_id)
        max_plot_id = len(plot_list) - 1

        variables = {
            "ChapterContent": chapter_content,
            "ChapterCnt": chapter_cnt,
            "SeasonCnt": generate_input.season_nums,
            "EpisodeCnt": generate_input.episode_nums,
            "Adapatation": adaptation,
            "Suggestion": suggestion,
            "MAX_PLOT_ID": max_plot_id,
            "Instruction": generate_input.instruction
        }

        season_division = query_llm(
            ctx=self.ctx,
            task=Task(
                stage="divide",
                prompt_name="season_division",
                domain=story_info.plot_type,
                post_process_func=self._process_season_division,
                post_process_kwargs={"plot_list": plot_list, "season_nums": generate_input.season_nums},
                variables=variables,
                debug_info=f"generate season division",
                extra={"story_info": story_info.__dict__, "generate_input": generate_input.__dict__},
            ),
            model_name=self.cfg.model_name,
            max_retries=self.cfg.retry_cnt
        )

        season_division_url = await self.data_loader.upload(season_division, story_info=story_info)

        logger.info_context(
            self.ctx, f"Generated season division uploaded to COS: {season_division_url}"
        )

        return season_division

    def _process_re_season_division(self, response, season_nums, max_plot_id):
        json_data = json_repair.loads(response)
        point_planning = self._process_points(json_data, season_nums, max_plot_id)
        return point_planning

    async def regenerate_season_division(
        self,
        story_info,
        generate_input,
        adaptation,
        season_division,
    ):
        chapter_content = season_division.get_plot_text()
        chapter_cnt = await self.data_loader.get_max_chapter_idx(generate_input.novel_id)

        variables = {
            "ChapterContent": chapter_content,
            "ChapterCnt": chapter_cnt,
            "SeasonCnt": generate_input.season_nums,
            "Points": season_division.point_planning.points
        }

        try:
            max_plot_id = season_division.plot_planning[-1].plot_id if season_division.plot_planning else 0
            point_planning = query_llm(
                ctx=self.ctx,
                task=Task(
                    stage="divide",
                    prompt_name="regenerate_season_division",
                    domain=story_info.plot_type,
                    post_process_func=self._process_re_season_division,
                    post_process_kwargs={"season_nums": generate_input.season_nums, "max_plot_id": max_plot_id},
                    variables=variables,
                    debug_info=f"regenerate season division",
                    extra={"story_info": story_info.__dict__, "generate_input": generate_input.__dict__},
                ),
                model_name=self.cfg.model_name,
                max_retries=self.cfg.retry_cnt
            )
            season_division.point_planning = PointPlanning.from_dict(point_planning)

            logger.info_context(self.ctx, f"season division: {season_division}")
            season_division_url = await self.data_loader.upload(season_division, story_info=story_info)
            logger.info_context(self.ctx, f"Generated season division uploaded to COS: {season_division_url}")
        except Exception as e:
            logger.error_context(self.ctx, f"Error in regenerate_season_division: {e}")
            raise

        return season_division
