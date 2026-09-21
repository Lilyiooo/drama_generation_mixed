from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import json_repair
from trpc.log import logger
from trpc.exceptions import NewBusinessError

from service.drama_by_fiction.error_codes import BizCode, BIZ_MSG
from service.drama_by_fiction.core.data_manager import DataManager
from service.drama_by_fiction.core.script_divide import AnimeScriptDivider
from service.drama_by_fiction.core.episode_outlines import EpisodeOutlineGenerator
from service.drama_by_fiction.data_model import (
    FictionConfig,
    StoryInfo,
    GenerateInput,
    WorldBuilding,
    RoleInfo,
    EventProposal,
    PlotProposal,
    EpisodeProposal,
    SeasonProposal,
)
from service.drama_by_fiction.utils import (
    _query_llm_with_postprocess,
    Task,
)


class PointPlanner:
    def __init__(self, ctx):
        self.ctx = ctx
        self.data_loader = DataManager(ctx)
        self.cfg = FictionConfig()
        self.divider = AnimeScriptDivider(ctx)
        self.outline_generator = EpisodeOutlineGenerator(ctx)

    def get_chapter_range(self, proposal: SeasonProposal, point_id):
        try:
            point = proposal.point_planning.points[point_id]
            start_plot_id, end_plot_id = point.start_plot_id, point.end_plot_id

            chapter_from = proposal.plot_planning[start_plot_id].chapter_from
            chapter_to = proposal.plot_planning[end_plot_id].chapter_to
        except Exception as e:
            logger.error_context(self.ctx, f"Error in get_chapter_range: {e}")
            raise

        return chapter_from, chapter_to

    async def generate_point_plan(
        self,
        story_info: StoryInfo,
        generate_input: GenerateInput,
        world_building: WorldBuilding,
        role_info: RoleInfo,
        suggestion="暂无",
    ):
        try:
            plot_list, story_arc_text = await self.data_loader.load_format_story_arcs(
                generate_input.novel_id
            )
            story_arc_text = story_arc_text[: self.cfg.max_word_cnt]
        except Exception as e:
            logger.error_context(self.ctx, f"Failed to load story arcs: {e}")
            raise RuntimeError(f"获取事件列表失败: {e}")

        # 验证情节数量 > season_num
        if len(plot_list) < generate_input.season_nums:
            err_msg = f"情节量不足，每季至少包含一个情节。情节量: {len(plot_list)}, 规划季数: {generate_input.season_nums}"
            logger.error_context(self.ctx, err_msg)
            raise NewBusinessError(
                BizCode.PARAM_INVALID, f"{BIZ_MSG[BizCode.PARAM_INVALID]}: {err_msg}"
            )

        adaptation = await self.data_loader.load_adaptation(generate_input.novel_id)
        try:
            season_division = await self.divider.generate_season_division(
                generate_input=generate_input,
                adaptation=adaptation,
                plot_list=plot_list,
                chapter_content=story_arc_text,
                suggestion=suggestion,
            )
            await self.data_loader.upload(season_division, story_id=story_info.story_id)

        except Exception as e:
            logger.error_context(self.ctx, f"Failed to generate season division: {e}")
            raise

        # 卡点规划
        try:
            season_proposals = await self.generate_season_proposals(
                story_info,
                generate_input,
                adaptation,
                world_building,
                role_info,
                season_division,
                suggestion=suggestion,
            )
            proposals = [season_division] + season_proposals
        except Exception as e:
            logger.error_context(self.ctx, f"Failed to generate season proposals: {e}")
            raise

        return proposals

    def _generate_season_proposal_sync(
        self,
        generate_input,
        adaptation,
        world_building,
        role_info,
        season_id,
        plot_list,
        chapter_content,
        chapter_from,
        chapter_to,
        outline_chapter_content,
        season_proposal_description="暂无",
        suggestion="暂无",
    ):
        """单个季方案生成的同步核心逻辑（供线程池调用）。

        包含两次 LLM 调用：季方案 + 分集大纲。
        所有异步数据由外部预加载，upload 由外部统一执行。
        """
        chapter_range = f"§{chapter_from} - §{chapter_to}"

        variables = {
            "SeasonCnt": generate_input.season_nums,
            "SeasonIndex": season_id + 1,
            "EpisodesCnt": generate_input.episode_nums,
            "ChapterRange": chapter_range,
            "ChapterContent": chapter_content,
            "Adaptation": adaptation,
            "WorldBuilding": world_building,
            "RoleInfos": role_info,
            "DivisionDescription": season_proposal_description,
            "Suggestion": suggestion,
        }

        season_proposal: SeasonProposal = _query_llm_with_postprocess(
            ctx=self.ctx,
            task=Task(
                stage="adaptation",
                prompt_name="season_proposal",
                variables=variables,
                post_process_func=self._process_season_proposal,
                post_process_kwargs={"plot_list": plot_list},
                debug_info="generate plot point planning",
            ),
            model_name=self.cfg.model_name,
            max_retries=self.cfg.retry_cnt,
        )
        season_proposal.title = f"第{season_id + 1}季"
        season_proposal.season_id = season_id

        # 生成分集大纲（同步 LLM 调用）
        brief_outlines = self.outline_generator.do_generate_episode_plans(
            generate_input=generate_input,
            season_id=season_id,
            world_building=world_building,
            role_info=role_info,
            chapter_from=chapter_from,
            chapter_to=chapter_to,
            season_proposal=season_proposal,
            chapter_content=outline_chapter_content,
            story_outline="暂无",
            suggestion=suggestion,
        )
        season_proposal.episode_planning = self._process_episode_planning(
            brief_outlines, plot_list
        )

        logger.info_context(self.ctx, f"Season proposal: {season_proposal}")
        return season_proposal, brief_outlines

    async def generate_season_proposals(
        self,
        story_info,
        generate_input,
        adaptation,
        world_building,
        role_info,
        season_division: SeasonProposal,
        suggestion="暂无",
    ):

        #  用线程池并行执行各季任务
        season_count = len(season_division.point_planning.points)
        with ThreadPoolExecutor(max_workers=season_count) as executor:
            futures = []
            for season_id, point in enumerate(season_division.point_planning.points):
                start_plot_id, end_plot_id = point.start_plot_id, point.end_plot_id
                chapter_from = season_division.plot_planning[start_plot_id].chapter_from
                chapter_to = season_division.plot_planning[end_plot_id].chapter_to
                plot_list, arc_text = await self.data_loader.load_format_story_arcs(
                    generate_input.novel_id, chapter_from, chapter_to
                )
                chapter_content = arc_text[: self.cfg.max_word_cnt]
                outline_chapter_content = await self.data_loader.load_contents_adaptive(
                    generate_input.novel_id,
                    chapter_from,
                    chapter_to,
                    self.cfg.max_word_cnt,
                )

                future = executor.submit(
                    self._generate_season_proposal_sync,
                    generate_input=generate_input,
                    adaptation=adaptation,
                    world_building=world_building,
                    role_info=role_info,
                    season_id=season_id,
                    plot_list=plot_list,
                    chapter_content=chapter_content,
                    chapter_from=chapter_from,
                    chapter_to=chapter_to,
                    outline_chapter_content=outline_chapter_content,
                    season_proposal_description=season_division.point_planning.description,
                    suggestion=suggestion,
                )
                futures.append(future)
            results = [f.result() for f in futures]

        # 4. 统一上传
        proposals = []
        for season_proposal, brief_outlines in results:
            await self.data_loader.upload(
                season_proposal,
                story_id=story_info.story_id,
                season_id=season_proposal.season_id,
            )
            await self.outline_generator.upload_episode_outlines(
                story_info, brief_outlines
            )
            proposals.append(season_proposal)

        return proposals

    def _process_plot_planning(self, plot_planning, plot_list):
        plots = deepcopy(plot_list)
        result = {}
        for plot in plots:
            result[plot["plot_id"]] = plot
        for plan in plot_planning:
            if plan["plot_id"] not in result:
                continue
            result[plan["plot_id"]]["status"] = plan["status"]
        return list(result.values())

    def _process_point_planning(self, point_planning, max_plot_id):
        POINT_NAME_MAP = {
            "opening": "首集开始",
            "first_three_eps": "第三集结尾",
            "mid_season_climax": "季中高潮",
            "season_finale": "结局",
        }
        desc_texts = []
        points = []
        for pp, point_name in POINT_NAME_MAP.items():
            point = point_planning[pp]
            desc_texts.append(f"### {point_name}:\n{point['description']}")
            item = {
                "point_id": len(points),
                "point_name": point_name,
                "start_plot_id": min(point["start_plot_id"], max_plot_id),
                "end_plot_id": min(point["end_plot_id"], max_plot_id),
            }
            points.append(item)
        point_planning = {"points": points, "description": "\n\n".join(desc_texts)}
        return point_planning

    def _process_season_proposal(self, response, plot_list):
        json_data = json_repair.loads(response)

        plot_planning = self._process_plot_planning(
            json_data["plot_planning"], plot_list
        )
        max_plot_id = len(plot_list) - 1
        point_planning = self._process_point_planning(
            json_data["point_planning"], max_plot_id
        )

        storylines = json_data["story_lines"]
        other_adaptation_proposal = json_data["other_adaptation_proposal"]
        season_adaptation = {
            "plot_planning": plot_planning,
            "point_planning": point_planning,
            "storylines": storylines,
            "other_adaptation_proposal": other_adaptation_proposal,
        }

        season_adaptation = SeasonProposal.from_dict(season_adaptation)
        return season_adaptation

    def _process_episode_planning(
        self, brief_outlines, plot_list
    ) -> list[EpisodeProposal]:
        results = []
        for outline in brief_outlines:
            outline_ch_ranges = set(outline.ch_ranges)
            selected_plots = []
            for plot in plot_list:
                selected_events = []
                for event in plot["events"]:
                    chapters = []
                    for ch in range(event["chapter_from"], event["chapter_to"] + 1):
                        if ch in outline_ch_ranges:
                            chapters.append(ch)
                    if chapters:
                        selected_events.append(
                            EventProposal(event_id=event["event_id"], chapters=chapters)
                        )
                if selected_events:
                    selected_plots.append(
                        PlotProposal(plot_id=plot["plot_id"], events=selected_events)
                    )
            episode = EpisodeProposal(
                episode_id=outline.episode_id,
                episode_title=outline.episode_title,
                plots=selected_plots,
            )
            results.append(episode)
        return results

    async def generate_season_proposal(
        self,
        story_info,
        generate_input,
        adaptation,
        world_building,
        role_infos,
        season_id,
        chapter_from,
        chapter_to,
        division_description="暂无",
        suggestion="暂无",
    ):
        """生成单个季方案（异步方法），内部加载数据并上传结果。"""
        plot_list, arc_text = await self.data_loader.load_format_story_arcs(
            generate_input.novel_id, chapter_from, chapter_to
        )
        chapter_content = arc_text[: self.cfg.max_word_cnt]
        outline_chapter_content = await self.data_loader.load_contents_adaptive(
            generate_input.novel_id, chapter_from, chapter_to, self.cfg.max_word_cnt
        )

        season_proposal, brief_outlines = self._generate_season_proposal_sync(
            generate_input=generate_input,
            adaptation=adaptation,
            world_building=world_building,
            role_infos=role_infos,
            season_id=season_id,
            plot_list=plot_list,
            chapter_content=chapter_content,
            chapter_from=chapter_from,
            chapter_to=chapter_to,
            outline_chapter_content=outline_chapter_content,
            season_proposal_description=division_description,
            suggestion=suggestion,
        )

        season_proposal_url = await self.data_loader.upload(
            season_proposal, story_id=story_info.story_id, season_id=season_id
        )
        logger.info_context(
            self.ctx, f"Season proposal uploaded to COS: {season_proposal_url}"
        )
        await self.outline_generator.upload_episode_outlines(story_info, brief_outlines)
        return season_proposal

    def _process_season_points(self, response, max_plot_id):
        json_data = json_repair.loads(response)
        point_planning = self._process_point_planning(
            json_data["point_planning"], max_plot_id
        )
        return point_planning
