
from trpc.log import logger
import traceback

import asyncio
import copy
import json_repair
from service.shortanime_by_fiction.data_manager import DataManager
from service.shortanime_by_fiction.core.proposal.divider import AnimeScriptDivider
from service.shortanime_by_fiction.data_models.proposal import (
    FictionConfig,
    WorldBuilding,
    Adaptation,
    RoleInfo,
    SeasonProposal,
    PointPlanning
)
from service.shortanime_by_fiction.llms import (
    query_llm,
    Task,
)
from service.shortanime_by_fiction.configs import model_cfg

class ProposalGenerator:
    def __init__(self, ctx):
        self.ctx = ctx
        self.data_loader = DataManager(ctx)
        self.cfg = FictionConfig()
        self.divider = AnimeScriptDivider(ctx)

    def _adaptation_json2text(self, response):
        json_data = json_repair.loads(response)
        text =  "### 整体改编重点" + "\n".join("- " + x for x in json_data['overall_focus'])
        text +=  "\n\n### 开场规则" + "\n".join("- " + x for x in json_data['opening_rules'])
        text +=  "\n\n### 机制呈现规则" + "\n".join("- " + x for x in json_data['mechanism_rules'])
        text +=  "\n\n### 角色与情绪规则" + "\n".join("- " + x for x in json_data['character_emotion_rules'])
        text +=  "\n\n### 禁忌" + "\n".join("- " + x for x in json_data['taboos'])

        adaptation = Adaptation(text)
        return adaptation

    def get_chapter_range(self, proposal, point_id):
        try:
            point = proposal.point_planning.points[point_id]
            start_plot_id, end_plot_id = point.start_plot_id, point.end_plot_id

            chapter_from = proposal.plot_planning[start_plot_id].chapter_from
            chapter_to = proposal.plot_planning[end_plot_id].chapter_to
        except Exception as e:
            logger.error_context(self.ctx, f"Error in get_chapter_range: {e}")
            raise

        return chapter_from, chapter_to

    async def generate_adaptation(
        self,
        story_info,
        generate_input,
        plot_list,
        chapter_content,
        suggestion='暂无'
    ):
        variables = {
            "ChapterContent": chapter_content,
            "Instruction": generate_input.instruction,
            "SeasonNums": generate_input.season_nums,
            "EpisodeNums": generate_input.episode_nums,
            "WordCount": f"{generate_input.min_word_count_per_episode}-{generate_input.max_word_count_per_episode}字",
            "Suggestion": suggestion
        }

        adaptation = query_llm(
            ctx=self.ctx,
            task=Task(
                stage="adaptation",
                prompt_name="adaptation",
                variables=variables,
                domain=story_info.plot_type,
                post_process_func=self._adaptation_json2text,
                debug_info=f"generate adaptation proposal",
                extra={"story_info": story_info.__dict__, "generate_input": generate_input.__dict__},
            ),
            model_name=self.cfg.model_name,
            max_retries=self.cfg.retry_cnt,
            max_new_tokens=20480,
        )
        logger.debug_context(self.ctx, f"adaptation: {adaptation}")

        adaptation_url = await self.data_loader.upload(
            adaptation,
            story_info=story_info
        )
        logger.info_context(self.ctx, f"Adaptation uploaded to COS: {adaptation_url}")
        return adaptation

    def _process_worldbuilding_response(self, response):
        json_data = json_repair.loads(response)
        world_building = WorldBuilding(content=json_data['world_building'])
        return world_building

    async def gen_adapted_worldbuilding(
        self,
        story_info,
        generate_input,
        adaptation,
        plot_list,
        chapter_content,
        suggestion='暂无'
    ):

        variables = {
            "ChapterAbstract": chapter_content,
            "Adaptation": adaptation,
            "Suggestion": suggestion,
            "Instruction": generate_input.instruction
        }

        world_building = query_llm(
            ctx=self.ctx,
            task=Task(
                stage="adaptation",
                prompt_name="world_building",
                variables=variables,
                domain=story_info.plot_type,
                post_process_func=self._process_worldbuilding_response,
                debug_info=f"generate world building",
                extra={"story_info": story_info.__dict__, "generate_input": generate_input.__dict__},
            ),
            model_name=self.cfg.model_name,
            max_retries=self.cfg.retry_cnt,
        )
        world_building_url = await self.data_loader.upload(
            world_building,
            story_info=story_info
        )
        logger.info_context(self.ctx, f"World building uploaded to COS: {world_building_url}")

        return world_building

    def _process_role_info_response(self, response):
        json_data = json_repair.loads(response)
        text = "### 角色信息"
        for role_info in json_data['role_infos']:
            text += f"\n\n#### {role_info['name']}"
            text += f"\n\n#### 改编思路\n{role_info['adaptation']}"
            text += f"\n\n#### 角色设定\n{role_info['information']}"
        text += "\n\n### 其它角色改编思路"
        text += f"\n\n{json_data['other_role_adaptation']}"
        return text

    async def gen_adapted_role_infos(
        self,
        story_info,
        generate_input,
        adaptation,
        plot_list,
        chapter_content,
        suggestion='暂无'
    ):

        variables = {
            "ChapterAbstract": chapter_content,
            "Adaptation": adaptation,
            "Suggestion": suggestion,
            "Instruction": generate_input.instruction
        }

        result = query_llm(
            ctx=self.ctx,
            task=Task(
                stage="adaptation",
                prompt_name="role_info",
                variables=variables,
                domain=story_info.plot_type,
                post_process_func=self._process_role_info_response,
                debug_info=f"generate role infos",
                extra={"story_info": story_info.__dict__, "generate_input": generate_input.__dict__},
            ),
            model_name=self.cfg.model_name,
            max_retries=self.cfg.retry_cnt,
        )
        role_infos = RoleInfo(content=result)
        role_infos_url = await self.data_loader.upload(
            role_infos,
            story_info=story_info
        )
        logger.info_context(self.ctx, f"Role infos uploaded to COS: {role_infos_url}")

        return role_infos

    async def generate_proposal(
        self,
        story_info,
        generate_input,
        suggestion='暂无'
    ):
        try:
            plot_list, chapter_content = await self.data_loader.load_story_arcs_text(
                generate_input.novel_id
            )
            chapter_content = chapter_content[:model_cfg.content_max_word_cnt]
        except Exception as e:
            logger.error_context(self.ctx, f"Failed to load story arcs: {traceback.format_exc()}")
            raise ValueError(f"获取事件列表失败: {repr(e)}")

        # 验证情节数量 > season_num
        if len(plot_list) < generate_input.season_nums:
            logger.error_context(self.ctx, f"情节量不足, 情节量: {len(plot_list)}, 规划季数: {generate_input.season_nums}")
            raise ValueError(f"情节量不足, 情节量: {len(plot_list)}, 规划季数: {generate_input.season_nums}。请检查输入。")

        try:
            adaptation = await self.generate_adaptation(
                story_info,
                generate_input,
                plot_list,
                chapter_content,
                suggestion=suggestion
            )
        except Exception as e:
            logger.error_context(self.ctx, f"Failed to generate adaptation: {e}")
            raise

        try:
            world_building_task = self.gen_adapted_worldbuilding(
                story_info,
                generate_input,
                adaptation,
                plot_list,
                chapter_content,
                suggestion=suggestion
            )

            role_infos_task = self.gen_adapted_role_infos(
                story_info,
                generate_input,
                adaptation,
                plot_list,
                chapter_content,
                suggestion=suggestion
            )

            season_division_task = self.divider.generate_season_division(
                story_info,
                generate_input,
                adaptation,
                plot_list,
                chapter_content,
                suggestion=suggestion
            )

            world_building, role_infos, season_division = await asyncio.gather(
                world_building_task,
                role_infos_task,
                season_division_task
            )

        except Exception as e:
            logger.error_context(self.ctx, f"Failed to generate world building, role infos or season division: {e}")
            raise

        try:
            season_proposals = await self.generate_season_proposals(
                story_info,
                generate_input,
                adaptation,
                season_division,
                suggestion=suggestion
            )
            proposals = [season_division] + season_proposals
        except Exception as e:
            logger.error_context(self.ctx, f"Failed to generate season proposals: {e}")
            raise

        return role_infos, world_building, proposals

    async def generate_season_proposals(
        self,
        story_info,
        generate_input,
        adaptation,
        season_division,
        suggestion='暂无'
    ):
        tasks = []
        for season_id, point in enumerate(season_division.point_planning.points):
            start_plot_id, end_plot_id = point.start_plot_id, point.end_plot_id

            plot_list, chapter_content = SeasonProposal.get_plots_and_content(season_division, start_plot_id, end_plot_id)
            chapter_content = chapter_content[:model_cfg.content_max_word_cnt]
            chapter_from = season_division.plot_planning[start_plot_id].chapter_from
            chapter_to = season_division.plot_planning[end_plot_id].chapter_to
            chapter_range = f"§{chapter_from} - §{chapter_to}"

            task = self.generate_season_proposal(
                story_info,
                generate_input,
                adaptation,
                season_id,
                plot_list,
                chapter_content,
                chapter_range,
                division_description=season_division.point_planning.description,
                suggestion=suggestion
            )
            tasks.append(task)

        proposals = await asyncio.gather(*tasks)
        return proposals

    def _process_plot_planning(self, plot_planning, plot_list):
        """
            将status填入plot_list
        """
        result = {}
        for plot in plot_list:
            result[plot['plot_id']] = {
                "plot_id": plot['plot_id'],
                "title": plot['title'],
                "description": plot['description'],
                "status": "删除",
                "chapter_from": plot['chapter_from'],
                "chapter_to": plot['chapter_to'],
                "chapters": []
            }
        for plan in plot_planning:
            result[plan['plot_id']]['status'] = plan['status']
        return list(result.values())

    def _process_point_planning(self, point_planning, max_plot_id):
        description = "### 首集开篇:\n" + point_planning['opening']['description']
        description += "\n\n### 前三集卡点说明:\n" + point_planning['first_three_eps']['description']
        description += "\n\n### 一卡卡点说明:\n" + point_planning['first_paywall']['description']
        description += "\n\n### 结局说明:\n" + point_planning['season_finale']['description']
        points = []
        for point_name, point in zip(['首集', '前三集', '一卡', '结局'], \
                                     [point_planning['opening'], point_planning['first_three_eps'], point_planning['first_paywall'], point_planning['season_finale']]):
            point_id = len(points)
            points.append({
                "point_id": point_id,
                "point_name": point_name,
                "start_plot_id": min(point['start_plot_id'], max_plot_id),
                "end_plot_id": min(point['end_plot_id'], max_plot_id)
            })
        point_planning = {
            "points": points,
            "description": description
        }
        return point_planning

    def _process_season_proposal(self, response, plot_list):
        json_data = json_repair.loads(response)

        plot_planning = self._process_plot_planning(json_data['plot_planning'], plot_list)
        max_plot_id = len(plot_list)
        point_planning = self._process_point_planning(json_data['point_planning'], max_plot_id)

        storylines = json_data['story_lines']
        other_adaptation_proposal = json_data['other_adaptation_proposal']
        season_adaptation = {
            "plot_planning": plot_planning,
            "point_planning": point_planning,
            "storylines": storylines,
            "other_adaptation_proposal": other_adaptation_proposal
        }

        season_adaptation = SeasonProposal.from_dict(season_adaptation)
        return season_adaptation

    async def generate_season_proposal(
        self,
        story_info,
        generate_input,
        adaptation,
        season_id,
        plot_list,
        chapter_content,
        chapter_range,
        division_description='暂无',
        suggestion='暂无'
    ):

        variables = {
            "SeasonCnt": generate_input.season_nums,
            "SeasonIndex": season_id + 1,
            "EpisodesCnt": generate_input.episode_nums,
            "WordCount": f"{generate_input.min_word_count_per_episode}-{generate_input.max_word_count_per_episode}字",
            "ChapterRange": chapter_range,
            "ChapterContent": chapter_content,
            "Adaptation": adaptation,
            "DivisionDescription": division_description,
            "Suggestion": suggestion,
            "Instruction": generate_input.instruction
        }

        season_proposal = query_llm(
            ctx=self.ctx,
            task=Task(
                stage="adaptation",
                prompt_name="season_proposal",
                variables=variables,
                domain=story_info.plot_type,
                post_process_func=self._process_season_proposal,
                post_process_kwargs={"plot_list": plot_list},
                debug_info=f"generate plot point planning",
                extra={"story_info": story_info.__dict__, "generate_input": generate_input.__dict__},
            ),
            model_name=self.cfg.model_name,
            max_retries=self.cfg.retry_cnt,
        )
        season_proposal.title = f"第{season_id + 1}季"
        season_proposal.season_id = season_id

        logger.debug_context(self.ctx, f"Season proposal: {season_proposal}")
        season_proposal_url = await self.data_loader.upload(
            season_proposal,
            story_info=story_info,
            season_id=season_id
        )
        logger.info_context(self.ctx, f"Season proposal uploaded to COS: {season_proposal_url}")

        return season_proposal

    async def regenerate_season_division(
        self,
        story_info,
        generate_input,
        season_division,
        suggestion='暂无'
    ):
        adaptation = await self.data_loader.load_adaptation(story_info)

        try:
            season_division = await self.divider.regenerate_season_division(
                story_info,
                generate_input,
                adaptation,
                season_division
            )
        except Exception as e:
            logger.error_context(self.ctx, f"Failed to regenerate season division: {e}")
            raise

        try:
            season_proposals = await self.generate_season_proposals(
                story_info,
                generate_input,
                adaptation,
                season_division,
                suggestion=suggestion
            )
            proposals = [season_division] + season_proposals
        except Exception as e:
            logger.error_context(self.ctx, f"Failed to generate season proposals: {e}")
            raise

        return proposals

    def _process_season_points(self, response, max_plot_id):
        json_data = json_repair.loads(response)
        point_planning = self._process_point_planning(json_data['point_planning'], max_plot_id)
        return point_planning

    async def regenerate_season_points(
        self,
        story_info,
        generate_input,
        season_id,
        season_proposal,
    ):
        chapter_content = season_proposal.get_plot_text()
        variables = {
            "ChapterContent": chapter_content,
            "Points": season_proposal.point_planning.points
        }

        try:
            max_plot_id = len(season_proposal.plot_planning)
            point_planning = query_llm(
                ctx=self.ctx,
                task=Task(
                    stage="adaptation",
                    prompt_name="regenerate_point_description",
                    variables=variables,
                    domain=story_info.plot_type,
                    post_process_func=self._process_season_points,
                    post_process_kwargs={"max_plot_id": max_plot_id},
                    debug_info=f"regenerate points",
                    extra={"story_info": story_info.__dict__, "generate_input": generate_input.__dict__},
                ),
                model_name=self.cfg.model_name,
                max_retries=self.cfg.retry_cnt,
            )
            season_proposal.point_planning = PointPlanning.from_dict(point_planning)

            season_proposal = SeasonProposal.from_dict(season_proposal)

            season_proposal_url = await self.data_loader.upload(
                season_proposal,
                story_info=story_info,
                season_id=season_id
            )
            logger.info_context(self.ctx, f"Season proposal uploaded to COS: {season_proposal_url}")

        except Exception as e:
            logger.error_context(self.ctx, f"Regenerate points failed: {e}")

        return season_proposal
