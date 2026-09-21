from concurrent.futures import ThreadPoolExecutor

import json_repair
from trpc.log import logger

from service.drama_by_fiction.core.data_manager import DataManager
from service.drama_by_fiction.core.script_divide import AnimeScriptDivider
from service.drama_by_fiction.core.episode_outlines import EpisodeOutlineGenerator
from service.drama_by_fiction.data_model import (
    FictionConfig,
    Adaptation,
    WorldBuilding,
    RoleInfo,
    SeasonProposal,
)
from service.drama_by_fiction.utils import (
    _query_llm_with_postprocess,
    Task,
)


class ProposalGenerator:
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

    def _adaptation_json2text(self, response):
        json_data = json_repair.loads(response)
        DIM_DESC = {
            "overall_focus": "整体改编重点",
            "opening_rules": "开场规则",
            "narrative_rhythm_rules": "叙事节奏规则",
            "mechanism_rules": "机制呈现规则",
            "character_emotion_rules": "角色与情绪规则",
            "story_structure_rules": "故事结构规则",
            "taboos": "禁忌",
        }
        text = ""
        for dim, desc in DIM_DESC.items():
            text += f"\n\n### {desc}\n" + "\n".join("- " + x for x in json_data[dim])
        adaptation = Adaptation(text.strip())
        return adaptation

    async def generate_adaptation(
        self, story_info, generate_input, story_arc_text, suggestion="暂无"
    ):

        variables = {
            "StoryArcs": story_arc_text,
            "SeasonNums": generate_input.season_nums,
            "EpisodeNums": generate_input.episode_nums,
            "Instruction": generate_input.instruction,
            "Suggestion": suggestion,
        }

        adaptation = _query_llm_with_postprocess(
            ctx=self.ctx,
            task=Task(
                stage="adaptation",
                prompt_name="adaptation",
                variables=variables,
                post_process_func=self._adaptation_json2text,
                debug_info=f"generate adaptation proposal",
            ),
            model_name=self.cfg.model_name,
            max_retries=self.cfg.retry_cnt,
        )
        logger.info_context(self.ctx, f"adaptation: {adaptation}")

        adaptation_url = await self.data_loader.upload(
            adaptation, story_id=story_info.story_id
        )
        logger.info_context(self.ctx, f"Adaptation uploaded to COS: {adaptation_url}")
        return adaptation

    def _process_worldbuilding_response(self, response):
        json_data = json_repair.loads(response)
        world_building = WorldBuilding(content=json_data["world_building"])
        return world_building

    def gen_adapted_worldbuilding(
        self,
        adaptation,
        story_arc_text,
        suggestion="暂无",
    ):
        """生成世界观（同步方法，供线程池调用）。"""
        variables = {
            "StoryArcs": story_arc_text,
            "Adaptation": adaptation,
            "Suggestion": suggestion,
        }

        world_building = _query_llm_with_postprocess(
            ctx=self.ctx,
            task=Task(
                stage="adaptation",
                prompt_name="world_building",
                variables=variables,
                post_process_func=self._process_worldbuilding_response,
                debug_info="generate world building",
            ),
            model_name=self.cfg.model_name,
            max_retries=self.cfg.retry_cnt,
        )
        return world_building

    def _process_role_info_response(self, response):
        json_data = json_repair.loads(response)
        text = "### 角色信息"
        for role_info in json_data["role_infos"]:
            text += f"\n\n#### {role_info['name']}"
            text += f"\n\n#### 改编思路\n{role_info['adaptation']}"
            text += f"\n\n#### 角色设定\n{role_info['information']}"
        text += "\n\n### 其它角色改编思路"
        text += f"\n\n{json_data['other_role_adaptation']}"
        return text

    def gen_adapted_role_infos(
        self,
        adaptation,
        story_arc_text,
        suggestion="暂无",
    ):
        """生成角色信息（同步方法，供线程池调用）。"""
        variables = {
            "StoryArcs": story_arc_text,
            "Adaptation": adaptation,
            "Suggestion": suggestion,
        }

        result = _query_llm_with_postprocess(
            ctx=self.ctx,
            task=Task(
                stage="adaptation",
                prompt_name="role_info",
                variables=variables,
                post_process_func=self._process_role_info_response,
                debug_info="generate role infos",
            ),
            model_name=self.cfg.model_name,
            max_retries=self.cfg.retry_cnt,
        )
        role_infos = RoleInfo(content=result)
        return role_infos

    async def generate_proposal(self, story_info, generate_input, suggestion="暂无"):
        try:
            _, story_arc_text = await self.data_loader.load_format_story_arcs(
                generate_input.novel_id
            )
            story_arc_text = story_arc_text[: self.cfg.max_word_cnt]
        except Exception as e:
            logger.error_context(self.ctx, f"Failed to load story arcs: {e}")
            raise RuntimeError(f"获取事件列表失败: {e}")

        try:
            adaptation = await self.generate_adaptation(
                story_info,
                generate_input,
                story_arc_text,
                suggestion=suggestion,
            )
        except Exception as e:
            logger.error_context(self.ctx, f"Failed to generate adaptation: {e}")
            raise

        try:
            # 使用线程池并行执行
            with ThreadPoolExecutor(max_workers=3) as executor:
                wb_future = executor.submit(
                    self.gen_adapted_worldbuilding,
                    adaptation=adaptation,
                    story_arc_text=story_arc_text,
                    suggestion=suggestion,
                )
                ri_future = executor.submit(
                    self.gen_adapted_role_infos,
                    adaptation=adaptation,
                    story_arc_text=story_arc_text,
                    suggestion=suggestion,
                )
                world_building = wb_future.result()
                role_infos = ri_future.result()

            # 统一上传
            for data in [world_building, role_infos]:
                await self.data_loader.upload(data, story_id=story_info.story_id)

        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Failed to generate world building: {e}",
            )
            raise
        return role_infos, world_building
