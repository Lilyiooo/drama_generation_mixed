from concurrent.futures import ThreadPoolExecutor

from trpc.log import logger
from trpc.exceptions import NewBusinessError

from service.drama_by_fiction.error_codes import BizCode, BIZ_MSG
from service.drama_by_fiction.data_model import FictionConfig, EpisodeOutline, Scene
from service.drama_by_fiction.core.data_manager import DataManager
from service.drama_by_fiction.utils import (
    _query_llm_with_postprocess,
    post_validate_json_schema,
    prompt_manager,
    Task,
    DETAIL_OUTLINE_SCHEMA,
    SCENE_PLAN_SCHEMA,
    SINGLE_SCENE_PLAN_SCHEMA,
)


class ScenePlanner:
    """专注于「分场规划」：批量生成分场计划、生成单场大纲。"""

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

    def _get_scene_range(self, words_range: tuple[int, int]) -> str:
        """根据字数范围计算场数"""
        WORDS_PER_SCENE = 500
        MIN_SCENE_CNT = 2
        MAX_SCENE_CNT = 40

        min_scene_cnt = max(int(words_range[0] / WORDS_PER_SCENE), MIN_SCENE_CNT)
        max_scene_cnt = min(int(words_range[1] / WORDS_PER_SCENE), MAX_SCENE_CNT)
        max_scene_cnt = max(max_scene_cnt, min_scene_cnt)
        return f"{min_scene_cnt}-{max_scene_cnt}场"

    def _build_event_plan_text(self, episode_proposal, season_proposal):
        """从 episode_proposal 和 season_proposal 构建本集情节规划文本"""
        # 构建 plot_id -> Plot 的映射
        plot_map = {plot.plot_id: plot for plot in season_proposal.plot_planning}

        lines = []
        lines.append(f"集标题：{episode_proposal.episode_title}")
        lines.append("")
        for plot_proposal in episode_proposal.plots:
            plot = plot_map.get(plot_proposal.plot_id)
            if not plot:
                continue
            lines.append(
                f"【情节 {plot.plot_id}】{plot.title} "
                f"章节范围：（§{plot.chapter_from}-§{plot.chapter_to}）"
            )
            if plot.description:
                lines.append(f"  {plot.description}")
            # 列出该情节下属于本集的事件
            event_map = {e.event_id: e for e in plot.events}
            for ep in plot_proposal.events:
                event = event_map.get(ep.event_id)
                if event:
                    lines.append(
                        f"  - 标题：{event.title} "
                        f"章节范围：（§{event.chapter_from}-§{event.chapter_to}）"
                    )
                    if event.description:
                        lines.append(f"    {event.description}")
            lines.append("")
        return "\n".join(lines).strip()

    def _gen_scene_plan_proc(self, task):
        epi_outline = task["epi_outline"]
        novel_text = task["NovelText"]
        if not novel_text:
            return epi_outline

        try:
            llm_task = Task(
                stage="plan",
                prompt_name="scene_plan",
                variables=task,
                debug_info="generate scene plan",
                post_process_func=post_validate_json_schema,
                post_process_kwargs={"schema": SCENE_PLAN_SCHEMA},
            )
            epi_plan = _query_llm_with_postprocess(
                ctx=self.ctx,
                task=llm_task,
                model_name=self.cfg.model_name,
                max_retries=self.cfg.retry_cnt,
            )

            epi_outline.scenes = [Scene.from_dict(x) for x in epi_plan]
            return epi_outline
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Error generating scene plan for task {task}: {e}",
            )
        return epi_outline

    def _gen_episode_outline_proc(self, task):
        epi_outline = task["epi_outline"]
        novel_text = task["NovelText"]
        if not novel_text:
            return epi_outline

        try:
            llm_task = Task(
                stage="plan",
                prompt_name="episode_outline",
                variables=task,
                debug_info="generate epi detail outline",
                post_process_func=post_validate_json_schema,
                post_process_kwargs={"schema": DETAIL_OUTLINE_SCHEMA},
            )
            rsp_data = _query_llm_with_postprocess(
                ctx=self.ctx,
                task=llm_task,
                model_name=self.cfg.model_name,
                max_retries=self.cfg.retry_cnt,
            )
            new_outline_text = rsp_data.get("outline", "").strip()
            if new_outline_text:
                epi_outline.content = new_outline_text
            return epi_outline
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Error generating episode outline for task {task}: {e}",
            )
        return epi_outline

    def _build_episode_task(
        self,
        season_id,
        episode_id,
        episode_proposal,
        season_proposal,
        novel_text,
        world_building,
        role_info,
        epi_outline=None,
        scene_range="暂无",
        suggestion="暂无",
    ):
        """为单集构建任务字典（供批量和单场复用）"""
        this_chapters = episode_proposal.get_chapters()

        event_plan_text = self._build_event_plan_text(episode_proposal, season_proposal)

        if epi_outline is None:
            epi_outline = EpisodeOutline(
                season_id=season_id,
                episode_id=episode_id,
                episode_title=episode_proposal.episode_title,
                ch_ranges=this_chapters,
                chapter_from=this_chapters[0] if this_chapters else 0,
                chapter_to=this_chapters[-1] if this_chapters else 0,
            )
        task = {
            "RoleInfos": role_info,
            "WorldBuilding": world_building,
            "NovelText": novel_text,
            "Suggestion": suggestion,
            "EventPlan": event_plan_text,
            "Outline": epi_outline.content,
            "SceneRange": scene_range,
            "epi_outline": epi_outline,
        }
        return task

    async def generate_episode_outlines(
        self,
        story_info,
        generate_input,
        season_id,
        episode_ids,
        world_building,
        role_info,
        season_proposal,
        suggestion="暂无",
    ):
        """批量生成指定季中多集的集大纲"""

        # 构建任务列表
        tasks = []
        episode_planning_map = {}
        for item in season_proposal.episode_planning:
            episode_id = int(item.episode_id)
            episode_planning_map[episode_id] = item

        for episode_id in episode_ids:
            if episode_id not in episode_planning_map:
                logger.warning_context(
                    self.ctx,
                    f"未找到{season_id}-{episode_id}的分集配置信息",
                )
                continue

            episode_proposal = episode_planning_map[episode_id]

            # 获取当前集的章节范围
            this_chapters = episode_proposal.get_chapters()

            if not this_chapters:
                logger.warning_context(
                    self.ctx, f"{season_id}-{episode_id} 配置的章节为空"
                )
                continue

            # 分层加载：核心章节原文 + 间隙章节摘要
            novel_text = await self.data_manager.load_contents_layered(
                generate_input.novel_id,
                key_chapters=this_chapters,
                max_length=self.cfg.max_word_cnt,
            )

            task = self._build_episode_task(
                season_id=season_id,
                episode_id=episode_id,
                episode_proposal=episode_proposal,
                season_proposal=season_proposal,
                novel_text=novel_text,
                world_building=world_building,
                role_info=role_info,
                suggestion=suggestion,
            )
            tasks.append(task)

        with ThreadPoolExecutor(max_workers=self.cfg.max_workers) as executor:
            results = list(executor.map(self._gen_episode_outline_proc, tasks))

        try:
            await self.upload_episode_outlines(story_info, results)
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Error uploading episode outlines for season {season_id}: {e}",
            )
            raise
        return results

    async def generate_scene_plans(
        self,
        story_info,
        generate_input,
        season_id,
        episode_ids,
        world_building,
        role_info,
        season_proposal,
        episode_outlines,
        suggestion="暂无",
    ):
        """批量生成指定季中多集的分场计划"""

        episode_outline_dict = {
            int(outline.episode_id): outline for outline in episode_outlines
        }

        # 计算场数范围
        scene_range = self._get_scene_range(
            [
                generate_input.min_word_count_per_episode,
                generate_input.max_word_count_per_episode,
            ]
        )

        # 构建任务列表
        tasks = []
        episode_planning_map = {}
        for item in season_proposal.episode_planning:
            episode_id = int(item.episode_id)
            episode_planning_map[episode_id] = item

        for episode_id in episode_ids:
            if episode_id not in episode_planning_map:
                logger.warning_context(
                    self.ctx,
                    f"未找到{season_id}-{episode_id}的分集配置信息",
                )
                continue

            if episode_id not in episode_outline_dict:
                logger.warning_context(
                    self.ctx,
                    f"未找到{season_id}-{episode_id}的集大纲信息",
                )
                continue

            episode_proposal = episode_planning_map[episode_id]
            episode_outline = episode_outline_dict[episode_id]

            # 获取当前集的章节范围
            this_chapters = episode_proposal.get_chapters()

            if not this_chapters:
                logger.warning_context(
                    self.ctx, f"{season_id}-{episode_id} 配置的章节为空"
                )
                continue

            # 分层加载：核心章节原文 + 间隙章节摘要
            novel_text = await self.data_manager.load_contents_layered(
                generate_input.novel_id,
                key_chapters=this_chapters,
                max_length=self.cfg.max_word_cnt,
            )

            task = self._build_episode_task(
                season_id=season_id,
                episode_id=episode_id,
                episode_proposal=episode_proposal,
                season_proposal=season_proposal,
                novel_text=novel_text,
                world_building=world_building,
                role_info=role_info,
                epi_outline=episode_outline,
                scene_range=scene_range,
                suggestion=suggestion,
            )
            tasks.append(task)

        with ThreadPoolExecutor(max_workers=self.cfg.max_workers) as executor:
            results = list(executor.map(self._gen_scene_plan_proc, tasks))

        try:
            await self.upload_episode_outlines(story_info, results)
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Error uploading episode outlines for season {season_id}: {e}",
            )
            raise
        return results

    def _gen_single_scene_plan_proc(self, task):
        """调用 LLM 生成单场大纲（使用专用 prompt，输出单个 JSON 对象）"""
        novel_text = task.get("NovelText")
        if not novel_text:
            return None
        try:
            llm_task = Task(
                stage="plan",
                prompt_name="single_scene_plan",
                variables=task,
                debug_info="generate single scene plan",
                post_process_func=post_validate_json_schema,
                post_process_kwargs={"schema": SINGLE_SCENE_PLAN_SCHEMA},
            )
            scene_plan = _query_llm_with_postprocess(
                ctx=self.ctx,
                task=llm_task,
                model_name=self.cfg.model_name,
                max_retries=self.cfg.retry_cnt,
            )
            return Scene.from_dict(scene_plan)
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Error generating single scene plan: {e}",
            )
        return None

    async def _build_single_scene_task(
        self,
        generate_input,
        season_id,
        episode_id,
        scene_index,
        world_building,
        role_info,
        season_proposal,
        episode_outline: EpisodeOutline,
        suggestion="暂无",
    ):
        """构建单场大纲生成所需的任务变量字典（公共逻辑）。"""
        existing_scenes = episode_outline.scenes or []
        if scene_index < 0 or scene_index >= len(existing_scenes):
            raise NewBusinessError(
                BizCode.PARAM_INVALID,
                f"{BIZ_MSG[BizCode.PARAM_INVALID]}: scene_index={scene_index} 超出范围，"
                f"当前集共 {len(existing_scenes)} 场",
            )

        # 从 season_proposal 中获取本集的 episode_proposal
        episode_planning_map = {
            int(item.episode_id): item for item in season_proposal.episode_planning
        }
        episode_proposal = episode_planning_map.get(episode_id)
        if not episode_proposal:
            raise NewBusinessError(
                BizCode.PARAM_INVALID,
                f"{BIZ_MSG[BizCode.PARAM_INVALID]}: 未找到 {season_id+1}-{episode_id+1} 的分集配置信息",
            )

        this_chapters = episode_proposal.get_chapters()
        if not this_chapters:
            raise NewBusinessError(
                BizCode.PARAM_INVALID,
                f"{BIZ_MSG[BizCode.PARAM_INVALID]}: {season_id+1}-{episode_id+1} 配置的章节为空",
            )

        # 加载小说文本
        novel_text = await self.data_manager.load_contents_layered(
            generate_input.novel_id,
            key_chapters=this_chapters,
            max_length=self.cfg.max_word_cnt,
        )

        # 构建已有场次的上下文文本（兼容未生成的空场次）
        other_scenes_text = []
        for idx, scene in enumerate(existing_scenes):
            if idx == scene_index:
                other_scenes_text.append(f"【第{idx + 1}场 — 待重新生成】")
            else:
                title, content = scene.title, scene.content
                scene_text = (
                    f"【第{idx + 1}场】{title}\n{content}"
                    if content
                    else f"【第{idx + 1}场 — 尚未生成】"
                )
                other_scenes_text.append(scene_text)

        other_scenes_joined = (
            "\n\n".join(other_scenes_text)
            if other_scenes_text
            else "暂无已有分场（本场为首次生成）"
        )

        # 构建单场专用的任务变量
        task = {
            "RoleInfos": role_info,
            "WorldBuilding": world_building,
            "NovelText": novel_text,
            "Suggestion": suggestion,
            "Outline": episode_outline.content,
            "OtherScenes": other_scenes_joined,
            "TargetSceneIndex": scene_index + 1,
        }
        return task

    async def generate_single_scene_outline(
        self,
        story_info,
        generate_input,
        season_id,
        episode_id,
        scene_index,
        world_building,
        role_info,
        season_proposal,
        episode_outline: EpisodeOutline,
        suggestion="暂无",
    ):
        """重新生成某季某集中指定场次（scene_index, 0-based）的场大纲，其余场次保持不变。

        使用专用的 single_scene_plan prompt，只让 LLM 输出指定的那一场，
        而非生成整集所有分场后再截取。
        """
        task = await self._build_single_scene_task(
            generate_input,
            season_id,
            episode_id,
            scene_index,
            world_building,
            role_info,
            season_proposal,
            episode_outline,
            suggestion=suggestion,
        )

        # 使用单场专用的处理方法
        scene_plan = self._gen_single_scene_plan_proc(task)

        existing_scenes = episode_outline.scenes or []
        if scene_plan:
            existing_scenes_copy = list(existing_scenes)
            existing_scenes_copy[scene_index] = scene_plan
            episode_outline.scenes = existing_scenes_copy
        else:
            scene_plan = episode_outline.scenes[scene_index]
            logger.warning_context(
                self.ctx,
                f"单场大纲生成失败，season={season_id}, episode={episode_id}, "
                f"scene_index={scene_index}，保持原有内容不变",
            )

        # 上传更新后的大纲
        await self.data_manager.upload(
            episode_outline,
            story_id=story_info.story_id,
            season_id=season_id,
            episode_id=episode_id,
        )

        return scene_plan

    async def build_single_scene_outline_prompt(
        self,
        story_info,
        generate_input,
        season_id,
        episode_id,
        scene_index,
        world_building,
        role_info,
        season_proposal,
        episode_outline: EpisodeOutline,
        suggestion="暂无",
    ) -> str:
        """构建单场大纲生成的完整提示词，不调用 LLM。"""
        task = await self._build_single_scene_task(
            generate_input,
            season_id,
            episode_id,
            scene_index,
            world_building,
            role_info,
            season_proposal,
            episode_outline,
            suggestion=suggestion,
        )
        prompt = prompt_manager.format_prompt(
            stage="plan",
            name="single_scene_plan",
            variables=task,
        )
        return prompt.strip()
