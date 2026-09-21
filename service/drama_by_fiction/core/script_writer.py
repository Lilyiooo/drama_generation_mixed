from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from trpc.log import logger
from trpc.exceptions import NewBusinessError

from service.drama_by_fiction.error_codes import BizCode, BIZ_MSG
from service.drama_by_fiction.data_model import (
    FictionConfig,
    StoryInfo,
    GenerateInput,
    SeasonProposal,
    EpisodeOutline,
    EpisodeScript,
    SceneScript,
)
from service.drama_by_fiction.core.data_manager import DataManager
from service.drama_by_fiction.utils import (
    _query_llm_with_postprocess,
    prompt_manager,
    Task,
)


class ScriptWriter:
    def __init__(self, ctx):
        self.data_manager = DataManager(ctx)
        self.cfg = FictionConfig()
        self.ctx = ctx

    @staticmethod
    def _get_scene_word_hint(plot_type: str) -> str:
        """根据剧本类型返回单场字数参考基准。

        仅作为 LLM 的参考提示，实际字数由模型根据场大纲内容自行决定。
        """
        if plot_type == "CARTOON":
            return "单场参考字数约450-600字（动漫节奏，请根据本场大纲实际情节酌情调整）"
        else:
            return "单场参考字数约250-450字（请根据本场大纲实际情节酌情调整）"

    async def upload_episode_scripts(self, story_id: str, episode_scripts):
        for script in episode_scripts:
            await self.data_manager.upload(
                script,
                story_id=story_id,
                season_id=script.season_id,
                episode_id=script.episode_id,
            )

    async def _build_tasks_for_season(
        self,
        novel_id,
        season_id,
        episode_ids,
        role_info,
        world_building,
        scene_word_hint,
        episode_outline_dict,
        proposal,
        suggestion,
    ):
        """为指定season构建任务列表"""
        tasks = []

        episode_planning_map = {
            int(item.episode_id): item for item in proposal.episode_planning
        }

        logger.info_context(
            self.ctx,
            f"episode_planning_map keys: {list(episode_planning_map.keys())},"
            f"episode_outline_dict keys: {list(episode_outline_dict.keys())}, "
            f"episode_ids: {list(episode_ids)}",
        )

        for episode_id in episode_ids:
            prev_episode_id = episode_id - 1

            prev_summary = (
                episode_outline_dict[episode_id - 1].content
                if episode_id > 0 and prev_episode_id in episode_outline_dict
                else "暂无"
            )

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
                novel_id,
                key_chapters=this_chapters,
                max_length=self.cfg.max_word_cnt,
            )

            scenes = episode_outline_dict[episode_id].scenes

            task = {
                "SeasonIndex": season_id + 1,
                "EpisodeIndex": episode_id + 1,
                "RoleInfo": role_info,
                "WorldBuilding": world_building,
                "PrevSummary": prev_summary,
                "Outline": episode_outline_dict[episode_id].content,
                "Suggestion": suggestion,
                "NovelText": novel_text,
                "SceneWordHint": scene_word_hint,
                "AdaptationProposal": proposal.other_adaptation_proposal,
                "EpisodeNum": int(episode_id) + 1,
                "scenes": scenes,
            }
            tasks.append(task)

        return tasks

    def _build_scene_variables(self, task_variables, scene_outline, scene_index):
        """构建单场剧本生成的公共变量（不含上下文部分）

        :param task_variables: 包含公共变量的字典
        :param scene_outline: 当前场次的大纲内容
        :param scene_index: 当前场次索引（从1开始）
        :return: 包含场次专用变量的字典
        """
        scene_variables = task_variables.copy()
        scene_variables["SceneOutline"] = scene_outline
        scene_variables["SceneIndex"] = scene_index
        return scene_variables

    def _call_scene_llm(self, scene_variables):
        """调用 LLM 生成单场剧本

        :param scene_variables: 已填充完毕的场次变量字典
        :return: 当前场次的剧本文本
        """
        episode_index = scene_variables["EpisodeIndex"]
        scene_index = scene_variables["SceneIndex"]

        scene_script = _query_llm_with_postprocess(
            ctx=self.ctx,
            task=Task(
                stage="write",
                prompt_name="scene_script",
                variables=scene_variables,
                debug_info=(
                    f"generate scene {scene_index} script for episode {episode_index}"
                ),
            ),
            model_name=self.cfg.model_name,
            max_retries=self.cfg.retry_cnt,
        )
        scene_lines = [line for line in scene_script.splitlines() if line.strip()]
        scene_script = "\n".join(scene_lines)
        return scene_script

    def _generate_scene_for_episode(
        self,
        task_variables,
        scene_outline,
        scene_index,
        generated_script_so_far,
    ):
        """整集串行生成时，生成单场剧本。

        逐场顺序生成，只需参考前面已经累积的剧本，
        后续场还未生成，FollowingScriptReference 固定为 "暂无"。

        :param task_variables: 包含公共变量的字典
        :param scene_outline: 当前场次的大纲内容
        :param scene_index: 当前场次索引（从1开始）
        :param generated_script_so_far: 前序场次已累积的剧本文本
        :return: 当前场次的剧本文本
        """
        scene_variables = self._build_scene_variables(
            task_variables, scene_outline, scene_index
        )
        scene_variables["GeneratedScriptSoFar"] = (
            generated_script_so_far
            if generated_script_so_far
            else ("暂无（当前为本集第一场）" if scene_index == 1 else "暂无")
        )
        scene_variables["FollowingScriptReference"] = "暂无"

        return self._call_scene_llm(scene_variables)

    def _generate_scene_standalone(
        self,
        task_variables,
        scene_outline,
        scene_index,
        generated_script_so_far,
        following_script_reference="",
    ):
        """外部接口直接生成某一场时调用。

        可能需要同时参考本集已生成的所有场（前置场 + 后续场），
        适用于重新生成某场或跳场生成的场景。

        :param task_variables: 包含公共变量的字典
        :param scene_outline: 当前场次的大纲内容
        :param scene_index: 当前场次索引（从1开始）
        :param generated_script_so_far: 前置场的剧本文本
        :param following_script_reference: 后续场已有的剧本文本
        :return: 当前场次的剧本文本
        """
        scene_variables = self._build_scene_variables(
            task_variables, scene_outline, scene_index
        )
        scene_variables["GeneratedScriptSoFar"] = (
            generated_script_so_far
            if generated_script_so_far
            else ("暂无（当前为本集第一场）" if scene_index == 1 else "暂无")
        )
        scene_variables["FollowingScriptReference"] = (
            following_script_reference if following_script_reference else "暂无"
        )

        return self._call_scene_llm(scene_variables)

    def _generate_episode_script(self, task):
        """按场串行生成单个剧集的完整剧本

        遍历本集所有场次大纲，逐场调用LLM生成剧本，
        每次生成下一场时将已生成的内容送入上下文。
        """
        try:
            result = {
                "season_id": int(task["SeasonIndex"]) - 1,
                "episode_id": int(task["EpisodeIndex"]) - 1,
                "script_content": "",
                "scenes": [],
            }

            # 构建公共任务变量
            task_variables = task.copy()
            scenes = task.get("scenes", [])

            # 如果没有场次大纲，返回空的集剧本
            if not scenes:
                logger.warning_context(
                    self.ctx,
                    f"No scene outlines for episode {task['EpisodeIndex']}, "
                    f"returning empty episode script.",
                )
                return EpisodeScript.from_dict(result)

            # 按场串行生成
            generated_parts = []
            generated_scenes = []
            for idx, scene in enumerate(scenes):
                scene_index = idx + 1
                title, content = scene.title, scene.content
                scene_outline = f"{title}\n{content}"

                logger.info_context(
                    self.ctx,
                    f"Generating scene {scene_index}/{len(scenes)}"
                    f" for episode {task['EpisodeIndex']}",
                )

                generated_so_far = "\n\n".join(generated_parts)
                scene_script = self._generate_scene_for_episode(
                    task_variables=task_variables,
                    scene_outline=scene_outline,
                    scene_index=scene_index,
                    generated_script_so_far=generated_so_far,
                )
                generated_parts.append(scene_script)
                scene_item = {
                    "scene_id": idx,
                    "scene_title": title,
                    "script_content": scene_script,
                }
                generated_scenes.append(scene_item)

                logger.info_context(
                    self.ctx,
                    f"Scene {scene_index}/{len(scenes)} generated"
                    f" for episode {task['EpisodeIndex']},"
                    f" length: {len(scene_script)}",
                )
            result["scenes"] = generated_scenes
            result["script_content"] = "\n\n".join(generated_parts)
            script = EpisodeScript.from_dict(result)
            return script
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Error generating episode {task.get('EpisodeIndex', 'unknown')}: {e}",
            )

        return EpisodeScript.from_dict(result)

    async def generate_episode_scripts(
        self,
        story_info: StoryInfo,
        generate_input: GenerateInput,
        season_id,
        episode_ids,
        world_building,
        role_info,
        season_proposal: SeasonProposal,
        episode_outlines,
        suggestion="暂无",
    ):
        # 准备公共数据
        scene_word_hint = self._get_scene_word_hint(story_info.plot_type)

        logger.info_context(self.ctx, f"scene_word_hint: {scene_word_hint}")

        episode_outline_dict = {
            outline.episode_id: outline for outline in episode_outlines
        }

        # 构建任务列表
        generate_tasks = await self._build_tasks_for_season(
            generate_input.novel_id,
            season_id,
            episode_ids,
            role_info,
            world_building,
            scene_word_hint,
            episode_outline_dict,
            season_proposal,
            suggestion,
        )
        logger.info_context(
            self.ctx, f"gen script SE{season_id+1} task count: {len(generate_tasks)}"
        )

        # 各集之间并行，每集内部按场串行生成
        with ThreadPoolExecutor(max_workers=self.cfg.max_workers) as executor:
            results = executor.map(self._generate_episode_script, generate_tasks)
            valid_results = [
                result for result in tqdm(results, total=len(generate_tasks))
            ]

        await self.upload_episode_scripts(story_info.story_id, valid_results)

        return valid_results

    async def _build_single_scene_variables(
        self,
        story_info: StoryInfo,
        generate_input: GenerateInput,
        season_id: int,
        episode_id: int,
        scene_index: int,
        world_building,
        role_info,
        season_proposal: SeasonProposal,
        episode_outline: EpisodeOutline,
        existing_episode_script: EpisodeScript = None,
        suggestion="暂无",
    ):
        """构建单场剧本生成所需的场次变量字典（公共逻辑）。

        :return: (scene_variables, title) 元组
        """
        scenes = episode_outline.scenes or []
        if scene_index < 0 or scene_index >= len(scenes):
            raise NewBusinessError(
                BizCode.PARAM_INVALID,
                f"{BIZ_MSG[BizCode.PARAM_INVALID]}: scene_index={scene_index} 超出范围，"
                f"当前集共 {len(scenes)} 场",
            )

        target_scene = scenes[scene_index]
        title = (
            target_scene.title
            if hasattr(target_scene, "title")
            else target_scene.get("title", "")
        )
        content = (
            target_scene.content
            if hasattr(target_scene, "content")
            else target_scene.get("content", "")
        )
        scene_outline_text = f"{title}\n{content}"

        # 构建前后文上下文，两级降级策略
        generated_script_so_far = ""
        following_script_reference = ""

        if existing_episode_script and existing_episode_script.scenes:
            existing_scenes = existing_episode_script.scenes

            # 提取前置场剧本
            if scene_index > 0:
                prev_parts = []
                for idx in range(scene_index):
                    if idx < len(existing_scenes):
                        existing_scene = existing_scenes[idx]
                        scene_content = (
                            existing_scene.script_content
                            if hasattr(existing_scene, "script_content")
                            else existing_scene.get("script_content", "")
                        )
                        if scene_content:
                            prev_parts.append(scene_content)
                if prev_parts:
                    generated_script_so_far = "\n\n".join(prev_parts)

            # 提取后续场剧本
            if scene_index + 1 < len(existing_scenes):
                following_parts = []
                for idx in range(scene_index + 1, len(existing_scenes)):
                    existing_scene = existing_scenes[idx]
                    scene_content = (
                        existing_scene.script_content
                        if hasattr(existing_scene, "script_content")
                        else existing_scene.get("script_content", "")
                    )
                    if scene_content:
                        following_parts.append(scene_content)
                if following_parts:
                    following_script_reference = "\n\n".join(following_parts)

        if not generated_script_so_far and scene_index > 0:
            # 最终兜底：用前置场大纲摘要代替
            prev_outlines = []
            for idx in range(scene_index):
                s = scenes[idx]
                s_title = s.title if hasattr(s, "title") else s.get("title", "")
                s_content = s.content if hasattr(s, "content") else s.get("content", "")
                prev_outlines.append(f"【第{idx + 1}场】{s_title}\n{s_content}")
            generated_script_so_far = (
                "（以下为前序场次的大纲摘要，非完整剧本，请据此衔接当前场次）\n\n"
                + "\n\n".join(prev_outlines)
            )

        # 准备公共变量
        scene_word_hint = self._get_scene_word_hint(story_info.plot_type)

        # 从 season_proposal 中获取本集的 episode_proposal
        episode_planning_map = {
            int(item.episode_id): item for item in season_proposal.episode_planning
        }
        episode_proposal = episode_planning_map.get(episode_id)

        # 加载小说文本
        novel_text = ""
        if episode_proposal:
            this_chapters = episode_proposal.get_chapters()
            if this_chapters:
                novel_text = await self.data_manager.load_contents_layered(
                    generate_input.novel_id,
                    key_chapters=this_chapters,
                    max_length=self.cfg.max_word_cnt,
                )

        task_variables = {
            "SeasonIndex": season_id + 1,
            "EpisodeIndex": episode_id + 1,
            "RoleInfo": role_info,
            "WorldBuilding": world_building,
            "PrevSummary": "暂无",
            "Outline": episode_outline.content,
            "Suggestion": suggestion,
            "NovelText": novel_text,
            "SceneWordHint": scene_word_hint,
            "AdaptationProposal": season_proposal.other_adaptation_proposal,
            "EpisodeNum": episode_id + 1,
        }

        scene_variables = self._build_scene_variables(
            task_variables, scene_outline_text, scene_index + 1
        )
        scene_variables["GeneratedScriptSoFar"] = (
            generated_script_so_far
            if generated_script_so_far
            else ("暂无（当前为本集第一场）" if scene_index == 0 else "暂无")
        )
        scene_variables["FollowingScriptReference"] = (
            following_script_reference if following_script_reference else "暂无"
        )

        return scene_variables, title

    async def generate_single_scene_script(
        self,
        story_info: StoryInfo,
        generate_input: GenerateInput,
        season_id: int,
        episode_id: int,
        scene_index: int,
        world_building,
        role_info,
        season_proposal: SeasonProposal,
        episode_outline: EpisodeOutline,
        existing_episode_script: EpisodeScript = None,
        suggestion="暂无",
    ) -> SceneScript:
        """生成某季某集某场（scene_index, 0-based）的场剧本。"""
        scene_variables, title = await self._build_single_scene_variables(
            story_info,
            generate_input,
            season_id,
            episode_id,
            scene_index,
            world_building,
            role_info,
            season_proposal,
            episode_outline,
            existing_episode_script,
            suggestion=suggestion,
        )

        scene_script_text = self._call_scene_llm(scene_variables)

        result = SceneScript(
            scene_id=scene_index,
            scene_title=title,
            script_content=scene_script_text,
        )

        return result

    async def build_single_scene_script_prompt(
        self,
        story_info: StoryInfo,
        generate_input: GenerateInput,
        season_id: int,
        episode_id: int,
        scene_index: int,
        world_building,
        role_info,
        season_proposal: SeasonProposal,
        episode_outline: EpisodeOutline,
        existing_episode_script: EpisodeScript = None,
        suggestion="暂无",
    ) -> str:
        """构建单场剧本生成的完整提示词，不调用 LLM。"""
        scene_variables, _ = await self._build_single_scene_variables(
            story_info,
            generate_input,
            season_id,
            episode_id,
            scene_index,
            world_building,
            role_info,
            season_proposal,
            episode_outline,
            existing_episode_script,
            suggestion=suggestion,
        )
        prompt = prompt_manager.format_prompt(
            stage="write",
            name="scene_script",
            variables=scene_variables,
        )
        return prompt.strip()
