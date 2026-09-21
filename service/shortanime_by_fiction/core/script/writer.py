import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import json_repair
from tqdm import tqdm
from trpc.log import logger
from trpc_script_drama_operator import pb

from service.shortanime_by_fiction.data_models.drama import (
    EpisodeScript,
    EpisodeOutline,
)
from service.shortanime_by_fiction.data_models.fiction import StoryInfo
from service.shortanime_by_fiction.configs import FictionConfig, model_cfg
from service.shortanime_by_fiction.data_manager import DataManager
from service.shortanime_by_fiction.llms import (
    query_llm,
    Task,
    ContextLimitExceededError,
)
from service.shortanime_by_fiction.core.script.script_validator import WordCountValidator
# from service.shortanime_by_fiction.narrative.dnmg.state_manager import MemoryStateManager
from trpc.log import logger as _trpc_logger


class ScriptWriter:
    def __init__(self, ctx):
        self.data_loader = DataManager(ctx)
        self.cfg = FictionConfig()
        self.ctx = ctx
        self.wc_validator = WordCountValidator(ctx, self.cfg)
        # self.memory_state_manager = self._build_memory_state_manager() if self.cfg.use_memory_graph else None

    # def _build_memory_state_manager(self) -> MemoryStateManager:
    #     """构造 MemoryStateManager，将项目依赖注入（adapter 层）。"""
    #     ctx = self.ctx
    #     cfg = self.cfg

    #     def llm_caller(stage: str, prompt_name: str, variables: dict, domain: str, debug_info: str = "") -> str:
    #         from service.shortanime_by_fiction.llms import query_llm, Task
    #         model_name = (cfg.extract_model_name or cfg.model_name) if stage == "dnmg" else cfg.model_name
    #         _trpc_logger.info(f"[llm_caller] stage={stage} prompt={prompt_name} model={model_name}")
    #         return query_llm(
    #             ctx=ctx,
    #             task=Task(
    #                 stage=stage,
    #                 prompt_name=prompt_name,
    #                 variables=variables,
    #                 domain=domain,
    #                 debug_info=debug_info,
    #             ),
    #             model_name=model_name,
    #             max_retries=cfg.retry_cnt,
    #         )

    #     return MemoryStateManager(
    #         llm_caller=llm_caller,
    #         cos_client=self.data_loader.cos,
    #         cfg=cfg,
    #         logger=_trpc_logger,
    #         ctx=ctx,
    #     )

    def _post_process_script(self, response: str, prev_script: str = "暂无", follow_script: str = "暂无") -> dict:
        data = json_repair.loads(response)
        parsed = {
            "script_content": data.get("content", "").strip(),
            "word_tier": data.get("word_tier", ""),
            "tier_reason": data.get("tier_reason", ""),
        }

        try:
            min_word, max_word = int(data["min_word"]), int(data["max_word"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"min_word/max_word 非法: {e}")

        if min_word > max_word:
            raise ValueError(f"min_word/max_word 范围非法: min_word={min_word} > max_word={max_word}")

        if self.cfg.enable_script_fix:
            parsed["script_content"] = self.wc_validator.apply_fix(
                parsed["script_content"], min_word, max_word,
                min_ratio=0.6, prev_script=prev_script, follow_script=follow_script,
            )
        return parsed

    def _get_prompt_name(self, episode_index):
        """根据剧集索引获取对应的prompt名称"""
        if int(episode_index) - 1 == 0:
            return "first_episode_script"
        else:
            return "episode_script"

    def _get_scene_and_word_range(self, plot_type, min_word_count: int, max_word_count: int):
        import math

        if plot_type == 'SHORT_CARTOON':
            min_scene_cnt = int(1 + (min_word_count - 800) / 1000)
            max_scene_cnt = math.ceil(2 + (max_word_count - 1200) / 700)
        else:
            min_scene_cnt = int(1 + (min_word_count - 1000) / 1000)
            max_scene_cnt = math.ceil(2 + (max_word_count - 1400) / 700)

        min_scene_cnt = max(min_scene_cnt, 1)
        max_scene_cnt = min(max_scene_cnt, 40)

        min_word_count = max(min_word_count, 500)
        max_word_count = min(max_word_count, 5000)

        scene_range = f"{min_scene_cnt}-{max_scene_cnt}场"
        word_range = f"{min_word_count}-{max_word_count}字"
        return scene_range, word_range

    async def upload_episode_scripts(self, story_info: StoryInfo, episode_scripts):
        for script in episode_scripts:
            await self.data_loader.upload(
                script,
                story_info=story_info,
                season_id=script.season_id,
                episode_id=script.episode_id,
            )

    def _generate_single_script(self, task, story_info, generate_input=None):
        """生成单个剧集脚本的核心函数"""
        try:
            episode_index = task["EpisodeIndex"]
            result = {
                "season_id": int(task["SeasonIndex"]) - 1,
                "episode_id": int(episode_index) - 1,
                "script_content": "",
            }

            task_variables = task.copy()

            prompt_name = self._get_prompt_name(episode_index)

            script = query_llm(
                ctx=self.ctx,
                task=Task(
                    stage="write",
                    prompt_name=prompt_name,
                    variables=task_variables,
                    domain=story_info.plot_type,
                    debug_info=f"generate episode script for episode {episode_index}",
                    post_process_func=self._post_process_script,
                    post_process_kwargs={
                        "prev_script": task_variables.get("PreviousEpisodeScript", "暂无"),
                        "follow_script": task_variables.get("FollowEpisodeScript", "暂无"),
                    },
                    extra={"story_info": story_info.__dict__, "generate_input": generate_input.__dict__ if generate_input is not None else {}},
                ),
                model_name=self.cfg.model_name,
                max_retries=self.cfg.retry_cnt,
            )
            result.update(script)
            script = EpisodeScript.from_dict(result)

        except Exception as e:
            logger.error_context(self.ctx, f"Error generating episode {task.get('EpisodeIndex', 'unknown')}: {e}")
            raise

        return script

    async def _build_tasks_for_season(
        self,
        story_info: StoryInfo,
        novel_id,
        season_id,
        episode_ids,
        role_info,
        world_building,
        scene_range,
        word_range,
        episode_outline_dict,
        proposal,
        suggestion,
        prev_summary='暂无',
        previous_script=None,
        follow_script=None,
        state_bars: dict = None,
    ):
        """为指定season构建任务列表"""
        state_bars = state_bars or {}
        tasks = []
        for episode_id in episode_ids:

            # prev_summary = episode_outline_dict[episode_id - 1].content if episode_id > 0 else "暂无"
            chapter_from = episode_outline_dict[episode_id].chapter_from
            chapter_to = episode_outline_dict[episode_id].chapter_to

            if chapter_from == 0 and chapter_to == 0:
                content = ""
            else:
                content = await self.data_loader.load_chapter_contents_text(
                    novel_id, chapter_from, chapter_to
                )
            
            content = content[: model_cfg.summary_truncate_length]
            task = {
                "SeasonIndex": season_id + 1,
                "EpisodeIndex": episode_id + 1,
                "RoleInfo": role_info,
                "WorldBuilding": world_building,
                "PrevSummary": prev_summary,
                "PreviousEpisodeScript": previous_script.script_content if previous_script is not None else "暂无",
                "FollowEpisodeScript": follow_script.script_content if follow_script is not None else "暂无",
                "Outline": episode_outline_dict[episode_id].content,
                "Suggestion": suggestion,
                "NovelText": content,
                "SceneRange": scene_range,
                "WordRange": word_range,
                "AdaptationProposal": proposal.other_adaptation_proposal,
                "EpisodeNum": int(episode_id) + 1,
                "StateBar": state_bars.get(episode_id, "暂无"),
            }
            tasks.append(task)

        return tasks

    # async def _generate_parallel(
    #     self,
    #     story_info,
    #     generate_input,
    #     season_id,
    #     episode_ids,
    #     role_info,
    #     world_building,
    #     scene_range,
    #     word_range,
    #     episode_outline_dict,
    #     season_proposal,
    #     suggestion,
    #     drama=None,
    #     **_kwargs,
    # ):
    #     """并行模式：使用 ThreadPoolExecutor 并发生成所有剧集剧本。"""
    #     generate_tasks = await self._build_tasks_for_season(
    #         story_info,
    #         generate_input.novel_id,
    #         season_id,
    #         episode_ids,
    #         role_info,
    #         world_building,
    #         scene_range,
    #         word_range,
    #         episode_outline_dict,
    #         season_proposal,
    #         suggestion,
    #     )
    #     logger.info_context(self.ctx, f"gen script SE{season_id+1} task count: {len(generate_tasks)}")

    #     # 从 drama 构建已有脚本查找表
    #     scripts_dict = {}
    #     if drama:
    #         for season in drama.seasons:
    #             if season.season_id == season_id:
    #                 for ep in season.episodes:
    #                     scripts_dict[ep.episode_id] = ep.script_content

    #     def generate_func(task):
    #         ep_id = task["EpisodeIndex"] - 1
    #         prev = scripts_dict.get(ep_id - 1, "暂无")
    #         follow = scripts_dict.get(ep_id + 1, "暂无")
    #         return self._generate_single_script(task, story_info, generate_input, prev, follow)

    #     with ThreadPoolExecutor(max_workers=model_cfg.max_workers) as executor:
    #         results = executor.map(generate_func, generate_tasks)
    #         valid_results = [result for result in tqdm(results, total=len(generate_tasks))]
    #     return valid_results

    async def _generate_sequential(
        self,
        story_info,
        generate_input,
        season_id,
        episode_ids,
        role_info,
        world_building,
        scene_range,
        word_range,
        season_proposal,
        suggestion,
        episode_outline_dict,
        scripts_dict=None,
        run_id: str = None,
        **_kwargs,
    ):
        """逐集串行模式：按集序依次生成，将上一集剧本内容传递给下一集。
        若启用记忆图谱（use_memory_graph=True），每集生成前额外检索状态条，生成后更新图谱。
        """
        use_memory = self.cfg.use_memory_graph and self.memory_state_manager is not None
        valid_results = []

        # 备份 scripts_dict 以便动态更新
        scripts_dict_copy = dict(scripts_dict or {})

        # sequential 生成前从 demo 快照恢复，确保每次从相同基准状态出发
        if use_memory:
            self.memory_state_manager.restore_from_demo_snapshot(story_info, season_id)

        for episode_id in tqdm(sorted(episode_ids), desc=f"Generating SE{season_id+1} scripts (sequential)"):
            state_bars = {}

            #  ---- 记忆管理 ---
            prev_summary = "暂无"
            previous_script = scripts_dict_copy.get(episode_id - 1, None)
            follow_script = scripts_dict_copy.get(episode_id + 1, None)

            if use_memory:
                outline_text = episode_outline_dict[episode_id].content
                state_bar = await self.memory_state_manager.retrieve_state_bar(
                    story_info=story_info,
                    season_id=season_id,
                    episode_id=episode_id,
                    outline_text=outline_text,
                )
                logger.info_context(
                    self.ctx,
                    f"[MemoryGraph] Retrieved state_bar for SE{season_id+1}EP{episode_id+1}"
                )
                logger.debug_context(
                    self.ctx,
                    f"[MemoryGraph] state_bar content for SE{season_id+1}EP{episode_id+1}:\n{state_bar}"
                )
                state_bars = {episode_id: state_bar}

            tasks = await self._build_tasks_for_season(
                story_info,
                generate_input.novel_id,
                season_id,
                [episode_id],
                role_info,
                world_building,
                scene_range,
                word_range,
                episode_outline_dict,
                season_proposal,
                suggestion,
                prev_summary,
                previous_script,
                follow_script,
                state_bars=state_bars,
            )
            task = tasks[0]
            script = self._generate_single_script(task, story_info, generate_input)
            valid_results.append(script)

            # 动态更新 scripts_dict_copy，使下一集能获取本集生成内容
            scripts_dict_copy[episode_id] = script

            if use_memory:
                await self.memory_state_manager.update_state(
                    story_info=story_info,
                    season_id=season_id,
                    episode_id=episode_id,
                    script_content=script.script_content,
                    run_id=run_id,
                )
        return valid_results

    async def generate_episode_scripts(
        self,
        story_info,
        generate_input,
        season_id,
        episode_ids,
        world_building,
        role_info,
        chapter_from,
        chapter_to,
        season_proposal,
        story_outline,
        episode_outline_dict,
        scripts_dict,
        suggestion='暂无',
        run_id: str = None
    ):
        scene_range, word_range = self._get_scene_and_word_range(story_info.plot_type, generate_input.min_word_count_per_episode, generate_input.max_word_count_per_episode)
        logger.info_context(self.ctx, f"scene_range: {scene_range}, word_range: {word_range}")

        method = self.cfg.script_generate_method
        logger.info_context(self.ctx, f"gen script SE{season_id+1} episode_ids: {sorted(episode_ids)}, method: {method}")

        common_kwargs = dict(
            story_info=story_info,
            generate_input=generate_input,
            season_id=season_id,
            episode_ids=episode_ids,
            role_info=role_info,
            world_building=world_building,
            scene_range=scene_range,
            word_range=word_range,
            season_proposal=season_proposal,
            suggestion=suggestion,
            episode_outline_dict=episode_outline_dict,
            scripts_dict=scripts_dict,
            run_id=run_id
        )

        if method != 'sequential':
            raise ValueError("错误的生成模式")

        valid_results = await self._generate_sequential(**common_kwargs)

        await self.upload_episode_scripts(story_info, valid_results)

        return valid_results

    # async def refine_episode_scripts(
    #     self,
    #     story_info: StoryInfo,
    #     novel_id,
    #     min_word_count_per_episode,
    #     max_word_count_per_episode,
    #     episode_nums,
    #     role_infos,
    #     seasons,
    #     select_ranges,
    #     suggestion='暂无'
    # ):
    #     world_building = await self.data_loader.load_world_building(story_info)
    #     adaptation = await self.data_loader.load_adaptation_proposal(story_info)

    #     results = []
    #     for select_range in select_ranges:

    #         for episode_id in select_range.episode_ids:
    #             prev_summary = await self._load_prev_summary(story_info, select_range.season_id, episode_id)
    #             follow_summary = await self._load_follow_summary(story_info, select_range.season_id, episode_id, episode_nums)
    #             episode_outline = await self.data_loader.load_episode_outline(story_info, select_range.season_id, episode_id)

    #             season_outline = await self.data_loader.load_season_outline(story_info, select_range.season_id)

    #             start = seasons[select_range.season_id].episodes[episode_id].chapter_from
    #             end = seasons[select_range.season_id].episodes[episode_id].chapter_to
    #             chapter_content = await self.data_loader.load_chapter_contents(
    #                 novel_id, start, end
    #             )
    #             origin_script = await self.data_loader.load_episode_script(
    #                         story_info, select_range.season_id, episode_id
    #                     )
    #             follow_script = await self.data_loader.load_episode_script(
    #                         story_info, select_range.season_id, episode_id + 1
    #                     )
    #             scene_range, word_range = self._get_scene_and_word_range(story_info.plot_type, min_word_count_per_episode, max_word_count_per_episode)

    #             variables = {
    #                 "WorldBuilding": world_building.content,
    #                 "RoleInfo": role_infos,
    #                 "SeasonOutline": '',
    #                 "EpisodeOutline": episode_outline.content,
    #                 "PrevSummary": prev_summary,
    #                 "FollowScript": follow_script.script_content,
    #                 "NovelText": chapter_content,
    #                 "OriginScript": origin_script.script_content,
    #                 "EpisodeNum": episode_id + 1,
    #                 "SceneRange": scene_range,
    #                 "WordRange": word_range
    #             }
    #             if suggestion:
    #                 variables["Suggestion"] = suggestion

    #             script = query_llm(
    #                 ctx=self.ctx,
    #                 task=Task(
    #                     stage="write",
    #                     prompt_name="refine_episode_script",
    #                     variables=variables,
    #                     debug_info=f"refine_episode_script",
    #                     extra={"story_info": story_info.__dict__, "generate_input": {"novel_id": novel_id, "min_word_count_per_episode": min_word_count_per_episode, "max_word_count_per_episode": max_word_count_per_episode}},
    #                 ),
    #                 model_name=self.cfg.model_name,
    #                 max_retries=self.cfg.retry_cnt
    #             )

    #             parsed = self._parse_script_response(script)
    #             result = {
    #                 "season_id": select_range.season_id,
    #                 "episode_id": episode_id,
    #                 **parsed,
    #             }
    #             results.append(result)

    #     await self.upload_episode_scripts(story_info, results)


    def _post_process_demo_scripts(self, response, season_id, chapter_from, chapter_to):
        """对demo剧本进行后处理"""
        json_data = json_repair.loads(response)

        epi_scripts = []
        epi_outlines = []

        for item in json_data.get("scripts", []):
            episode_idx = int(item["episode_number"])
            assert episode_idx in [1, 2, 3], "Only first three episodes are supported."

            script_content = item.get("content", "").strip()

            try:
                min_word, max_word = int(item["min_word"]), int(item["max_word"])
                if min_word <= max_word and self.cfg.enable_script_fix:
                    script_content = self.wc_validator.apply_fix(
                        script_content, min_word, max_word, min_ratio=0.6, label=f"demo ep{episode_idx}"
                    )
            except (KeyError, TypeError, ValueError) as e:
                logger.warning_context(self.ctx, f"[demo ep{episode_idx}] min_word/max_word 解析失败，跳过字数校验: {e}")

            epi_scripts.append(EpisodeScript(
                season_id=season_id,
                episode_id=episode_idx - 1,
                script_content=script_content,
                word_tier=item.get("word_tier", ""),
                tier_reason=item.get("tier_reason", ""),
            ))

        for item in json_data.get("outlines", []):
            episode_idx = int(item["episode_number"])
            assert episode_idx in [1, 2, 3], "Only first three episodes are supported."
            epi_outlines.append(EpisodeOutline(
                season_id=season_id,
                act_id=0,
                episode_id=episode_idx - 1,
                chapter_from=chapter_from,
                chapter_to=chapter_to,
                content=EpisodeOutline.to_text(episode_idx - 1, item.get("chapter_range", []), item.get("synopsis", ""), item.get("events", []), item.get("ending_hook", "")),
                episode_title=item.get("episode_title", ""),
                events=item.get("events", []),
                ending_hook=item.get("ending_hook", ""),
            ))

        return {"scripts": epi_scripts, "outlines": epi_outlines}

    async def generate_demo_scripts(
        self,
        story_info,
        generate_input,
        season_id,
        world_building,
        role_info,
        chapter_from,
        chapter_to,
        season_proposal,
        suggestion="暂无"
    ):
        try:
            chapter_content = await self.data_loader.load_contents_text(
                generate_input.novel_id, chapter_from, chapter_to, ctype='summary'
            )
            chapter_content = chapter_content[:model_cfg.content_max_word_cnt]

            scene_range, word_range = self._get_scene_and_word_range(story_info.plot_type, generate_input.min_word_count_per_episode, generate_input.max_word_count_per_episode)

            variables = {
                "NovelText": chapter_content,
                "RoleInfo": role_info,
                "WorldBuilding": world_building,
                "SeasonProposal": season_proposal.to_proposal_text(),
                "SceneRange": scene_range,
                "WordRange": word_range,
                "Suggestion": suggestion
            }

            result = query_llm(
                ctx=self.ctx,
                task=Task(
                    stage="write",
                    prompt_name="demo_script",
                    variables=variables,
                    domain=story_info.plot_type,
                    post_process_func=self._post_process_demo_scripts,
                    post_process_kwargs={"season_id": season_id, "chapter_from": chapter_from, "chapter_to": chapter_to},
                    debug_info=f"generate first three episodes script",
                    extra={"story_info": story_info.__dict__, "generate_input": generate_input.__dict__},
                ),
                model_name=self.cfg.model_name,
                max_retries=self.cfg.retry_cnt,
            )

            scripts, outlines = result["scripts"], result["outlines"]

            for episode_id, script in enumerate(scripts):
                script_url = await self.data_loader.upload(
                    script,
                    story_info=story_info,
                    season_id=season_id,
                    episode_id=episode_id
                )
                logger.info_context(self.ctx, f"First three episodes script uploaded to COS: {script_url}")
            for episode_id, outline in enumerate(outlines):
                outline_url = await self.data_loader.upload(
                    outline,
                    story_info=story_info,
                    season_id=season_id,
                    episode_id=episode_id
                )


            # 用已生成的 1-3 集剧本从零构建 DNMG，供后续 sequential 生成使用
            if self.cfg.use_memory_graph and self.memory_state_manager is not None:
                episode_script_pairs = [(s.episode_id, s.script_content) for s in scripts]
                await self.memory_state_manager.rebuild_from_scripts(
                    story_info=story_info,
                    season_id=season_id,
                    episode_scripts=episode_script_pairs,
                )
                logger.info_context(self.ctx, f"DNMG rebuilt from demo scripts for SE{season_id+1}")

        except Exception as e:
            logger.error_context(self.ctx, f"Error generating first three episodes script: {e}")
            raise

        return scripts, outlines
