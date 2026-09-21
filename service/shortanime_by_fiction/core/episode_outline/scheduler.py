
import re
import json_repair
import math
from trpc.log import logger
from service.shortanime_by_fiction.configs import FictionConfig, model_cfg
from service.shortanime_by_fiction.data_models.drama import EpisodeOutline
from service.shortanime_by_fiction.data_manager import DataManager
from service.shortanime_by_fiction.core.script.writer import ScriptWriter
from service.shortanime_by_fiction.core.analyze_fiction import FictionAnalyzer
# from service.shortanime_by_fiction.narrative.cpg.proposal_adapter import ProposalAdapter
# from service.shortanime_by_fiction.narrative.cpg.scheduler import PacingScheduler

from service.shortanime_by_fiction.llms import (
    query_llm,
    Task
)

class ProgressScheduler:

    def __init__(self, ctx):
        self.ctx = ctx
        self.data_manager = DataManager(ctx)
        self.cfg = FictionConfig()
        self.writer = ScriptWriter(ctx)

    async def upload_episode_outlines(
        self, story_info, episode_outlines
    ):
        for epi_outline in episode_outlines:
            await self.data_manager.upload(
                epi_outline,
                story_info=story_info,
                season_id=epi_outline.season_id,
                episode_id=epi_outline.episode_id,
            )

    def _get_ch_range_contents(self, novel_id, chapter_range):

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
            except:
                continue
        ch_ranges = sorted(ch_ranges)

        if not ch_ranges:
            return 0, 0, []
        start, end = ch_ranges[0], ch_ranges[-1]
        return start, end, ch_ranges

    def _process_episode_outline(self, response, novel_id, season_id, start_ep, end_ep):
        start_ep, end_ep = start_ep + 1, end_ep + 1
        json_data = json_repair.loads(response)

        episode_outlines = []
        completed_episodes = []
        for outline in json_data:
            chapter_range = outline["chapter_range"]
            start, end, ch_ranges = self._get_ch_range_contents(
                novel_id, chapter_range
            )
            episode_id = outline["episode_number"]-1
            completed_episodes.append(outline["episode_number"])
            episode_outline = {
                "season_id": season_id,
                "episode_id": episode_id,
                "ch_ranges": ch_ranges,
                "chapter_from": start,
                "chapter_to": end,
                "chapter_range": chapter_range,
            }
            episode_outline = EpisodeOutline.from_dict(episode_outline)
            episode_outline.content = EpisodeOutline.to_text(episode_id, chapter_range, outline["synopsis"], outline["events"], outline["ending_hook"])
            episode_outlines.append(episode_outline)

        # 检查是否所有章节都被覆盖
        if completed_episodes != list(range(start_ep, end_ep + 1)):
            raise ValueError(f"Invalid episode range, expected episode number {start_ep}~{end_ep}, got {completed_episodes}")

        return episode_outlines

    async def _generate_episode_outlines(
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
        prev_summary,
        episode_from,
        episode_to,
        suggestion="暂无",
        ctype="summary"
    ):
        try:
            chapter_content = await self.data_manager.load_contents_adaptive(
                generate_input.novel_id, chapter_from, chapter_to, model_cfg.content_max_word_cnt
            )
        except Exception as e:
            logger.error_context(self.ctx, f"Error loading chapter content for season {season_id}: {e}")
            raise

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
            "StartEp": episode_from + 1,
            "EndEp": episode_to + 1,
            "Suggestion": suggestion
        }

        task = Task(
            stage="plan",
            prompt_name="episode_outline",
            domain=story_info.plot_type,
            variables=variables,
            debug_info="generate episode outline",
            post_process_func=self._process_episode_outline,
            post_process_kwargs={
                "novel_id": generate_input.novel_id,
                "season_id": season_id,
                "start_ep": episode_from,
                "end_ep": episode_to
            },
            extra={"story_info": story_info.__dict__, "generate_input": generate_input.__dict__},
        )

        try:
            episode_outlines = query_llm(
                ctx=self.ctx,
                task=task,
                model_name=self.cfg.model_name,
                max_retries=self.cfg.retry_cnt,
            )
        except Exception as e:
            logger.error_context(self.ctx, f"Error generating episode outlines with LLM for season {season_id}: {e}")
            raise

        return episode_outlines

    async def _generate_episode_outlines_block(
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
        prev_episode_outlines,
        episode_from,
        episode_to,
        suggestion="暂无",
        ctype="summary",
        max_retries=2,
        eval_threshold=2.5,
        with_eval=False
    ):
        prev_summary = "\n\n".join([outline.sysnopsis_to_text() for outline in prev_episode_outlines])
        episode_outlines = await self._generate_episode_outlines(
            story_info=story_info,
            generate_input=generate_input,
            season_id=season_id,
            world_building=world_building,
            role_info=role_info,
            chapter_from=chapter_from,
            chapter_to=chapter_to,
            season_proposal=season_proposal,
            story_outline=story_outline,
            prev_summary=prev_summary,
            episode_from=episode_from,
            episode_to=episode_to,
            suggestion=suggestion,
            ctype=ctype
        )

        return episode_outlines

    async def generate_episode_outlines(
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
        suggestion='暂无',
        max_retries=2,
        eval_threshold=2.5,
        with_eval=False
    ):
        # # ── CPG: 加载/构建因果情节图 G₀ → G* → 分集调度 ─────────────────
        # cpg_assignment = {}
        # if self.cfg.episode_outline_method == "cpg":
        #     try:
        #         analyzer = FictionAnalyzer(self.ctx)
        #         g0 = await analyzer.get_cpg(
        #             story_info=story_info,
        #             novel_id=generate_input.novel_id,
        #             chapter_from=chapter_from,
        #             chapter_to=chapter_to,
        #         )
        #         # Step 1：将结构化删除方案转为 Layer 1 delete_plot ops，经 CausalGraphAdapter 两层执行
        #         g_star = g0.deepcopy()
        #         if season_proposal and season_proposal.plot_planning:
        #             delete_plot_ops = [
        #                 {"op": "delete_plot", "plot_id": f"P{p.plot_id:02d}", "reason": f"改编方案删除情节: {p.title}"}
        #                 for p in season_proposal.plot_planning
        #                 if getattr(p, "status", "") == "删除"
        #             ]
        #             if delete_plot_ops:
        #                 from service.shortanime_by_fiction.narrative.cpg.adapter import CausalGraphAdapter
        #                 adapter = CausalGraphAdapter(self.ctx)
        #                 g_star = adapter.execute(g_star, delete_plot_ops, domain=story_info.plot_type)
        #                 # 同步移除 plot 层记录
        #                 deleted_plot_ids = {op["plot_id"] for op in delete_plot_ops}
        #                 g_star.plots = [p for p in g_star.plots if p.plot_id not in deleted_plot_ids]


        #         pacing = PacingScheduler(minutes_per_episode=15.0)
        #         cpg_assignment = pacing.schedule(g_star, episode_count=int(generate_input.episode_nums))
        #         logger.debug_context(self.ctx, f"CPG season {season_id}: {g_star.summary()}, assignment episodes={len(cpg_assignment)}")
        #     except Exception as e:
        #         logger.warning_context(self.ctx, f"CPG build failed for season {season_id}, skipping: {e}")
        #         cpg_assignment = {}
        # # ── CPG end ──────────────────────────────────────────────────────

        if self.cfg.episode_outline_method == "cpg" and cpg_assignment:
            from service.shortanime_by_fiction.core.episode_outline.cpg_expander import CPGOutlineExpander
            expander = CPGOutlineExpander(self.ctx)
            episode_outlines = await expander.expand(
                cpg_assignment=cpg_assignment,
                story_info=story_info,
                generate_input=generate_input,
                season_id=season_id,
                season_proposal=season_proposal,
                world_building=world_building,
                role_info=role_info,
                story_outline=story_outline,
                suggestion=suggestion,
            )
        else:
            episode_outlines, prev_episode_outlines = [], []
            batch_size = model_cfg.episode_batch_size
            total_episodes = int(generate_input.episode_nums)
            num_batches = math.ceil(total_episodes / batch_size)

            for i in range(num_batches):
                episode_from = i * batch_size
                episode_to = min((i + 1) * batch_size - 1, total_episodes - 1)

                outlines = await self._generate_episode_outlines_block(
                    story_info=story_info,
                    generate_input=generate_input,
                    season_id=season_id,
                    world_building=world_building,
                    role_info=role_info,
                    chapter_from=chapter_from,
                    chapter_to=chapter_to,
                    season_proposal=season_proposal,
                    story_outline=story_outline,
                    prev_episode_outlines=prev_episode_outlines,
                    episode_from=episode_from,
                    episode_to=episode_to,
                    suggestion=suggestion,
                    max_retries=max_retries,
                    eval_threshold=eval_threshold,
                    with_eval=with_eval
                )

                episode_outlines.extend(outlines)
                prev_episode_outlines = episode_outlines

        if len(episode_outlines) != generate_input.episode_nums:
            logger.warning_context(self.ctx, f"Episode outlines length mismatch: {len(episode_outlines)} != {generate_input.episode_nums}")

        # await self.upload_episode_outlines(story_info, episode_outlines)
        return episode_outlines

    def _process_single_outline(self, response, episode_outline):
        json_data = json_repair.loads(response)
        episode_outline.content = json_data['content']
        return episode_outline

    async def _get_chapter_range(self, outline: str, story_info, season_id, episode_id):
        match = re.search(r'【章节范围】[：:]\s*\[([^\]]+)\]', outline)
        if match:
            nums = []
            for x in match.group(1).split(','):
                s = x.strip().strip("'\"")
                if s.isdigit():
                    nums.append(int(s))
                elif '-' in s:
                    parts = s.split('-')
                    if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
                        nums.extend([int(parts[0].strip()), int(parts[1].strip())])
            if nums:
                chapter_from, chapter_to = min(nums), max(nums)
                logger.info_context(self.ctx, f"[大纲章节号抽取] from outline text: chapter_from={chapter_from}, chapter_to={chapter_to}")
                return chapter_from, chapter_to
        # fallback: load from COS
        logger.warning_context(self.ctx, f"[大纲章节号抽取] no 【章节范围】 matched, fallback to COS for season={season_id} episode={episode_id}")
        episode_outline = await self.data_manager.load_episode_outline(story_info, season_id, episode_id)
        if episode_outline is None:
            raise ValueError(f"第{episode_id+1}集大纲获取章节号失败")
        return episode_outline.chapter_from, episode_outline.chapter_to

    async def regenerate_episode_outline_single(
        self,
        story_info,
        generate_input,
        outline,
        season_id,
        episode_id,
        suggestion
    ):
        chapter_from, chapter_to = await self._get_chapter_range(outline, story_info, season_id, episode_id)
        chapter_content = await self.data_manager.load_contents_text(
            generate_input.novel_id, chapter_from, chapter_to
        )
        variables = {
            "Outline": outline,
            "Suggestion": suggestion,
            "NovelText": chapter_content
        }

        try:
            episode_outline = EpisodeOutline(
                season_id=season_id,
                episode_id=episode_id,
                content=outline,
            )
            refine_outline = query_llm(
                ctx=self.ctx,
                task=Task(
                    stage="plan",
                    prompt_name="refine_ep_outline",
                    variables=variables,
                    domain=story_info.plot_type,
                    debug_info="refine episode outline",
                    post_process_func=self._process_single_outline,
                    post_process_kwargs={"episode_outline": episode_outline},
                    extra={"story_info": story_info.__dict__, "generate_input": generate_input.__dict__},
                ),
                model_name=self.cfg.model_name,
                max_retries=self.cfg.retry_cnt,
            )
            episode_outline_url = await self.data_manager.upload(
                refine_outline, story_info=story_info, season_id=season_id, episode_id=episode_id
            )
            logger.debug_context(self.ctx, f"Episode outline uploaded to COS: {episode_outline_url}")

            return refine_outline

        except Exception as e:
            logger.error_context(self.ctx, f"Error: {e}")
            raise
