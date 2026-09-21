"""
core/episode_outline/cpg_expander.py

CPGOutlineExpander — 将 PacingScheduler 输出的 cpg_assignment 展开为 EpisodeOutline 列表。

流程：
  cpg_assignment: {episode_id → [CausalEvent, ...]}
    ↓
  对每一集，把 CausalEvent 列表格式化为骨架文本
    ↓
  LLM 做 fabula→sjuzhet 叙事展开（synopsis / events / ending_hook）
    ↓
  组装 EpisodeOutline（ch_ranges 从 source_chapters 聚合）
"""

from __future__ import annotations

import json_repair
from typing import Dict, List

from trpc.log import logger

from service.shortanime_by_fiction.configs import FictionConfig, model_cfg
from service.shortanime_by_fiction.data_models.cpg import CausalEvent, CausalPlotGraph
from service.shortanime_by_fiction.data_models.drama import EpisodeOutline
from service.shortanime_by_fiction.llms import query_llm, Task


class CPGOutlineExpander:
    """将 cpg_assignment 展开为 EpisodeOutline 列表。

    Usage::

        expander = CPGOutlineExpander(ctx)
        episode_outlines = await expander.expand(
            cpg_assignment=cpg_assignment,
            story_info=story_info,
            generate_input=generate_input,
            season_id=season_id,
            season_proposal=season_proposal,
            world_building=world_building,
            role_info=role_info,
            story_outline=story_outline,
            prev_summary=prev_summary,
        )
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.cfg = FictionConfig()

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    async def expand(
        self,
        cpg_assignment: Dict[int, List[CausalEvent]],
        story_info,
        generate_input,
        season_id: int,
        season_proposal,
        world_building: str,
        role_info: str,
        story_outline: str,
        prev_summary: str = "暂无",
        suggestion: str = "暂无",
    ) -> List[EpisodeOutline]:
        """将 cpg_assignment 中所有集逐一展开为 EpisodeOutline。

        Returns:
            按 episode_id 升序排列的 EpisodeOutline 列表
        """
        episode_outlines = []

        sorted_episodes = sorted(cpg_assignment.items())  # 按 episode_id 升序
        total_episodes = int(generate_input.episode_nums)

        for ep_id, events in sorted_episodes:
            if not events:
                logger.warning_context(
                    self.ctx, f"CPG expander: episode {ep_id} has no events, skipping"
                )
                continue

            try:
                outline = await self._expand_episode(
                    ep_id=ep_id,
                    events=events,
                    story_info=story_info,
                    generate_input=generate_input,
                    season_id=season_id,
                    season_proposal=season_proposal,
                    world_building=world_building,
                    role_info=role_info,
                    story_outline=story_outline,
                    prev_summary=prev_summary,
                    total_episodes=total_episodes,
                    suggestion=suggestion,
                )
                episode_outlines.append(outline)
                # 滚动更新 prev_summary
                prev_summary = "\n\n".join(
                    [o.sysnopsis_to_text() for o in episode_outlines]
                )
            except Exception as e:
                logger.error_context(
                    self.ctx,
                    f"CPG expander: failed to expand episode {ep_id}: {e}",
                )
                raise

        return episode_outlines

    # ------------------------------------------------------------------
    # 私有方法
    # ------------------------------------------------------------------

    def _format_skeleton(self, events: List[CausalEvent]) -> str:
        """把 CausalEvent 列表格式化为骨架文本，供 LLM 展开。"""
        lines = []
        for e in events:
            chapters = (
                f"{e.source_chapters[0]}~{e.source_chapters[-1]}章"
                if len(e.source_chapters) > 1
                else (f"{e.source_chapters[0]}章" if e.source_chapters else "新增")
            )
            lines.append(
                f"[{e.id}] {e.title}"
                f"（beat_type={e.beat_type}，drama_weight={e.drama_weight:.2f}，章节={chapters}）"
                f"\n  涉及角色：{', '.join(e.characters) if e.characters else '暂无'}"
                f"\n  涉及场景：{', '.join(e.locations) if e.locations else '暂无'}"
                f"\n  事件简述：{e.synopsis}"
            )
        return "\n\n".join(lines)

    def _collect_ch_ranges(self, events: List[CausalEvent]):
        """从事件列表聚合 ch_ranges / chapter_from / chapter_to / chapter_range。"""
        all_chapters = sorted(
            {ch for e in events for ch in e.source_chapters if ch > 0}
        )
        if not all_chapters:
            return [], 0, 0, ["0"]

        chapter_from = all_chapters[0]
        chapter_to = all_chapters[-1]

        # 压缩为连续段，如 [1,2,3,5] → ["1-3","5"]
        segments = []
        start = all_chapters[0]
        prev = all_chapters[0]
        for ch in all_chapters[1:]:
            if ch == prev + 1:
                prev = ch
            else:
                segments.append(f"{start}-{prev}" if start != prev else str(start))
                start = prev = ch
        segments.append(f"{start}-{prev}" if start != prev else str(start))

        return all_chapters, chapter_from, chapter_to, segments

    def _process_episode(self, response, ep_id: int, events: List[CausalEvent], season_id: int):
        """post_process_func：解析 LLM 返回的单集展开 JSON，组装 EpisodeOutline。"""
        json_data = json_repair.loads(response)

        synopsis = json_data.get("synopsis", "")
        ev_list = json_data.get("events", [])
        ending_hook = json_data.get("ending_hook", "")

        ch_ranges, chapter_from, chapter_to, chapter_range = self._collect_ch_ranges(events)

        episode_outline = EpisodeOutline.from_dict({
            "season_id": season_id,
            "episode_id": ep_id,
            "ch_ranges": ch_ranges,
            "chapter_from": chapter_from,
            "chapter_to": chapter_to,
            "chapter_range": chapter_range,
        })
        episode_outline.content = EpisodeOutline.to_text(
            ep_id, chapter_range, synopsis, ev_list, ending_hook
        )
        return episode_outline

    async def _expand_episode(
        self,
        ep_id: int,
        events: List[CausalEvent],
        story_info,
        generate_input,
        season_id: int,
        season_proposal,
        world_building: str,
        role_info: str,
        story_outline: str,
        prev_summary: str,
        total_episodes: int,
        suggestion: str,
    ) -> EpisodeOutline:
        """调用 LLM 将单集骨架展开为 EpisodeOutline。"""
        skeleton = self._format_skeleton(events)

        variables = {
            "EpisodeSkeleton": skeleton,
            "EpisodeIndex": ep_id + 1,
            "TotalEpisodes": total_episodes,
            "SeasonIndex": season_id + 1,
            "SeasonCnt": generate_input.season_nums,
            "WorldBuilding": world_building,
            "RoleInfos": role_info,
            "StoryOutline": story_outline,
            "Proposal": season_proposal.to_proposal_text(),
            "PrevSummary": prev_summary,
            "Suggestion": suggestion,
        }

        task = Task(
            stage="adaptation",
            prompt_name="cpg_expand_episode",
            domain="COMMON",
            variables=variables,
            debug_info=f"cpg expand episode {ep_id + 1}",
            post_process_func=self._process_episode,
            post_process_kwargs={
                "ep_id": ep_id,
                "events": events,
                "season_id": season_id,
            },
            extra={
                "story_info": story_info.__dict__,
                "generate_input": generate_input.__dict__,
            },
        )

        episode_outline = query_llm(
            ctx=self.ctx,
            task=task,
            model_name=self.cfg.model_name,
            max_retries=self.cfg.retry_cnt,
        )
        return episode_outline
