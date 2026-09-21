import collections
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from trpc.log import logger
from trpc.context import Context


from service.drama_by_fiction.data_model import FictionConfig, StoryArcs
from service.drama_by_fiction.core.data_manager import DataManager
from service.drama_by_fiction.utils import (
    _query_llm_with_postprocess,
    post_validate_json_schema,
    Task,
)
from service.drama_by_fiction.utils.llm_validate import (
    STORY_ARC_SCHEMA,
    STORY_EVENT_SCHEMA,
)


class FictionAnalyzer:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.data_loader = DataManager(ctx)
        self.cfg = FictionConfig()

    def _gen_story_arc_proc(self, task):
        """单个故事弧生成任务处理"""

        idx = task["idx"]
        llm_task = Task(
            stage="analyze",
            prompt_name="story_arc",
            variables={
                "ChapterAbstract": task["content"],
                "ChapterRange": self.cfg.arc_range,
            },
            debug_info=f"generate block {idx} arcs",
            post_process_func=post_validate_json_schema,
            post_process_kwargs={"schema": STORY_ARC_SCHEMA},
        )
        story_arcs = _query_llm_with_postprocess(
            ctx=self.ctx,
            task=llm_task,
            model_name=self.cfg.model_name,
            max_retries=self.cfg.retry_cnt,
        )
        return story_arcs if story_arcs else []

    def _fuse_story_arcs(self, map_results, total_chapters: int = None):
        """故事弧校准融合
        由于block块截断、部分重叠，生成的故事弧可能存在 章节范围重复、不一致、block的开始或结束区域的故事弧不完整的问题
        需要进行校准融合，融合成一系列连续、不重叠的最终故事弧。
        核心思想：
         - 将所有block块中的故事弧的开始和结束作为候选的分界点，
         - 对所有候选点计算一个“置信度分数”；
         - 从头到尾，依次选择最优的边界点来切分故事弧。

        1. 数据预处理：遍历所有block处理结果，提取所有故事弧，每个故事弧包含：
        start：起始章节； end：结束章节； block范围 [a, b]

        2. 边界点置信度评分
        - 确定分割点： 两个连续章节之间的分割线。用它前面那一章的章节号来代表它。
          对于开始位置N，结束位置M的故事弧，其分割点分别对应 N - 1 和 M；

        - 置信度分数： 越靠近边缘的分割点受截断的影响越大，位于block中心的故事弧更可能是自然形成的，
          因此在基础分上增加一个权重分，越靠近block边界分越小，越靠近中心分越大。

        3. 贪心算法构建故事弧：
          1）从第1章开始作为当前故事弧起点
          2）在 [arc_start + min_arc_len - 1, arc_start + max_arc_len - 1] 窗口内搜索候选边界点
          3）选择置信度最高的边界点作为当前故事弧的结束点
          4）将下一章设置为新的起点，重复过程

          特殊边界处理：
          * 接近小说结尾时，避免产生过短的故事弧
          * 当搜索窗口内无候选分界点时，使用回退长度（fallback_len）强行分割

        Args:
            fiction_id: 小说ID
            map_results: 各区块的故事弧结果
            total_chapters: 总章节数

        Returns:
            list: 融合后的故事弧列表 [[start, end], ...]
        """
        # =========================================================================
        # 第一步：数据收集与预处理
        # =========================================================================
        raw_arcs = []
        for idx, result in enumerate(map_results):
            start_idx = (
                idx * (self.cfg.chapter_block_cnt - self.cfg.overlap_chapter_cnt) + 1
            )
            end_idx = start_idx + self.cfg.chapter_block_cnt
            for arc in result:
                item = {
                    "start": arc["start_chapter_num"],
                    "end": arc["end_chapter_num"],
                    "block": [start_idx, end_idx],
                }
                raw_arcs.append(item)

        if not raw_arcs:
            logger.warning_context(self.ctx, "输入的原始故事弧列表为空。")
            return []

        total_chapters = total_chapters if total_chapters else raw_arcs[-1]["end"]

        # =========================================================================
        # 第二步：边界点置信度评分
        # =========================================================================
        boundary_scores = collections.defaultdict(float)

        def calculate_score(point, start, end):
            # 中心度加权：越靠近block中心，权重越高
            position_ratio = (point - start) / (end - start)
            centrality_weight = 1.0 - (2 * position_ratio - 1) ** 2
            return 1.0 + centrality_weight

        for arc in raw_arcs:
            arc_start, arc_end = arc["start"], arc["end"]
            block_start, block_end = arc["block"]

            block_span = block_end - block_start
            if block_span <= 0:
                continue

            # 结束点
            boundary_scores[arc_end] += calculate_score(arc_end, block_start, block_end)
            # 开始点
            if arc_start > 1:
                boundary_scores[arc_start - 1] += calculate_score(
                    arc_start - 1, block_start, block_end
                )

        # =========================================================================
        # 第三步：贪心算法构建最终故事弧
        # =========================================================================
        final_arcs = []
        current_chapter = 1

        while current_chapter <= total_chapters:
            arc_start = current_chapter

            # 确定寻找最佳结束点的搜索窗口
            search_window_start = arc_start + self.cfg.min_arc_len - 1
            search_window_end = arc_start + self.cfg.max_arc_len - 1

            # 筛选出落在搜索窗口内的所有候选边界点
            candidates = {
                point: score
                for point, score in boundary_scores.items()
                if search_window_start <= point <= search_window_end
            }

            # 如果当前章节接近小说结尾
            if total_chapters - arc_start < self.cfg.min_arc_len:
                final_arcs.append([arc_start, total_chapters])
                break

            best_end = 0
            if candidates:
                best_end = max(candidates, key=candidates.get)
            else:
                logger.warning_context(
                    self.ctx,
                    f"在章节 {arc_start} 附近 [{search_window_start}, {search_window_end}] "
                    f"未找到候选边界点。将使用回退长度 {self.cfg.fallback_len}。",
                )
                best_end = arc_start + self.cfg.fallback_len - 1

            arc_end = min(best_end, total_chapters)

            # 避免产生过短的最后一个故事弧
            if total_chapters - arc_end < self.cfg.min_arc_len / 2:
                arc_end = total_chapters

            final_arcs.append([arc_start, arc_end])
            current_chapter = arc_end + 1
        return final_arcs

    async def generate_story_arcs(self, novel_id, start_idx=1, end_idx=None):
        chapter_summaries = await self.data_loader.load_chapter_summaries(
            novel_id, start_idx, end_idx
        )
        if not chapter_summaries:
            raise RuntimeError(f"no chapter summaries found for {novel_id}")

        chapter_blocks = [
            chapter_summaries[idx : idx + self.cfg.chapter_block_cnt]
            for idx in range(
                0,
                len(chapter_summaries),
                self.cfg.chapter_block_cnt - self.cfg.overlap_chapter_cnt,
            )
        ]
        chapter_blocks = ["\n\n".join(block) for block in chapter_blocks]

        map_tasks = []
        for idx in range(len(chapter_blocks)):
            task = {
                "idx": idx,
                "novel_id": novel_id,
                "content": chapter_blocks[idx],
            }
            map_tasks.append(task)

        map_results = []
        with ThreadPoolExecutor(max_workers=self.cfg.max_workers) as executor:
            results = executor.map(self._gen_story_arc_proc, map_tasks)
            for x in tqdm(results, total=len(map_tasks)):
                map_results.append(x)

        return map_results

    def _extract_events_proc(self, task):
        """单个事件提取任务处理"""
        try:
            story_id = task["story_id"]
            llm_task = Task(
                stage="analyze",
                prompt_name="extract_event",
                variables={
                    "ChapterAbstract": task["content"],
                },
                debug_info=f"generate block {story_id} events",
                post_process_func=post_validate_json_schema,
                post_process_kwargs={"schema": STORY_EVENT_SCHEMA},
            )
            events_data = _query_llm_with_postprocess(
                ctx=self.ctx,
                task=llm_task,
                model_name=self.cfg.model_name,
                max_retries=self.cfg.retry_cnt,
            )
            if not events_data:
                return None
            result = {
                "story_id": task["story_id"],
                "start_idx": task["start_idx"],
                "end_idx": task["end_idx"],
                "arc_title": events_data["arc_title"],
                "arc_summary": events_data["arc_summary"],
                "events": events_data["events"],
            }
            return result

        except Exception as e:
            logger.error(f"extract events proc error: {e}")
        return None

    def _postprocess_story_characters(self, events_data, top_k=8):
        """后处理：1.提取故事关联的主要角色; 2.更新故事和事件中角色顺序，按全局频次排序

        Args:
            events_data: 事件数据列表
            top_k: 取出现频次最高的前k个角色

        Returns:
            list: 处理后的事件数据
        """
        character_freqs = collections.defaultdict(int)
        for story_events in events_data:
            for event in story_events.get("events", []):
                for character in event.get("characters", []):
                    character_freqs[character] += 1

        # 按频次排序，取前top_k个
        sorted_characters = sorted(
            character_freqs.items(), key=lambda x: x[1], reverse=True
        )
        top_characters = [char for char, _ in sorted_characters[:top_k]]

        # 获取 story 关联的 characters
        for story_events in events_data:
            story_characters = set()
            for event in story_events.get("events", []):
                event_characters = event.get("characters", [])

                # 按照全局频次对事件中的角色进行排序
                sorted_characters = sorted(
                    event_characters,
                    key=lambda char: character_freqs.get(char, 0),
                    reverse=True,
                )
                event["characters"] = sorted_characters

                # story 关联的 characters 需要是 top_characters 中的
                for character in sorted_characters:
                    if character in top_characters:
                        story_characters.add(character)

            story_characters = list(story_characters)
            # 按照全局频次对故事弧的角色进行排序
            story_characters = sorted(
                story_characters, key=lambda x: character_freqs[x], reverse=True
            )
            story_events["characters"] = story_characters

        return events_data

    async def extract_story_events(self, novel_id, story_arcs: StoryArcs):
        tasks = []
        for idx, (arc_start, arc_end) in enumerate(story_arcs):
            summaries = await self.data_loader.load_chapter_summaries(
                novel_id, arc_start, arc_end
            )
            summaries = "\n\n".join(summaries)

            task = {
                "story_id": idx + 1,
                "start_idx": arc_start,
                "end_idx": arc_end,
                "content": summaries,
            }
            tasks.append(task)

        with ThreadPoolExecutor(max_workers=self.cfg.max_workers) as executor:
            map_results = executor.map(self._extract_events_proc, tasks)
            results = list(map_results)
        story_events = [x for x in results if x]

        return story_events

    async def generate_story_events(self, novel_id):
        try:
            # 先加载辅助创作故事弧事件
            event_data = await self.data_loader.load_story_arcs(novel_id)
            if event_data:
                logger.info(f"load story events from story cache for {novel_id}")
                return event_data

            # 再加载小说故事弧事件
            event_data = await self.data_loader.load_novel_arcs(novel_id)
            if event_data:
                logger.info(f"load story events from novel cache for {novel_id}")
                story_arcs = StoryArcs(data={"arcs": event_data})
                await self.data_loader.upload(story_arcs, novel_id=novel_id)
                return event_data

            # 没有找到，开始生成
            logger.info(f"generate story events for {novel_id}")

            # 1. 生成故事弧
            story_arcs_data = await self.generate_story_arcs(novel_id)

            # 2. 融合故事弧
            final_arcs = self._fuse_story_arcs(story_arcs_data)

            # 3. 提取事件
            event_data = await self.extract_story_events(novel_id, final_arcs)

            # 4. 后处理角色
            event_data = self._postprocess_story_characters(event_data)

            # 5.上传到COS
            story_arcs = StoryArcs(data={"arcs": event_data})
            await self.data_loader.upload(story_arcs, novel_id=novel_id)
            return event_data

        except Exception as e:
            logger.error(f"generate story events error: {e}")
            raise
