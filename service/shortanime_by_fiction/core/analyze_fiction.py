from concurrent.futures import ThreadPoolExecutor
import collections

from tqdm import tqdm
from trpc.log import logger

import re
import json_repair
from service.shortanime_by_fiction.data_models import FictionConfig, RoleInfo, StoryArcs, WorldBuilding
from service.shortanime_by_fiction.data_manager import DataManager
from service.shortanime_by_fiction.data_models.fiction import StoryInfo
from service.shortanime_by_fiction.llms import (
    Task,
    query_llm,
    validate_json_schema,
)
from service.shortanime_by_fiction.configs import model_cfg
# from service.shortanime_by_fiction.narrative.cpg import CausalGraphBuilder
# from service.shortanime_by_fiction.data_models.cpg import CausalPlotGraph


class FictionAnalyzer:
    def __init__(self, ctx):
        self.ctx = ctx
        self.data_loader = DataManager(ctx)
        self.cfg = FictionConfig()

    def _gen_story_arc_proc(self, task):
        story_arc_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_chapter_num": {"type": "integer"},
                    "end_chapter_num": {"type": "integer"},
                    "title": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["start_chapter_num", "end_chapter_num", "title", "content"]
            }
        }

        idx = task["idx"]
        response = query_llm(
            ctx=self.ctx,
            task=Task(
                stage="analyze",
                domain="COMMON",
                prompt_name="story_arc",
                variables={
                    "ChapterAbstract": task["content"],
                    "ChapterRange": self.cfg.arc_range,
                },
                debug_info=f"generate block {idx} arcs",
                validate_func=lambda x: validate_json_schema(x, story_arc_schema),
            ),
            model_name=self.cfg.model_name,
            max_retries=self.cfg.retry_cnt,
        )
        story_arcs = json_repair.loads(response) 
        return story_arcs

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
        with ThreadPoolExecutor(max_workers=model_cfg.max_workers) as executor:
            results = executor.map(self._gen_story_arc_proc, map_tasks)
            for x in tqdm(results, total=len(map_tasks)):
                map_results.append(x or [])
        return map_results

    def _fuse_story_arcs(self, map_results, total_chapters: int = None):
        """故事弧校准融合：边界点置信度评分 + 贪心切分。

        map_results: List[List[arc]]，外层=block，内层=该 block 生成的弧。
        返回: List[arc]（拍平、连续不重叠），每个 arc 含
              start_chapter_num/end_chapter_num/title/content。
        """
        block_step = self.cfg.chapter_block_cnt - self.cfg.overlap_chapter_cnt

        raw_arcs = []
        for idx, block_arcs in enumerate(map_results):
            block_start = idx * block_step + 1
            block_end = block_start + self.cfg.chapter_block_cnt
            for arc in block_arcs:
                raw_arcs.append({
                    "start": arc["start_chapter_num"],
                    "end": arc["end_chapter_num"],
                    "title": arc.get("title", ""),
                    "content": arc.get("content", ""),
                    "block": [block_start, block_end],
                })

        if not raw_arcs:
            logger.warning_context(self.ctx, "输入的原始故事弧列表为空。")
            return []

        total_chapters = total_chapters or max(a["end"] for a in raw_arcs)

        boundary_scores = collections.defaultdict(float)
        # 记录每个边界点上累计的弧元数据（title/content），融合时用得分最高的那个
        boundary_payload = collections.defaultdict(list)

        def calc_score(point, bs, be):
            ratio = (point - bs) / (be - bs)
            centrality = 1.0 - (2 * ratio - 1) ** 2
            return 1.0 + centrality

        for arc in raw_arcs:
            bs, be = arc["block"]
            if be - bs <= 0:
                continue
            s_end = calc_score(arc["end"], bs, be)
            boundary_scores[arc["end"]] += s_end
            boundary_payload[arc["end"]].append((s_end, arc))
            if arc["start"] > 1:
                s_start = calc_score(arc["start"] - 1, bs, be)
                boundary_scores[arc["start"] - 1] += s_start

        final_arcs = []
        cur = 1
        while cur <= total_chapters:
            arc_start = cur
            win_lo = arc_start + self.cfg.min_arc_len - 1
            win_hi = arc_start + self.cfg.max_arc_len - 1

            if total_chapters - arc_start < self.cfg.min_arc_len:
                arc_end = total_chapters
            else:
                candidates = {
                    p: s for p, s in boundary_scores.items() if win_lo <= p <= win_hi
                }
                if candidates:
                    arc_end = max(candidates, key=candidates.get)
                else:
                    logger.warning_context(
                        self.ctx,
                        f"章节 {arc_start} 附近 [{win_lo}, {win_hi}] 无候选边界点,"
                        f"使用回退长度 {self.cfg.fallback_len}。",
                    )
                    arc_end = arc_start + self.cfg.fallback_len - 1
                arc_end = min(arc_end, total_chapters)
                if total_chapters - arc_end < self.cfg.min_arc_len / 2:
                    arc_end = total_chapters

            # 取在该结束点上得分最高的原始 arc 作为 title/content 来源
            payloads = boundary_payload.get(arc_end, [])
            best_meta = max(payloads, key=lambda x: x[0])[1] if payloads else None

            final_arcs.append({
                "start_chapter_num": arc_start,
                "end_chapter_num": arc_end,
                "title": best_meta["title"] if best_meta else "",
                "content": best_meta["content"] if best_meta else "",
            })
            cur = arc_end + 1

        return final_arcs

    # ------------------------------------------------------------------
    # 事件抽取层(已实现,未接入 get_story_arcs;调用方需要时手动串)
    # ------------------------------------------------------------------
    def _extract_events_proc(self, task):
        """单个故事弧的事件抽取。返回 {start_idx, end_idx, arc_title, arc_summary, events}。"""
        story_event_schema = {
            "type": "object",
            "properties": {
                "arc_title": {"type": "string"},
                "arc_summary": {"type": "string"},
                "events": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "summary": {"type": "string"},
                            "characters": {
                                "type": "array", "items": {"type": "string"}
                            },
                        },
                        "required": ["title", "summary"],
                    },
                },
            },
            "required": ["arc_title", "arc_summary", "events"],
        }
        try:
            story_id = task["story_id"]
            response = query_llm(
                ctx=self.ctx,
                task=Task(
                    stage="analyze",
                    domain="COMMON",
                    prompt_name="extract_event",
                    variables={"ChapterAbstract": task["content"]},
                    debug_info=f"extract events for story {story_id}",
                    validate_func=lambda x: validate_json_schema(x, story_event_schema),
                ),
                model_name=self.cfg.model_name,
                max_retries=self.cfg.retry_cnt,
            )
            events_data = json_repair.loads(response)
            if not events_data:
                return None
            return {
                "story_id": story_id,
                "start_idx": task["start_idx"],
                "end_idx": task["end_idx"],
                "arc_title": events_data["arc_title"],
                "arc_summary": events_data["arc_summary"],
                "events": events_data["events"],
            }
        except Exception as e:
            logger.error(f"extract events proc error: {e}")
            return None

    async def extract_story_events(self, novel_id, fused_arcs):
        """对每个 fused arc 抽取事件。
        fused_arcs: List[arc dict],含 start_chapter_num/end_chapter_num。
        返回 List[{story_id, start_idx, end_idx, arc_title, arc_summary, events}]。
        """
        tasks = []
        for idx, arc in enumerate(fused_arcs):
            arc_start = arc["start_chapter_num"]
            arc_end = arc["end_chapter_num"]
            summaries = await self.data_loader.load_chapter_summaries(
                novel_id, arc_start, arc_end
            )
            tasks.append({
                "story_id": idx + 1,
                "start_idx": arc_start,
                "end_idx": arc_end,
                "content": "\n\n".join(summaries),
            })

        with ThreadPoolExecutor(max_workers=self.cfg.max_workers) as executor:
            results = list(executor.map(self._extract_events_proc, tasks))
        return [x for x in results if x]

    def _postprocess_story_characters(self, events_data, top_k=8):
        """按全局频次提取主要角色,并对 arc/event 内的 characters 排序。
        events_data: extract_story_events 的返回值;原地添加 'characters' 字段。
        """
        char_freqs = collections.defaultdict(int)
        for story in events_data:
            for event in story.get("events", []):
                for c in event.get("characters", []):
                    char_freqs[c] += 1

        top_chars = [
            c for c, _ in sorted(char_freqs.items(), key=lambda x: x[1], reverse=True)[:top_k]
        ]

        for story in events_data:
            story_chars = set()
            for event in story.get("events", []):
                event_chars = sorted(
                    event.get("characters", []),
                    key=lambda c: char_freqs.get(c, 0),
                    reverse=True,
                )
                event["characters"] = event_chars
                for c in event_chars:
                    if c in top_chars:
                        story_chars.add(c)
            story["characters"] = sorted(
                list(story_chars), key=lambda c: char_freqs[c], reverse=True
            )
        return events_data

    async def get_story_arcs(self, novel_id):
        story_arc = await self.data_loader.load_story_arcs(novel_id)
        if not story_arc:
            logger.info_context(
                self.ctx, f"Story arcs not found for novel_id: {novel_id}, start generating..."
            )
            map_results = await self.generate_story_arcs(novel_id)
            fused = self._fuse_story_arcs(map_results)

            story_arcs = StoryArcs(story_arcs=fused)
            arcs_url = await self.data_loader.upload(story_arcs, novel_id=novel_id)
            # 同步落本地 cache,下次直接走 L0
            try:
                from service.shortanime_by_fiction.data_manager.fiction_manager import CACHE_DIR
                story_arcs.save_to_file(f"{CACHE_DIR}/{novel_id}/{novel_id}_arc.json")
            except Exception as e:
                logger.warning_context(self.ctx, f"save local cache failed: {e}")
            logger.info_context(self.ctx, f"Generated story arcs uploaded to COS: {arcs_url}")
            return story_arcs
        return story_arc
    
    async def generate_role_info(
        self, 
        novel_id, 
        adaptation_proposal=None, 
        start_idx=1, 
        end_idx=None, 
        origin_role_info=None,
        suggestion=None,
        gtype="init_generate",
        ctype="summary"
    ):
        chapter_content = await self.data_loader.load_contents_text(
            novel_id, start_idx, end_idx, ctype
        )

        variables = {
            "ChapterAbstract": chapter_content,
            "AdaptationProposal": adaptation_proposal
        }

        if origin_role_info:
            variables["RoleInfo"] = origin_role_info
        if suggestion:
            variables["Suggestion"] = suggestion

        if gtype == "init_generate":
            prompt_name = "role_info"
        else:
            prompt_name = "refine_role_info"

        result = query_llm(
            ctx=self.ctx,
            task=Task(
                stage="analyze",
                prompt_name=prompt_name,
                variables=variables,
                domain="COMMON",
                debug_info=f"generate role info",
            ),
            model_name=self.cfg.model_name,
            max_retries=self.cfg.retry_cnt,
        )
        role_info = RoleInfo(content=result)
        return role_info

    # async def regenerate_role_info(self, novel_id, role_info, outline, suggestion):
    #     story_arcs = await self.data_loader.load_story_arcs(novel_id)
    #     params = {
    #         "RoleInfo": role_info,
    #         "Outline": outline,
    #         "Suggestion": suggestion,
    #         "ChapterAbstract": story_arcs,
    #     }
    #     role_info = _query_llm_with_validator(
    #         ctx=self.ctx,
    #         model_name=self.cfg.model_name,
    #         stage="analyze",
    #         prompt_name="refine_role_info",
    #         variables=params,
    #         debug_info=f"regenerate role info",
    #         max_retries=self.cfg.retry_cnt,
    #     )
    #     return role_info

    # ------------------------------------------------------------------
    # CPG 构建（因果情节图 G₀）
    # ------------------------------------------------------------------

    def _cpg_cos_path(self, novel_id: str, story_id: str) -> str:
        return f"drama_operation/short_anime/{novel_id}/{story_id}_cpg_g0.pkl"

    # async def generate_cpg(
    #     self,
    #     story_info: StoryInfo,
    #     novel_id: str,
    #     chapter_from: int = 1,
    #     chapter_to: int = None,
    # ) -> CausalPlotGraph:
    #     """从原著章节内容构建因果情节图 G₀。"""
    #     if chapter_to is None:
    #         chapter_to = await self.data_loader.get_max_chapter_idx(novel_id)

    #     builder = CausalGraphBuilder(self.ctx)
    #     g0 = await builder.build(story_info, novel_id, chapter_from, chapter_to)
    #     return g0

    # async def get_cpg(
    #     self,
    #     story_info: StoryInfo,
    #     novel_id: str,
    #     chapter_from: int = 1,
    #     chapter_to: int = None,
    # ) -> CausalPlotGraph:
    #     """构建因果情节图 G₀。"""
    #     return await self.generate_cpg(story_info, novel_id, chapter_from, chapter_to)

    async def generate_world_building(
        self, novel_id, adaptation_proposal=None, start_idx=1, end_idx=None, ctype="summary"
    ):
        chapter_content = await self.data_loader.load_contents_text(
            novel_id, start_idx, end_idx, ctype
        )

        variables = {
            "ChapterAbstract": chapter_content,
            "AdaptationProposal": adaptation_proposal
        }

        result = query_llm(
            ctx=self.ctx,
            task=Task(
                stage="analyze",
                prompt_name="world_building",
                variables=variables,
                domain="COMMON",
                debug_info=f"generate world building",
            ),
            model_name=self.cfg.model_name,
            max_retries=self.cfg.retry_cnt,
        )
        world_building = WorldBuilding(content=result)
        return world_building


class FictionRetriever:

    def __init__(self, ctx):
        self.ctx = ctx
        self.data_loader = DataManager(ctx)
        self.cfg = FictionConfig()
    
    async def run(self, novel_id, keywords, start_idx=1, end_idx=None, ctype="script"):
        fictions = await self.data_loader.load_chapter_contents(
            novel_id, start_idx, end_idx
        )
        results = []
        chapter_ids = []
        for chapter in fictions:
            for keyword in keywords:
                if keyword in chapter:
                    chapter_id = int(re.search(r'\[CHAPTER_ID (.*)\]', chapter).group(1))
                    if chapter_ids and chapter_ids[-1][-1] > chapter_id - 3:
                        chapter_ids[-1].append(chapter_id)
                    else:
                        chapter_ids.append([chapter_id])
                results.append(chapter)
        
        # 合并相邻的章节chapter_ids
        chapter_list = []
        for chapter_id in chapter_ids:
            if len(chapter_id) >= 4:
                chapter_list.extend(chapter_id)
                
        contents = await self.data_loader.load_contents_exact(novel_id, chapter_list, ctype="script")

        return contents
