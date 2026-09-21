import os

from trpc.log import logger
from cos_orm import BaseCosModel

from service.shortanime_by_fiction.data_models import FictionConfig
from service.shortanime_by_fiction.data_models import (
    NovelStructed,
    NovelArcs,
    StoryArcs,
    WorldBuilding,
    RoleInfo,
)
from service.shortanime_by_fiction.configs.model_config import model_cfg

CACHE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", ".cache", "shortanime_by_fiction"
)

COS = None


class FictionDataManager:
    def __init__(self, ctx):
        self.cos = COS
        self.ctx = ctx
        self.cfg = FictionConfig()

    def _clamp_chapter_range(self, chapter_from, chapter_to):
        """将章节范围钳位到 [MIN_CHAPTER_IDX, MAX_CHAPTER_IDX]"""
        if chapter_from is not None:
            chapter_from = max(chapter_from, model_cfg.min_chapter_idx)
        if chapter_to is not None:
            chapter_to = min(chapter_to, model_cfg.max_chapter_idx)
        return chapter_from, chapter_to

    async def load_story_arcs(self, novel_id):
        """3 级查找:本地 cache -> COS L1(StoryArcs) -> COS L2(ip_gpt NovelArcs)。
        命中后统一落本地 cache;L2 命中额外回写 L1。
        """
        local_path = f"{CACHE_DIR}/{novel_id}/{novel_id}_arc.json"

        # L0: 本地 cache
        if os.path.exists(local_path):
            try:
                arcs_data = StoryArcs.load_from_file(local_path)
                if arcs_data.story_arcs:
                    logger.info_context(
                        self.ctx, f"Loaded NovelArcs {novel_id} from local cache"
                    )
                    return arcs_data.story_arcs
            except Exception as e:
                logger.warning_context(self.ctx, f"local cache load failed: {e}")

        # L1: short_anime 自有 COS
        try:
            story_arcs = await StoryArcs.download(self.ctx, self.cos, novel_id=novel_id)
            if story_arcs and story_arcs.story_arcs:
                story_arcs.save_to_file(local_path)
                logger.info_context(
                    self.ctx, f"Loaded NovelArcs {novel_id} from COS"
                )
                return story_arcs.story_arcs
        except Exception as e:
            logger.info_context(self.ctx, f"L1 StoryArcs miss for {novel_id}: {e}")

        # L2: ip_gpt NovelArcs(drama 格式),命中后字段映射
        for novel_dir in self.cfg.novel_dirs:
            try:
                novel_arcs = await NovelArcs.download(
                    self.ctx, self.cos, novel_id=novel_id, novel_dir=novel_dir
                )
                arcs = novel_arcs.data.get("arcs", []) if novel_arcs else []
                if not arcs:
                    continue
                mapped = self._map_novel_arcs_to_story_arcs(arcs)
                story_arcs = StoryArcs(story_arcs=mapped)
                # 落本地
                story_arcs.save_to_file(local_path)
                logger.info_context(
                    self.ctx, f"Loaded NovelArcs {novel_id} from {novel_dir}"
                )
                return mapped
            except Exception:
                continue

        return None

    @staticmethod
    def _map_novel_arcs_to_story_arcs(arcs):
        """drama NovelArcs.data.arcs[*] -> shortanime StoryArcs item。
        丢弃 events/characters,只保留区间和标题/摘要。
        """
        mapped = []
        for a in arcs:
            mapped.append({
                "start_chapter_num": a.get("start_idx") or a.get("start_chapter_num"),
                "end_chapter_num": a.get("end_idx") or a.get("end_chapter_num"),
                "title": a.get("arc_title") or a.get("title", ""),
                "content": a.get("arc_summary") or a.get("content", ""),
            })
        return mapped

    async def load_fiction_world_building(self, novel_id):
        try:
            fiction_world_building = await WorldBuilding.download(self.ctx, self.cos, novel_id=novel_id)
            return fiction_world_building
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Failed to load WorldBuilding for {novel_id}: {e}",
            )
            return ""

    async def load_fiction_role_info(self, novel_id):
        try:
            fiction_role_info = await RoleInfo.download(self.ctx, self.cos, novel_id=novel_id)
            return fiction_role_info
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Failed to load RoleInfo for {novel_id}: {e}",
            )
            return ""

    async def _load_structured_data(self, novel_id):
        local_path = f"{CACHE_DIR}/{novel_id}/{novel_id}_structured.json"
        try:
            if os.path.exists(local_path):
                structured_data = NovelStructed.load_from_file(local_path)
                return structured_data.data

            for novel_dir in self.cfg.novel_dirs:
                try:
                    structured_data = await NovelStructed.download(
                        self.ctx, self.cos, novel_id=novel_id, novel_dir=novel_dir
                    )
                    logger.debug_context(
                        self.ctx, f"Loaded NovelStructed {novel_id} from {novel_dir}"
                    )
                    structured_data.save_to_file(local_path)
                    return structured_data.data
                except Exception:
                    continue

        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Failed to load NovelStructed for {novel_id}: {e}",
            )
            return []

    async def get_max_chapter_idx(self, novel_id):
        structured_data = await self._load_structured_data(novel_id)
        if not structured_data:
            return 0
        return max([x["chapter_idx"] for x in structured_data])

    async def load_chapter_summaries(self, novel_id, chapter_from=0, chapter_to=None):
        structured_data = await self._load_structured_data(novel_id)
        if not structured_data:
            return []

        chapter_from, chapter_to = self._clamp_chapter_range(chapter_from, chapter_to)
        if chapter_to is None:
            chapter_to = max([x["chapter_idx"] for x in structured_data])
        structured_data = [
            x for x in structured_data if chapter_from <= x["chapter_idx"] <= chapter_to
        ]

        chapter_summaries = []
        for segment in structured_data:
            summary_episode = [
                item
                for item in segment["for_emb"]
                if item["prompt_type"] == "summary_episode"
            ]
            if not summary_episode:
                continue
            chapter_mark = f"[CHAPTER_ID {segment['chapter_idx']}]"
            content = chapter_mark + "\n" + summary_episode[0]["text"].strip()
            chapter_summaries.append(content)

        return chapter_summaries

    async def load_chapter_summaries_list(self, novel_id, chapter_from=0, chapter_to=None):
        structured_data = await self._load_structured_data(novel_id)
        if not structured_data:
            return []

        chapter_from, chapter_to = self._clamp_chapter_range(chapter_from, chapter_to)
        chapter_to = chapter_to if chapter_to else len(structured_data)
        structured_data = [
            x for x in structured_data if chapter_from <= x["chapter_idx"] <= chapter_to
        ]

        chapter_summaries = []
        for segment in structured_data:
            summary_episode = [
                item
                for item in segment["for_emb"]
                if item["prompt_type"] == "summary_episode"
            ]
            if not summary_episode:
                continue
            content = summary_episode[0]["text"].strip()
            chapter_summaries.append({
                'chapter_num': segment['chapter_idx'],
                'content': content
            })

        return chapter_summaries

    async def load_chapter_contents(self, novel_id, chapter_from=0, chapter_to=None):
        """
            读取章节摘要
        """
        structured_data = await self._load_structured_data(novel_id)
        if not structured_data:
            return []

        chapter_from, chapter_to = self._clamp_chapter_range(chapter_from, chapter_to)
        chapter_to = chapter_to if chapter_to else len(structured_data)
        structured_data = [
            x for x in structured_data if chapter_from <= x["chapter_idx"] <= chapter_to
        ]

        chapter_contents = []
        for segment in structured_data:
            chapter_mark = f"[CHAPTER_ID {segment['chapter_idx']}]"
            lines = segment["head"] + segment["body"]
            lines = [line.strip() for line in lines if line.strip()]
            content = chapter_mark + "\n" + "\n".join(lines)
            chapter_contents.append(content)
        return chapter_contents

    async def load_chapter_contents_text(self, novel_id, chapter_from=0, chapter_to=None):
        chapter_contents = await self.load_chapter_contents(novel_id, chapter_from, chapter_to)
        return "\n\n".join(chapter_contents)

    async def load_story_arcs_text(self, novel_id, chapter_from=0, chapter_to=None):
        """
            加载故事弧
        """
        plot = []
        text = ""
        chapter_from, chapter_to = self._clamp_chapter_range(chapter_from, chapter_to)
        if chapter_to is None:
            chapter_to = await self.get_max_chapter_idx(novel_id)
        try:
            story_arcs = await self.load_story_arcs(novel_id)
            for story_arc in story_arcs:
                if chapter_from <= story_arc['start_chapter_num'] <= chapter_to or chapter_from <= story_arc['end_chapter_num'] <= chapter_to:
                    plot_id = len(plot)
                    plot.append({
                        "plot_id": plot_id,
                        "chapter_from": story_arc['start_chapter_num'],
                        "chapter_to": story_arc['end_chapter_num'],
                        "title": story_arc['title'],
                        "description": story_arc['content']
                    })
                    text += "\n\n" + f"PLOT_{plot_id} (§{story_arc['start_chapter_num']}-§{story_arc['end_chapter_num']}) [{story_arc['title']}]\n{story_arc['content']}"
            return plot, text.strip()
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Failed to load StoryArcs for {novel_id}: {e}",
            )
            raise

    # 自适应导入, arcs - summary - scripts
    async def load_contents_adaptive(self, novel_id, chapter_from=1, chapter_to=None, max_length=100000):
        chapter_from, chapter_to = self._clamp_chapter_range(chapter_from, chapter_to)
        scripts = await self.load_chapter_contents(novel_id, chapter_from, chapter_to)
        result_text = "\n\n".join(scripts)
        if len(result_text) <= max_length:
            logger.info_context(self.ctx, f"Loaded contents with `scripts`")
            return result_text
        summary = await self.load_chapter_summaries(novel_id, chapter_from, chapter_to)
        summary_text = "\n\n".join(summary)
        if len(summary_text) <= max_length:
            logger.info_context(self.ctx, f"Loaded contents with `summary`")
            return summary_text
        plot, arcs_text = await self.load_story_arcs_text(novel_id, chapter_from, chapter_to)
        logger.info_context(self.ctx, f"Loaded contents with `arcs`")
        return arcs_text

    async def load_contents_text(self, novel_id, chapter_from=1, chapter_to=None, ctype="summary"):
        chapter_from, chapter_to = self._clamp_chapter_range(chapter_from, chapter_to)
        if ctype == "summary":
            chapter_summaries = await self.load_chapter_summaries(novel_id, chapter_from, chapter_to)
            return "\n\n".join(chapter_summaries)
        elif ctype == "arcs":
            plot, arcs_text = await self.load_story_arcs_text(novel_id, chapter_from, chapter_to)
            return arcs_text
        else:
            scripts = await self.load_chapter_contents(novel_id, chapter_from, chapter_to)
            return "\n\n".join(scripts)

    # 精确提取章节
    async def load_chapter_summaries_exact(self, novel_id, chapter_list):
        structured_data = await self._load_structured_data(novel_id)
        if not structured_data:
            return []
        chapter_summaries = []
        for segment in structured_data:
            if segment["chapter_idx"] in chapter_list:
                summary_episode = [
                    item
                    for item in segment["for_emb"]
                    if item["prompt_type"] == "summary_episode"
                ]
                if not summary_episode:
                    continue
                chapter_mark = f"[CHAPTER_ID {segment['chapter_idx']}]"
                content = chapter_mark + "\n" + summary_episode[0]["text"].strip()
                chapter_summaries.append(content)
        return chapter_summaries

    async def load_chapter_contents_exact(self, novel_id, chapter_list, chapter_from=0, chapter_to=None):
        structured_data = await self._load_structured_data(novel_id)
        if not structured_data:
            return []

        chapter_from, chapter_to = self._clamp_chapter_range(chapter_from, chapter_to)
        chapter_to = chapter_to if chapter_to else len(structured_data)
        structured_data = [
            x for x in structured_data if chapter_from <= x["chapter_idx"] <= chapter_to
        ]
        chapter_contents = []
        for segment in structured_data:
            if segment['chapter_idx'] in chapter_list:
                lines = segment["head"] + segment["body"]
                lines = [line.strip() for line in lines if line.strip()]
                chapter_mark = f"[CHAPTER_ID {segment['chapter_idx']}]"
                content = chapter_mark + "\n" + "\n".join(lines)
                chapter_contents.append(content)

        return chapter_contents

    async def load_contents_exact(self, novel_id, chapter_list, chapter_from=0, chapter_to=None, ctype="summary"):
        if ctype == "summary":
            return await self.load_chapter_summaries_exact(novel_id, chapter_list)
        else:
            return await self.load_chapter_contents_exact(novel_id, chapter_list)
