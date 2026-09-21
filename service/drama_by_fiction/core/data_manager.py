import os

from trpc.log import logger
from cos_orm import BaseCosModel

from service.drama_by_fiction.data_model import FictionConfig
from service.drama_by_fiction.data_model import (
    Adaptation,
    SeasonProposal,
    NovelStructed,
    StoryArcs,
    NovelArcs,
)

CACHE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", ".cache", "drama_by_fiction"
)

COS = None


class DataManager:
    def __init__(self, ctx):
        self.cos = COS
        self.ctx = ctx
        self.cfg = FictionConfig()

    async def upload(self, data: BaseCosModel, **path_args):
        try:
            data_name = data.__class__.__name__
            url = data.get_path(**path_args)
            await data.upload(self.ctx, self.cos, **path_args)
            logger.info_context(self.ctx, f"Uploaded {data_name} to {url}")
            return url
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Uploaded {data_name} to {url} Failed: {e}",
            )
            return ""

    async def load_adaptation(self, story_id: str):
        try:
            adaptation = await Adaptation.download(
                self.ctx, self.cos, story_id=story_id
            )
            return adaptation
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Failed to load Adaptation for {story_id}: {e}",
            )
            return None

    async def load_season_proposal(self, story_id: str, season_id):
        try:
            season_adaptation = await SeasonProposal.download(
                self.ctx, self.cos, story_id=story_id, season_id=season_id
            )
            return season_adaptation
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Failed to load SeasonAdaptation for {story_id} season {season_id+1}: {e}",
            )
            return None

    async def load_novel_arcs(self, novel_id: str):
        local_path = f"{CACHE_DIR}/{novel_id}/{novel_id}_arc.json"
        try:
            if os.path.exists(local_path):
                arcs_data = NovelArcs.load_from_file(local_path)
                event_data = arcs_data.data.get("arcs", [])
                return event_data

            for novel_dir in self.cfg.novel_dirs:
                try:
                    arcs_data = await NovelArcs.download(
                        self.ctx, self.cos, novel_id=novel_id, novel_dir=novel_dir
                    )
                    logger.info_context(
                        self.ctx, f"Loaded NovelArcs {novel_id} from {novel_dir}"
                    )
                    arcs_data.save_to_file(local_path)
                    return arcs_data.data.get("arcs", [])
                except Exception:
                    continue

        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Failed to load NovelStructed for {novel_id}: {e}",
            )
            return []

    async def load_story_arcs(self, novel_id: str):
        local_path = f"{CACHE_DIR}/{novel_id}/{novel_id}_arc.json"
        try:
            if os.path.exists(local_path):
                arcs_data = StoryArcs.load_from_file(local_path)
                event_data = arcs_data.data.get("arcs", [])
                return event_data

            arcs_data = await StoryArcs.download(self.ctx, self.cos, novel_id=novel_id)
            arcs_data.save_to_file(local_path)
            return arcs_data.data.get("arcs", [])
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Failed to load StoryArcs for {novel_id}: {e}",
            )
            return []

    async def load_format_story_arcs(self, novel_id, start_idx=0, end_idx=None):
        end_idx = end_idx if end_idx else await self.get_max_chapter_idx(novel_id)

        plots, texts = [], []
        try:
            story_arcs = await self.load_story_arcs(novel_id)
            for story_arc in story_arcs:
                arc_start = story_arc["start_idx"]
                arc_end = story_arc["end_idx"]
                if not (
                    start_idx <= arc_start <= end_idx or start_idx <= arc_end <= end_idx
                ):
                    continue

                events = []
                for story_event in story_arc["events"]:

                    # chapter_range 可能多于2个元素
                    chapter_range = story_event.get("chapter_range", [0, 0])
                    event_start, event_end = chapter_range[0], chapter_range[-1]
                    event_item = {
                        "event_id": str(story_event.get("id", "")),
                        "chapter_from": int(event_start),
                        "chapter_to": int(event_end),
                        "title": str(story_event.get("title", "")),
                        "description": str(story_event.get("summary", "")),
                    }
                    events.append(event_item)
                plot_id = len(plots)
                plot = {
                    "plot_id": plot_id,
                    "chapter_from": arc_start,
                    "chapter_to": arc_end,
                    "title": story_arc["arc_title"],
                    "description": story_arc["arc_summary"],
                    "events": events,
                }
                plots.append(plot)
                texts.append(
                    f"PLOT_{plot_id} (§{arc_start}-§{arc_end}) [{plot['title']}]\n{plot['description']}"
                )
            arc_text = "\n\n".join(texts)
            logger.info_context(
                self.ctx,
                f"{novel_id} {len(plots)} plots, arc_text {len(arc_text)} words",
            )
            return plots, arc_text
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Failed to load StoryArcs for {novel_id}: {e}",
            )
            raise

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
                    logger.info_context(
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

    async def get_max_chapter_idx(self, novel_id, structured_data=None):
        if structured_data is None:
            structured_data = await self._load_structured_data(novel_id)
        if not structured_data:
            return 0
        items = [item for seg in structured_data for item in seg["for_emb"]]
        max_chapter_idx = max([x["ch_idx"] for x in items])
        logger.info_context(self.ctx, f"{novel_id} max_chapter_idx: {max_chapter_idx}")
        return max_chapter_idx

    async def load_chpater_items(
        self, novel_id, start_idx=0, end_idx=None, ctype="summary_episode"
    ):
        """以ch_idx为下标，同一ch_idx的多个文本段会合并为一个item，按ch_idx排序返回"""
        structured_data = await self._load_structured_data(novel_id)
        if not structured_data:
            return []

        max_chapter_idx = await self.get_max_chapter_idx(novel_id, structured_data)
        end_idx = end_idx if end_idx else max_chapter_idx

        raw_items = [
            item
            for seg in structured_data
            for item in seg["for_emb"]
            if item["prompt_type"] == ctype and start_idx <= item["ch_idx"] <= end_idx
        ]

        # 按ch_idx合并同章的多个文本段，按ch_idx排序返回
        # 摘要(summary_episode)的text第一行是章头标题需要剥离，原文(original_text)无章头直接拼接
        merged = {}
        for item in raw_items:
            ch_idx = int(item["ch_idx"])
            text = item["text"].strip()
            if ctype == "summary_episode":
                lines = text.split("\n")
                text = "\n".join(lines[1:]).strip() if len(lines) > 1 else lines[0]
            if ch_idx not in merged:
                merged[ch_idx] = {**item, "ch_idx": ch_idx, "text": text}
            else:
                merged[ch_idx]["text"] += "\n" + text

        return [merged[ch_idx] for ch_idx in sorted(merged.keys())]

    async def load_chapter_summaries(self, novel_id, start_idx=0, end_idx=None):
        items = await self.load_chpater_items(
            novel_id, start_idx, end_idx, ctype="summary_episode"
        )
        return [f"第 {item['ch_idx']} 章\n{item['text']}" for item in items]

    async def load_chapter_contents(self, novel_id, start_idx=0, end_idx=None):
        items = await self.load_chpater_items(
            novel_id, start_idx, end_idx, ctype="original_text"
        )
        return [f"第 {item['ch_idx']} 章\n{item['text']}" for item in items]

    async def load_chapter_contents_text(
        self, novel_id, chapter_from=0, chapter_to=None
    ):
        chapter_contents = await self.load_chapter_contents(
            novel_id, chapter_from, chapter_to
        )
        return "\n\n".join(chapter_contents)

    async def load_contents_adaptive(
        self, novel_id, chapter_from=1, chapter_to=None, max_length=100000
    ):
        """
        自适应加载小说内容，优先返回完整脚本，其次返回章节摘要，最后返回故事弧线
        """
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

        _, arcs_text = await self.load_format_story_arcs(
            novel_id, chapter_from, chapter_to
        )
        logger.info_context(self.ctx, f"Loaded contents with `arcs`")
        return arcs_text

    async def load_contents_layered(
        self,
        novel_id,
        key_chapters,
        chapter_from=None,
        chapter_to=None,
        max_length=100000,
    ):
        """
        分层加载小说内容：核心章节加载原文，间隙章节加载摘要。

        策略：
        1. key_chapters（实际引用章节）优先加载原文
        2. chapter_from~chapter_to 中非 key_chapters 的间隙章节加载摘要，补充上下文
        3. 如果原文超限，核心章节也降级为摘要
        4. 如果摘要还超限，回退到故事弧线

        Args:
            novel_id: 小说ID
            key_chapters: 实际引用的章节列表（核心章节），加载原文
            chapter_from: 范围起始章节（默认取 key_chapters 最小值）
            chapter_to: 范围结束章节（默认取 key_chapters 最大值）
            max_length: 最大字符数限制
        """
        if not key_chapters:
            return ""

        key_set = set(key_chapters)
        chapter_from = chapter_from if chapter_from is not None else min(key_chapters)
        chapter_to = chapter_to if chapter_to is not None else max(key_chapters)

        # 间隙章节 = 范围内但不在核心章节中的章节
        gap_chapters = [
            ch for ch in range(chapter_from, chapter_to + 1) if ch not in key_set
        ]

        # 尝试策略1：核心章节原文 + 间隙章节摘要
        key_contents = await self.load_chpater_items(
            novel_id, chapter_from, chapter_to, ctype="original_text"
        )
        key_contents = [item for item in key_contents if item["ch_idx"] in key_set]

        gap_summaries = []
        if gap_chapters:
            all_summaries = await self.load_chpater_items(
                novel_id, chapter_from, chapter_to, ctype="summary_episode"
            )
            gap_summaries = [
                item for item in all_summaries if item["ch_idx"] not in key_set
            ]

        # 合并并按章节顺序排列
        merged = self._merge_layered_items(key_contents, gap_summaries)
        result_text = "\n\n".join(merged)

        if len(result_text) <= max_length:
            logger.info_context(
                self.ctx,
                f"Loaded layered contents: {len(key_contents)} key(original) + "
                f"{len(gap_summaries)} gap(summary), total {len(result_text)} chars",
            )
            return result_text

        # 策略2：全部用摘要
        all_summaries = await self.load_chapter_summaries(
            novel_id, chapter_from, chapter_to
        )
        summary_text = "\n\n".join(all_summaries)
        if len(summary_text) <= max_length:
            logger.info_context(
                self.ctx, f"Loaded layered contents fallback to all summaries"
            )
            return summary_text

        # 策略3：回退到故事弧线
        _, arcs_text = await self.load_format_story_arcs(
            novel_id, chapter_from, chapter_to
        )
        logger.info_context(self.ctx, f"Loaded layered contents fallback to arcs")
        return arcs_text

    @staticmethod
    def _merge_layered_items(key_items, gap_items):
        """
        将核心章节原文和间隙章节摘要按章节顺序合并，并添加标记。

        核心章节标记为 [原文]，间隙章节标记为 [摘要]，便于 LLM 区分详略。
        """
        all_items = []
        for item in key_items:
            all_items.append((item["ch_idx"], item["text"], True))
        for item in gap_items:
            all_items.append((item["ch_idx"], item["text"], False))

        all_items.sort(key=lambda x: x[0])

        merged = []
        for ch_idx, text, is_key in all_items:
            tag = "[原文]" if is_key else "[摘要]"
            merged.append(f"第 {ch_idx} 章 {tag}\n{text}")
        return merged


if __name__ == "__main__":
    import asyncio
    import trpc_cos
    from trpc import context, config, plugin

    config.load_global_config("trpc_python.yaml", "utf-8")
    plugin.setup_master()

    COS = trpc_cos.Client("trpc.script.drama_operator.CommonCos")

    dm = DataManager(ctx=context.Context())
    dm.cos = COS
    novel_id = "Fiction-yFHu7j6y"

    # chapter_contents = asyncio.run(
    #     dm.load_contents_layered(novel_id, key_chapters=[1, 4, 6, 10])
    # )
    # print(chapter_contents)
    chapter_contents = asyncio.run(dm.load_chapter_contents(novel_id, 1, 220))
    save_path = f"{CACHE_DIR}/{novel_id}/{novel_id}_contents.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(chapter_contents))
