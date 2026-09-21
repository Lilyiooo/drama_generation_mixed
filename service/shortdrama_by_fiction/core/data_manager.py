
# run in server: PYTHONPATH=/usr/local/trpc/bin/lib:$PYTHONPATH python3 service/shortdrama_by_fiction/testinit.py 
import os
import json

from trpc.log import logger
from cos_orm import BaseCosModel
#from service.shortdrama_by_fiction.tools.cos_client import get_cos_client
from service.shortdrama_by_fiction.data_model import FictionConfig
from service.shortdrama_by_fiction.data_model import (
    RoleInfo,
    WorldBuilding,
    NovelStructed,
    StoryArcs,
    EpisodeOutline_Season,
    SeasonOutline,
    ShortDramaInfo,
    ShortDramaPartPlan
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".cache")
if os.path.exists("/group/30109/"):
    CACHE_DIR = "service/shortdrama_by_fiction/tests/data"

COS = None

class DataManager:
    def __init__(self, ctx):
        #get_cos_client()
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
                f"Uploaded {data_name} to cos Failed: {e}",
            )
            return ""

    async def load_season_outline(self, story_id, season_id):
        try:
            season_outline = await SeasonOutline.download(
                self.ctx, self.cos, story_id=story_id, season_id=season_id
            )
            return season_outline
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Failed to load SeasonOutline for {story_id} season {season_id+1}: {e}",
            )
            return None


    async def load_episodeoutline_season(self, story_id, season_id=1):
        try:
            outline = await EpisodeOutline_Season.download(
                self.ctx, self.cos, story_id=story_id, season_id=season_id
            )
            return outline
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Failed to load oneSE outline for {story_id} season {season_id}: {e}",
            )
            return None
    """ 
    async def load_shortdrama_outline(self, story_id, season_id=1):
        try:
            outline = await ShortDramaInfo.download(
                self.ctx, self.cos, story_id=story_id, season_id=season_id
            )
            return outline
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Failed to load shordrama outline for {story_id} season {season_id}: {e}",
            )
            return None
    """

    async def load_shortdrama_info(self, story_id, season_id):
        try:
            info = await ShortDramaInfo.download(
                self.ctx, self.cos, story_id=story_id, season_id=season_id
            )
            #return json.loads(info.content)
            return info
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Failed to load shortdrama info for {story_id} season {season_id}: {e}",
            )
            return None

    async def load_base_info(self, story_id):
        return "暂无"

    async def load_role_info(self, story_id, season_id):
        try:
            role_info = await RoleInfo.download(
                self.ctx, self.cos, story_id=story_id, season_id=season_id
            )
            return role_info.content
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Failed to load RoleInfo for {story_id} season {season_id+1}: {e}",
            )
            return "暂无"

    async def load_shortdrama_plan(self, story_id, season_id=1):
        try:
            shortdrama_plan = await ShortDramaPartPlan.download(
                self.ctx, self.cos, story_id=story_id, season_id=season_id
            )
            return shortdrama_plan.content
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Failed to load shortdrama plan for {story_id} season {season_id}: {e}",
            )
            return None
    async def load_base_info(self, story_id):
        return "暂无"


    async def load_world_building(self, story_id, season_id):
        try:
            world_building = await WorldBuilding.download(
                self.ctx, self.cos, story_id=story_id, season_id=season_id
            )
            return world_building.content
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Failed to load WorldBuilding for {story_id} season {season_id}: {e}",
            )
            return "暂无"

    async def load_story_arcs(self, novel_id):
        try:
            story_arcs = await StoryArcs.download(self.ctx, self.cos, novel_id=novel_id)
            return story_arcs.story_arc
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Failed to load StoryArcs for {novel_id}: {e}",
            )
            return ""


        
    async def _load_structured_data(self, novel_id):
        local_path = f"{CACHE_DIR}/{novel_id}/{novel_id}_structured.json"
        print("_______________local_path:", local_path)
        try:
            if os.path.exists(local_path):
                structured_data = NovelStructed.load_from_file(local_path)
                return structured_data.data

            structured_data = await NovelStructed.download(
                self.ctx, self.cos, novel_id=novel_id
            )
            structured_data.save_to_file(local_path)
            return structured_data.data
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Failed to load NovelStructed for {novel_id}: {e}",
            )
            return []

    async def load_chapter_summaries(self, novel_id, start_idx=0, end_idx=100000):
        structured_data = await self._load_structured_data(novel_id)
        if not structured_data:
            return []

        end_idx = end_idx if end_idx else len(structured_data)
        print(start_idx, end_idx, type(start_idx), type(end_idx))
        structured_data = [
            x for x in structured_data if start_idx <= x["chapter_idx"] <= end_idx
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

    async def load_chapter_contents(self, novel_id, start_idx=0, end_idx=100000):
        structured_data = await self._load_structured_data(novel_id)
        if not structured_data:
            return []

        end_idx = end_idx if end_idx else len(structured_data)
        structured_data = [
            x for x in structured_data if start_idx <= x["chapter_idx"] <= end_idx
        ]

        chapter_contents = []
        last_head = ""
        for segment in structured_data:
            if last_head != segment["head"]:
                lines = segment["head"] + segment["body"]
            else:
                lines = segment["body"]
            last_head = segment["head"]
            lines = [line.strip() for line in lines if line.strip()]
            content = "\n".join(lines)
            chapter_contents.append(content)
        return chapter_contents

    async def load_contents(self, novel_id, start_idx=1, end_idx=100000, ctype="summary"):
        if ctype == "summary":
            return await self.load_chapter_summaries(novel_id, start_idx, end_idx)
        else:
            return await self.load_chapter_contents(novel_id, start_idx, end_idx)
