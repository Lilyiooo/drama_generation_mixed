from trpc.log import logger
from cos_orm import BaseCosModel, set_default_timeout, set_default_retry

from service.shortanime_by_fiction.data_models import FictionConfig
from service.shortanime_by_fiction.data_models import (
    RoleInfo,
    WorldBuilding,
    StoryOutline,
    SeasonProposal,
    ActOutline,
    Adaptation,
    EpisodeOutline,
    EpisodeScript,
    StoryInfo,
)

COS = None

class DramaDataManager:
    def __init__(self, ctx):
        self.cos = COS
        self.ctx = ctx
        self.cfg = FictionConfig()
        set_default_timeout(60)
        set_default_retry(3)

    async def upload(self, data: BaseCosModel, **path_args):
        data_name = data.__class__.__name__
        url = data.get_path(**path_args)
        try:
            logger.debug_context(self.ctx, f"Start uploading {data_name} to {url}")
            await data.upload(self.ctx, self.cos, **path_args)
            logger.debug_context(self.ctx, f"Uploaded {data_name} to {url}")
            return url
        except Exception as e:
            logger.error_context(self.ctx, f"All upload attempts failed for {data_name} to {url}: {type(e).__name__}: {e}")
            return ""

    async def load_base_info(self, story_info: StoryInfo):
        return "暂无"

    async def load_role_info(self, story_info: StoryInfo):
        try:
            role_info = await RoleInfo.download(
                self.ctx, self.cos, story_info=story_info
            )
            return role_info
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Failed to load RoleInfo for {story_info.story_id}",
            )
            return "暂无"

    async def load_world_building(self, story_info: StoryInfo):
        try:
            world_building = await WorldBuilding.download(
                self.ctx, self.cos, story_info=story_info
            )
            return world_building
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Failed to load WorldBuilding for {story_info.story_id}: {e}",
            )
            return "暂无"

    async def load_adaptation(self, story_info: StoryInfo):
        try:
            adaptation = await Adaptation.download(self.ctx, self.cos, story_info=story_info)
            return adaptation
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Failed to load Adaptation for {story_info.story_id}: {e}",
            )
            return None

    async def load_season_division(self, story_info: StoryInfo):
        try:
            season_division = await SeasonProposal.download(self.ctx, self.cos, story_info=story_info)
            return season_division
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Failed to load DramaOutline for {story_info.story_id}: {e}",
            )
            return None

    async def load_season_proposal(self, story_info: StoryInfo, season_id):
        try:
            season_adaptation = await SeasonProposal.download(
                self.ctx, self.cos, story_info=story_info, season_id=season_id
            )
            return season_adaptation
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Failed to load SeasonAdaptation for {story_info.story_id} season {season_id+1}: {e}",
            )
            return None

    async def load_season_proposals(self, story_info: StoryInfo, season_nums):
        season_proposals = []
        try:
            for season_id in range(season_nums):
                season_proposal = await self.load_season_proposal(story_info, season_id)
                season_proposals.append(season_proposal)
            return season_proposals
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Failed to load SeasonProposal for {story_info.story_id}: {e}",
            )
        return None

    async def load_story_outline(self, story_info: StoryInfo, season_id):
        try:
            story_outline = await StoryOutline.download(
                self.ctx, self.cos, story_info=story_info, season_id=season_id
            )
            return story_outline
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Failed to load StoryOutline for {story_info.story_id} season {season_id+1}: {e}",
            )
            return None

    async def load_episode_outline(self, story_info: StoryInfo, season_id, episode_id):
        try:
            episode_outline = await EpisodeOutline.download(
                self.ctx, self.cos, story_info=story_info, season_id=season_id, episode_id=episode_id
            )
            return episode_outline
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Failed to load ActOutline for {story_info.story_id} season {season_id+1}: {e}",
            )
            return None

    async def load_episode_outlines(self, story_info: StoryInfo, season_id, episode_ids):
        episode_outlines = []
        for episode_id in episode_ids:
            try:
                episode_outline = await self.load_episode_outline(story_info, season_id, episode_id)
                if episode_outline is not None:
                    episode_outlines.append(episode_outline)
            except Exception as e:
                logger.error_context(
                    self.ctx,
                    f"Failed to load EpisodeOutline for {story_info.story_id} season {season_id+1} episode {episode_id+1}: {e}",
                )
        return episode_outlines

    async def load_episode_script(self, story_info: StoryInfo, season_id, episode_id):
        try:
            episode_script = await EpisodeScript.download(
                self.ctx, self.cos, story_info=story_info, season_id=season_id, episode_id=episode_id
            )
            return episode_script
        except Exception as e:
            logger.error_context(
                self.ctx,
                f"Failed to load EpisodeScript for {story_info.story_id} season {season_id+1} episode {episode_id+1}: {e}",
            )
            return None

    async def load_episode_scripts(self, story_info: StoryInfo, season_id, episode_ids):
        episode_scripts = []
        for episode_id in episode_ids:
            try:
                episode_script = await self.load_episode_script(story_info, season_id, episode_id)
                if episode_script is not None:
                    episode_scripts.append(episode_script)
            except Exception as e:
                logger.error_context(
                    self.ctx,
                    f"Failed to load EpisodeScript for {story_info.story_id} season {season_id+1} episode {episode_id+1}: {e}",
                )
        return episode_scripts
