from drama_local import runtime as context
from drama_local import models as pb
import asyncio

from .generate_story_outline_and_role import generate_story_outline_and_role
from .regenerate_story_outline import regenerate_story_outline
from .regenerate_role import regenerate_role
from .generate_episode_outline import generate_episode_outline
from .generate_episode_script import generate_episode_script
from .regenerate_all_episode_outline import regenerate_all_episode_outline
from .regenerate_episode_outline import regenerate_episode_outline_single
from .generate_script_proposal import generate_script_proposal
from .regenerate_script_proposal import regenerate_script_proposal
from .regenerate_season_division import regenerate_season_division
from .utils import *

from dataclasses import dataclass
from drama_local import runtime as context
from drama_local.runtime import LocalStore as Cos

from drama_local.runtime import logger
import traceback
from drama_local.log_request import log_request


class DramaByCreativity:
    def __init__(self, cos: Cos):
        self.cos = cos

    @log_request
    async def GenerateStoryOutline(
        self, ctx: context.Context, request: pb.GenerateStoryOutlineByCreativityReq
    ) -> pb.GenerateStoryOutlineRsp:
        try:
            outline_role_rets = await generate_story_outline_and_role(ctx, request, self.cos)
        except Exception as e:
            logger.error_context(ctx, f"Error: {e}, traceback: {traceback.format_exc()}")
            raise
        return outline_role_rets

    @log_request
    async def RegenerateStoryOutline(
        self, ctx: context.Context, request: pb.RegenerateStoryOutlineByCreativityReq
    ) -> pb.RegenerateStoryOutlineRsp:
        try:
            regenerated_story_outline = await regenerate_story_outline(ctx, request)
        except Exception as e:
            logger.error_context(ctx, f"Error: {e}, traceback: {traceback.format_exc()}")
            raise
        return regenerated_story_outline

    @log_request
    async def RegenerateRoleInfo(
        self, ctx: context.Context, request: pb.RegenerateStoryOutlineByCreativityReq
    ) -> pb.RegenerateStoryOutlineRsp:
        try:
            regenerated_role = await regenerate_role(ctx, request)
        except Exception as e:
            logger.error_context(ctx, f"Error: {e}, traceback: {traceback.format_exc()}")
            raise
        return regenerated_role

    @log_request
    async def GenerateEpisodeOutline(
        self, ctx: context.Context, request: pb.GenerateEpisodeOutlineByCreativityReq
    ) -> pb.GenerateEpisodeOutlineRsp:
        try:
            episode_outline = await generate_episode_outline(ctx, request, self.cos)
        except Exception as e:
            logger.error_context(ctx, f"Error: {e}, traceback: {traceback.format_exc()}")
            raise
        return episode_outline

    @log_request
    async def RegenerateEpisodeOutlineAll(
        self,
        ctx: context.Context,
        request: pb.RegenerateEpisodeOutlineAllByCreativityReq,
    ) -> pb.GenerateEpisodeOutlineRsp:
        try:
            regenerated_all_episode_outline = await regenerate_all_episode_outline(
                ctx, request
            )
        except Exception as e:
            logger.error_context(ctx, f"Error: {e}, traceback: {traceback.format_exc()}")
            raise
        return regenerated_all_episode_outline

    @log_request
    async def RegenerateEpisodeOutlineSingle(
        self,
        ctx: context.Context,
        request: pb.RegenerateEpisodeOutlineSingleByCreativityReq,
    ) -> pb.RegenerateEpisodeOutlineSingleRsp:
        try:
            regenerated_episode_outline = await regenerate_episode_outline_single(
                ctx, request
            )
        except Exception as e:
            logger.error_context(ctx, f"Error: {e}, traceback: {traceback.format_exc()}")
            raise
        return regenerated_episode_outline

    @log_request
    async def GenerateScriptProposal(
        self, ctx: context.Context, request: pb.GenerateScriptProposalByCreativityReq
    ) -> pb.GenerateScriptProposalRsp:
        try:
            script_proposal = await generate_script_proposal(ctx, request)
        except Exception as e:
            logger.error_context(ctx, f"Error: {e}, traceback: {traceback.format_exc()}")
            raise
        return script_proposal

    @log_request
    async def RegenerateScriptProposal(
        self, ctx: context.Context, request: pb.RegenerateScriptProposalByCreativityReq
    ) -> pb.GenerateScriptProposalRsp:
        try:
            script_proposal = await regenerate_script_proposal(ctx, request)
        except Exception as e:
            logger.error_context(ctx, f"Error: {e}, traceback: {traceback.format_exc()}")
            raise
        return script_proposal

    @log_request
    async def GenerateDrama(
        self, ctx: context.Context, request: pb.GenerateDramaByCreativityReq
    ) -> pb.GenerateDramaRsp:
        try:
            episode_script = await generate_episode_script(ctx, request, self.cos)
        except Exception as e:
            logger.error_context(ctx, f"Error: {e}, traceback: {traceback.format_exc()}")
            raise
        episode_script.episode_outline.CopyFrom(request.generate_data.episode_outline)
        return episode_script

    @log_request
    async def RegenerateSeasonDivision(self, ctx: context.Context,
                                       request: pb.RegenerateScriptProposalByCreativityReq) -> pb.GenerateScriptProposalRsp:
        try:
            script_proposal = await regenerate_season_division(ctx, request)
        except Exception as e:
            logger.error_context(ctx, f"Error: {e}, traceback: {traceback.format_exc()}")
            raise
        return script_proposal
