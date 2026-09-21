import functools

from trpc import context
from trpc.log import logger
import trpc_rainbow as _
from trpc.config import load, with_codec, with_provider
from trpc_script_drama_operator import rpc, pb
from trpc_cos import Client as Cos

from service.shortanime_by_fiction.llms import prompt_manager
from service.shortanime_by_fiction import data_manager
from service.shortanime_by_fiction.handlers.gen_proposal import (
    get_estimated_time,
    generate_proposal,
    regenerate_proposal,
    regenerate_season_division
)
from service.shortanime_by_fiction.handlers.gen_story_outline import (
    generate_story_outline,
    regenerate_story_outline
)
from service.shortanime_by_fiction.handlers.gen_episode_outline import (
    generate_episode_outline,
    regenerate_episode_outline_all,
    regenerate_episode_outline_single
)
from service.shortanime_by_fiction.handlers.gen_episode_script import (
    generate_demo_scripts,
    generate_scripts
)
from tools.log_request import log_request


def load_prompts(func):
    """
    装饰器函数：加载提示词配置
    """

    @functools.wraps(func)
    async def wrapper(self, ctx: context.Context, request, *args, **kwargs):
        # 加载提示词配置
        try:
            rainbow_config = await load(
                "shortanime_prompt.yaml",
                with_provider("rainbow"),
                with_codec("YamlDecoder"),
            )
            prompt_manager.prompt_data = rainbow_config.config_data
        except Exception as e:
            logger.error(f"load rainbow prompts error: {e}")

        # 调用原始函数
        return await func(self, ctx, request, *args, **kwargs)

    return wrapper


class ShortDramaByFiction(rpc.DramaByFictionServicer):
    def __init__(self, cos: Cos):
        self.cos = cos
        data_manager.COS = cos
        data_manager._drama_module.COS = cos
        data_manager._fiction_module.COS = cos

    @log_request
    @load_prompts
    async def GetEstimatedTime(
        self, ctx: context.Context, request: pb.GetEstimatedTimeReq
    ) -> pb.GetEstimatedTimeRsp:
        return await get_estimated_time(ctx, request)

    @log_request
    @load_prompts
    async def GenerateScriptProposal(
        self, ctx: context.Context, request: pb.GenerateScriptProposalByFictionReq
    ) -> pb.GenerateScriptProposalRsp:
        return await generate_proposal(ctx, request)

    @log_request
    @load_prompts
    async def RegenerateScriptProposal(
        self, ctx: context.Context, request: pb.RegenerateScriptProposalByFictionReq
    ) -> pb.GenerateScriptProposalRsp:
        return await regenerate_proposal(ctx, request)

    @log_request
    @load_prompts
    async def RegenerateSeasonDivision(
        self, ctx: context.Context, request: pb.RegenerateScriptProposalByFictionReq
    ) -> pb.GenerateScriptProposalRsp:
        return await regenerate_season_division(ctx, request)

    @log_request
    @load_prompts
    async def GenerateDemoDrama(
        self, ctx: context.Context, request: pb.GenerateDramaByFictionReq
    ) -> pb.GenerateDramaRsp:
        return await generate_demo_scripts(ctx, request)

    @log_request
    @load_prompts
    async def GenerateStoryOutline(
        self, ctx: context.Context, request: pb.GenerateStoryOutlineByFictionReq
    ) -> pb.GenerateStoryOutlineRsp:
        return await generate_story_outline(ctx, request)

    @log_request
    @load_prompts
    async def RegenerateStoryOutline(
        self, ctx: context.Context, request: pb.RegenerateStoryOutlineByFictionReq
    ) -> pb.RegenerateStoryOutlineRsp:
        return await regenerate_story_outline(ctx, request)

    # @log_request
    # @load_prompts
    # async def RegenerateRoleInfo(
    #     self, ctx: context.Context, request: pb.RegenerateStoryOutlineByFictionReq
    # ) -> pb.RegenerateStoryOutlineRsp:
    #     return await regenerate_role_info(ctx, request)

    @log_request
    @load_prompts
    async def GenerateEpisodeOutline(
        self, ctx: context.Context, request: pb.GenerateEpisodeOutlineByFictionReq
    ) -> pb.GenerateEpisodeOutlineRsp:
        return await generate_episode_outline(ctx, request)

    @log_request
    @load_prompts
    async def RegenerateEpisodeOutlineAll(
        self, ctx: context.Context, request: pb.RegenerateEpisodeOutlineAllByFictionReq
    ) -> pb.GenerateEpisodeOutlineRsp:
        return await regenerate_episode_outline_all(ctx, request)

    @log_request
    @load_prompts
    async def RegenerateEpisodeOutlineSingle(
        self,
        ctx: context.Context,
        request: pb.RegenerateEpisodeOutlineSingleByFictionReq,
    ) -> pb.RegenerateEpisodeOutlineSingleRsp:
        return await regenerate_episode_outline_single(ctx, request)

    @log_request
    @load_prompts
    async def GenerateDrama(
        self, ctx: context.Context, request: pb.GenerateDramaByFictionReq
    ) -> pb.GenerateDramaRsp:
        return await generate_scripts(ctx, request)