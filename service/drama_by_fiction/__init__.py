from trpc import context
from trpc_script_drama_operator import rpc, pb
from trpc_cos import Client as Cos

from service.drama_by_fiction.core import data_manager

from service.drama_by_fiction.estimate_time import get_estimated_time
from service.drama_by_fiction.gen_proposal import (
    generate_proposal,
    regenerate_proposal,
    generate_point_plan,
)

from service.drama_by_fiction.gen_story_outline import (
    generate_story_outline,
    regenerate_story_outline,
)
from service.drama_by_fiction.gen_episode_outline import (
    generate_episode_outlines,
    generate_episode_scene_plan,
    generate_scene_outline,
    generate_scene_outline_prompt,
)
from service.drama_by_fiction.gen_episode_script import (
    generate_scripts,
    generate_scene_script,
    generate_scene_script_prompt,
)

from tools.log_request import log_request


class DramaByFiction(rpc.DramaByFictionServicer):
    def __init__(self, cos: Cos):
        self.cos = cos
        data_manager.COS = cos

    @log_request
    async def GetEstimatedTime(
        self, ctx: context.Context, request: pb.GetEstimatedTimeReq
    ) -> pb.GetEstimatedTimeRsp:
        return await get_estimated_time(ctx, request)

    @log_request
    async def GenerateScriptProposal(
        self, ctx: context.Context, request: pb.GenerateScriptProposalByFictionReq
    ) -> pb.GenerateScriptProposalRsp:
        # 只生成世界观和人设
        return await generate_proposal(ctx, request)

    @log_request
    async def RegenerateScriptProposal(
        self, ctx: context.Context, request: pb.RegenerateScriptProposalByFictionReq
    ) -> pb.GenerateScriptProposalRsp:
        # 只生成世界观和人设
        return await regenerate_proposal(ctx, request)

    @log_request
    async def GeneratePointPlan(
        self, ctx: context.Context, request: pb.RegenerateScriptProposalByFictionReq
    ) -> pb.GenerateScriptProposalRsp:
        # 生成/重新生成卡点规划
        return await generate_point_plan(ctx, request)

    @log_request
    async def GenerateStoryOutline(
        self, ctx: context.Context, request: pb.GenerateStoryOutlineByFictionReq
    ) -> pb.GenerateStoryOutlineRsp:
        return await generate_story_outline(ctx, request)

    @log_request
    async def RegenerateStoryOutline(
        self, ctx: context.Context, request: pb.RegenerateStoryOutlineByFictionReq
    ) -> pb.RegenerateStoryOutlineRsp:
        return await regenerate_story_outline(ctx, request)

    @log_request
    async def GenerateEpisodeOutline(
        self, ctx: context.Context, request: pb.GenerateEpisodeOutlineByFictionReq
    ) -> pb.GenerateEpisodeOutlineRsp:
        # 生成集大纲
        return await generate_episode_outlines(ctx, request)

    @log_request
    async def GenerateEpisodeScenePlan(
        self, ctx: context.Context, request: pb.GenerateEpisodeOutlineByFictionReq
    ) -> pb.GenerateEpisodeOutlineRsp:
        # 生成集分场规划
        return await generate_episode_scene_plan(ctx, request)

    @log_request
    async def GenerateDrama(
        self, ctx: context.Context, request: pb.GenerateDramaByFictionReq
    ) -> pb.GenerateDramaRsp:
        return await generate_scripts(ctx, request)

    @log_request
    async def GenerateSceneOutline(
        self, ctx: context.Context, request: pb.GenerateSceneOutlineByFictionReq
    ) -> pb.GenerateSceneOutlineRsp:
        # 指定生成单场大纲
        return await generate_scene_outline(ctx, request)

    @log_request
    async def GenerateSceneDrama(
        self, ctx: context.Context, request: pb.GenerateSceneDramaByFictionReq
    ) -> pb.GenerateSceneDramaRsp:
        # 指定生成单场剧本
        return await generate_scene_script(ctx, request)

    @log_request
    async def GenerateSceneOutlinePrompt(
        self, ctx: context.Context, request: pb.GenerateSceneOutlineByFictionReq
    ) -> pb.GeneratePromptRsp:
        # 返回生成单场大纲的完整提示词
        return await generate_scene_outline_prompt(ctx, request)

    @log_request
    async def GenerateSceneDramaPrompt(
        self, ctx: context.Context, request: pb.GenerateSceneDramaByFictionReq
    ) -> pb.GeneratePromptRsp:
        # 返回生成单场剧本的完整提示词
        return await generate_scene_script_prompt(ctx, request)
