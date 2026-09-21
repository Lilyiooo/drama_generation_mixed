import trpc
from trpc import context
from trpc_script_drama_operator import rpc, pb

from trpc_cos import Client as Cos
from service.shortdrama_by_fiction.gen_story_outline import (
    generate_story_outlines_and_role_infos,
    regenerate_story_outline,
    regenerate_role_info,
)
from service.shortdrama_by_fiction.gen_episode_outline import (
    generate_episode_outline,
    regenerate_episode_outline_all,
    regenerate_episode_outline_single,
)
from service.shortdrama_by_fiction.gen_episode_script import generate_episode_scripts
from service.shortdrama_by_fiction.core import data_manager
from tools.log_request import log_request

class ShortDramaByFiction(rpc.DramaByFictionServicer):
    def __init__(self, cos: Cos):
        self.cos = cos
        data_manager.COS = cos

    @log_request
    async def GenerateStoryOutline(
        self, ctx: context.Context, request: pb.GenerateStoryOutlineByFictionReq
    ) -> pb.GenerateStoryOutlineRsp:
        return await generate_story_outlines_and_role_infos(ctx, request)

    @log_request
    async def RegenerateStoryOutline(
        self, ctx: context.Context, request: pb.RegenerateStoryOutlineByFictionReq
    ) -> pb.RegenerateStoryOutlineRsp:
        return await regenerate_story_outline(ctx, request)

    @log_request
    async def RegenerateRoleInfo(
        self, ctx: context.Context, request: pb.RegenerateStoryOutlineByFictionReq
    ) -> pb.RegenerateStoryOutlineRsp:
        return await regenerate_role_info(ctx, request)

    @log_request
    async def GenerateEpisodeOutline(
        self, ctx: context.Context, request: pb.GenerateEpisodeOutlineByFictionReq
    ) -> pb.GenerateEpisodeOutlineRsp:
        return await generate_episode_outline(ctx, request)

    @log_request
    async def RegenerateEpisodeOutlineAll(
        self, ctx: context.Context, request: pb.RegenerateEpisodeOutlineAllByFictionReq
    ) -> pb.GenerateEpisodeOutlineRsp:
        return await regenerate_episode_outline_all(ctx, request)

    @log_request
    async def RegenerateEpisodeOutlineSingle(
        self,
        ctx: context.Context,
        request: pb.RegenerateEpisodeOutlineSingleByFictionReq,
    ) -> pb.RegenerateEpisodeOutlineSingleRsp:
        return await regenerate_episode_outline_single(ctx, request)

    @log_request
    async def GenerateDrama(
        self, ctx: context.Context, request: pb.GenerateDramaByFictionReq
    ) -> pb.GenerateDramaRsp:
        return await generate_episode_scripts(ctx, request)


def format_print(data):
    import json
    from google.protobuf.json_format import MessageToDict

    episode_outlines_dict = MessageToDict(data, preserving_proto_field_name=True)
    episode_outlines = json.dumps(episode_outlines_dict, ensure_ascii=False, indent=2)
    print(episode_outlines)

async def load_episode_outlines(ctx, cos, story_id, season_cnt, episode_cnt):
    seasons = []
    for season_id in range(season_cnt):
        episodes = []
        _outline = await EpisodeOutline_Season.download(
            ctx, cos, story_id=story_id, season_id=season_id
        ) 
        episodes = _outline.to_dict() 
        print("episode_season outline",type(episodes))
        season = pb.EpisodeOutline_Season(episodes=episodes, season_id=season_id)
        seasons.append(season)

    return seasons

if __name__ == "__main__":
    import asyncio
    from trpc import config
    from trpc.plugin import plugin

    config.load_global_config("trpc_python.yaml", "utf-8")
    plugin.setup_master()
    
    drama = ShortDramaByFiction()
    ctx = context.Context()

    story_info = pb.StoryInfo(
        story_id="test0001", story_name="折腰", plot_type=pb.PlotType.SHORT
    )
    generate_input = pb.GenerateInputByFiction(
        novel_id="fiction-Ly0BCvck",
        common=pb.GenerateInputCommon(
        season_nums=1,
        episode_nums=20,
        duration=3
    ))

    request = pb.GenerateStoryOutlineByFictionReq(
        story_info=story_info, generate_input=generate_input
    )
    story_outlines = asyncio.run(drama.GenerateStoryOutline(ctx, request))
    format_print(story_outlines)
    # ------------------------------------------------------------------------------#

    # RegenerateStoryOutline
    from service.shortdrama_by_fiction.tests.test_data2 import *
    role_infos = [""]
    story_outline_pb = pb.StoryOutline(
        story_outline=story_outlines,
        role_info=role_infos,
    )
    regenerate_data = pb.RegenerateStoryOutline(
        story_outline=story_outline_pb,
        suggestion="剧情调整为大乔重生、大乔与魏劭之间的爱情，小乔与刘琰的婚姻双线，主题是放下仇恨，大小乔推动和解，魏绍、刘琰握手言欢",
    )

    request = pb.RegenerateStoryOutlineByFictionReq(
        story_info=story_info,
        generate_input=generate_input,
        regenerate_data=regenerate_data,
        season_id=1,
    )
    # story_outline = asyncio.run(drama.RegenerateStoryOutline(ctx, request))
    # format_print(story_outline)
    # ------------------------------------------------------------------------------#

    regenerate_data = pb.RegenerateStoryOutline(
        story_outline=story_outline_pb,
        suggestion="提取主要核心角色，按戏份排序，不少于10个角色",
    )

    request = pb.RegenerateStoryOutlineByFictionReq(
        story_info=story_info,
        generate_input=generate_input,
        regenerate_data=regenerate_data,
        season_id=1,
    )

    # role_info = asyncio.run(drama.RegenerateRoleInfo(ctx, request))
    # format_print(role_info)
    # ------------------------------------------------------------------------------#

    from service.shortdrama_by_fiction.test_data2 import *
    role_infos = [""]
    story_outline_pb = pb.StoryOutline(
        story_outline=story_outlines,
        role_info=role_infos,
    )
    request = pb.GenerateEpisodeOutlineByFictionReq(
        story_info=story_info,
        generate_input=generate_input,
        story_outline=story_outline_pb,
    )
    episode_outlines = asyncio.run(drama.GenerateEpisodeOutline(ctx, request))
    # format_print(episode_outlines)
    # ------------------------------------------------------------------------------#
    seasons = asyncio.run(load_episode_outlines(ctx, cos, story_info.story_id, 
                                                generate_input.common.season_nums, 
                                                generate_input.common.episode_nums))
    regenerate_data = pb.RegenerateEpisodeOutline(
        story_outline=story_outline_pb,
        episode_outline=pb.EpisodeOutline(seasons=seasons, episodes=10), #episode_outline_data["episode_outline"],
        suggestion="每集最少有1个名场面，带入情绪变化/矛盾转移信息，不超过500字",
    )
    request = pb.RegenerateEpisodeOutlineAllByFictionReq(
        story_info=story_info,
        generate_input=generate_input,
        regenerate_data=regenerate_data,
    )
    # new_episode_outlines = asyncio.run(drama.RegenerateEpisodeOutlineAll(ctx, request))
    # format_print(new_episode_outlines)
    # ------------------------------------------------------------------------------#
    seasons = asyncio.run(load_episode_outlines(ctx, cos, story_info.story_id, 
                                                generate_input.common.season_nums, 
                                                generate_input.common.episode_nums))
    se_idx = 0
    ep_idx = 2
    t_epi_outline = message_to_dict(seasons[se_idx].episodes[ep_idx])
    t_epi_outline["episode_outline"] = t_epi_outline["content"]
    del t_epi_outline["content"]
    request = pb.RegenerateEpisodeOutlineSingleByFictionReq(
        story_info=story_info,
        generate_input=generate_input,
        **t_epi_outline,
        suggestion="每集悬念前置",
    )
    # new_single_outline = asyncio.run(drama.RegenerateEpisodeOutlineSingle(ctx, request))
    # format_print(new_single_outline)
    # ------------------------------------------------------------------------------#

    select_ranges = [{"season_id": 1, "episode_ids": [1, 2, 3]}]
    request = pb.GenerateDramaByFictionReq(
        story_info=story_info,
        generate_input=generate_input,
        generate_data=pb.GenerateDrama(
            story_outline=story_outline_pb,
            episode_outline=pb.EpisodeOutline(seasons=seasons, episodes=20),
            prev_episode_content="",
            suggestion="",
            select_range=select_ranges,
        ),
    )
    episode_scripts = asyncio.run(drama.GenerateDrama(ctx, request))
    format_print(episode_scripts)
    # ------------------------------------------------------------------------------#
    