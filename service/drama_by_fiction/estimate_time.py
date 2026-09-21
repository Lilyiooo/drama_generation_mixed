from trpc import context
from trpc.log import logger
from trpc_script_drama_operator import pb

from service.drama_by_fiction.data_model import FictionConfig
from service.drama_by_fiction.utils.pb_utils import parse_generate_input


async def get_default_time(ctx: context.Context, request):
    generate_input = parse_generate_input(ctx, request)
    default = 240 + 300 * generate_input.season_nums
    return default


async def get_gen_story_outline_time(
    ctx: context.Context, request: pb.GenerateStoryOutlineByFictionReq
):
    generate_input = parse_generate_input(ctx, request)
    default = 60 + 300 * generate_input.season_nums
    return default


async def get_gen_episode_outline_time(
    ctx: context.Context, request: pb.GenerateEpisodeOutlineByFictionReq
):
    cfg = FictionConfig()
    season_cnt = len(request.generate_data.select_range)
    episode_cnts = [
        len(_range.episode_ids) for _range in request.generate_data.select_range
    ]
    episode_batch_cnt = max(max(episode_cnts) // cfg.max_workers, 1)
    estimated_time = 60 + season_cnt * episode_batch_cnt * 300
    return estimated_time


async def get_gen_drama_time(
    ctx: context.Context, request: pb.GenerateDramaByFictionReq
):
    cfg = FictionConfig()
    epi_outline = request.generate_data.episode_outline

    max_scene_cnt = 0
    for season in epi_outline.seasons:
        for episode in season.episodes:
            scene_cnt = len(episode.scenes)
            max_scene_cnt = max(max_scene_cnt, scene_cnt)

    season_cnt = len(request.generate_data.select_range)
    episode_cnts = [
        len(_range.episode_ids) for _range in request.generate_data.select_range
    ]
    episode_batch_cnt = max(max(episode_cnts) // cfg.max_workers, 1)
    estimated_time = 60 + season_cnt * episode_batch_cnt * max_scene_cnt * 300
    return estimated_time


# rpc_name 到具体 request 类型的映射
RPC_REQUEST_TYPE_MAP = {
    "GenerateScriptProposal": pb.GenerateScriptProposalByFictionReq,
    "RegenerateScriptProposal": pb.RegenerateScriptProposalByFictionReq,
    "GeneratePointPlan": pb.RegenerateScriptProposalByFictionReq,
    "GenerateStoryOutline": pb.GenerateStoryOutlineByFictionReq,
    "RegenerateStoryOutline": pb.RegenerateStoryOutlineByFictionReq,
    "GenerateEpisodeOutline": pb.GenerateEpisodeOutlineByFictionReq,
    "GenerateEpisodeScenePlan": pb.GenerateEpisodeOutlineByFictionReq,
    "GenerateDrama": pb.GenerateDramaByFictionReq,
    "GenerateSceneOutline": pb.GenerateSceneOutlineByFictionReq,
    "GenerateSceneDrama": pb.GenerateSceneDramaByFictionReq,
    "GenerateSceneOutlinePrompt": pb.GenerateSceneOutlineByFictionReq,
    "GenerateSceneDramaPrompt": pb.GenerateSceneDramaByFictionReq,
}


def _unpack_request(rpc_name: str, any_request):
    """将 Any 类型的 request 解包为具体的 protobuf 消息类型。"""
    # 从映射表中查找
    if rpc_name in RPC_REQUEST_TYPE_MAP:
        msg = RPC_REQUEST_TYPE_MAP[rpc_name]()
        any_request.Unpack(msg)
        return msg

    # 兜底：逐一尝试所有已知类型
    for msg_cls in RPC_REQUEST_TYPE_MAP.values():
        msg = msg_cls()
        try:
            if any_request.Is(msg.DESCRIPTOR):
                any_request.Unpack(msg)
                return msg
        except Exception:
            continue

    raise ValueError(f"无法解包 rpc_name={rpc_name} 对应的 Any request")


async def get_estimated_time(ctx: context.Context, request):
    rpc_name = request.rpc_name
    rpc_request = _unpack_request(rpc_name, request.request)

    estimated_time = 300

    if rpc_name == "GenerateEpisodeOutline":
        estimated_time = await get_gen_episode_outline_time(ctx, rpc_request)
    elif rpc_name == "GenerateDrama":
        estimated_time = await get_gen_drama_time(ctx, rpc_request)
    elif rpc_name == "GenerateStoryOutline":
        estimated_time = await get_gen_story_outline_time(ctx, rpc_request)
    elif rpc_name == "GenerateStoryOutline":
        estimated_time = 300
    elif rpc_name == "GenerateSceneOutline":
        estimated_time = 300
    elif rpc_name == "GenerateSceneDrama":
        estimated_time = 300
    else:
        estimated_time = await get_default_time(ctx, rpc_request)

    logger.info(f"rpc {rpc_name} estimated_time: {estimated_time}")
    return pb.GetEstimatedTimeRsp(estimated_time=estimated_time)
