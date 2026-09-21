from trpc import context
from trpc.log import logger
from trpc_script_drama_operator import rpc, pb

from service.shortanime_by_fiction.data_manager import DataManager
from service.shortanime_by_fiction.core.proposal.proposal_generator import ProposalGenerator
from service.shortanime_by_fiction.core.analyze_fiction import FictionAnalyzer
from service.shortanime_by_fiction.data_models import (
    StoryInfo, 
    GenerateInput,
    SeasonProposal
)
from service.shortanime_by_fiction.tools.pb_utils import (
    parse_story_info,
    parse_generate_input,
    parse_proposals,
    convert_to_script_proposal
)
from service.shortanime_by_fiction.tools.params_check import (
    check_proposals
)

async def get_estimated_time(
    ctx: context.Context,
    request
):
    story_info, generate_input = parse_story_info(ctx, request), parse_generate_input(ctx, request)
    estimated_time = 240 + 180 * generate_input.season_nums
    return pb.GetEstimatedTimeRsp(estimated_time=estimated_time)


async def generate_proposal(
    ctx: context.Context,
    request
):
    story_info, generate_input = parse_story_info(ctx, request), parse_generate_input(ctx, request)
    
    analyze = FictionAnalyzer(ctx)
    adapter = ProposalGenerator(ctx)

    # 触发故事弧提取
    story_arc = await analyze.get_story_arcs(generate_input.novel_id)

    role_infos, world_building, proposals = await adapter.generate_proposal(
        story_info, 
        generate_input
    )

    # 转化为pb格式
    role_infos = [role_infos.content] * generate_input.season_nums
    world_building = [world_building.content] * generate_input.season_nums
    script_proposal = convert_to_script_proposal(proposals)

    ret = pb.GenerateScriptProposalRsp(
        story_outline=pb.StoryOutline(
            story_outline="",
            role_info=role_infos,
            world_building=world_building
        ),
        script_proposal=script_proposal
    )

    return ret

async def regenerate_proposal(
    ctx: context.Context,
    request
):
    story_info, generate_input = parse_story_info(ctx, request), parse_generate_input(ctx, request)
    suggestion = request.regenerate_data.suggestion

    analyze = FictionAnalyzer(ctx)
    adapter = ProposalGenerator(ctx)

    role_infos, world_building, proposals = await adapter.generate_proposal(
        story_info, 
        generate_input,
        suggestion=suggestion
    )

    # 转化为pb格式
    role_infos = [role_infos.content] * generate_input.season_nums
    world_building = [world_building.content] * generate_input.season_nums
    script_proposal = convert_to_script_proposal(proposals)

    ret = pb.GenerateScriptProposalRsp(
        story_outline=pb.StoryOutline(
            story_outline="",
            role_info=role_infos,
            world_building=world_building
        ),
        script_proposal=script_proposal
    )

    return ret

async def regenerate_season_division(
    ctx: context.Context,
    request
):
    logger.info_context(ctx, f"Start regenerate season division...")
    story_info, generate_input = parse_story_info(ctx, request), parse_generate_input(ctx, request)

    data_manager = DataManager(ctx)
    adapter = ProposalGenerator(ctx)

    proposals = parse_proposals(ctx, request.regenerate_data.script_proposal)
    role_infos = request.regenerate_data.story_outline.role_info
    world_building = request.regenerate_data.story_outline.world_building
    season_id = request.regenerate_data.season_id

    check_proposals(ctx, story_info, generate_input, proposals)

    season_division = proposals[0]
    # 整体规划
    if season_id == -1:

        proposals = await adapter.regenerate_season_division(
            story_info,
            generate_input,
            season_division
        )

    # 分季规划
    else:
        season_proposal = proposals[season_id + 1]

        season_proposal = await adapter.regenerate_season_points(
            story_info,
            generate_input,
            season_id,
            season_proposal
        )

        proposals[season_id + 1] = season_proposal

    script_proposal = convert_to_script_proposal(proposals)
    
    ret = pb.GenerateScriptProposalRsp(
        story_outline=pb.StoryOutline(
            story_outline="",
            role_info=role_infos,
            world_building=world_building
        ),
        script_proposal=script_proposal
    )

    return ret