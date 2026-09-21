from trpc_script_drama_operator.drama_operator_pb2 import GenerateScriptProposalRsp


from trpc import context
from trpc_script_drama_operator import pb


from service.drama_by_fiction.core import script_proposal
from service.drama_by_fiction.core.fiction_analyzer import FictionAnalyzer
from service.drama_by_fiction.core.script_proposal import ProposalGenerator
from service.drama_by_fiction.core.point_planner import PointPlanner

from service.drama_by_fiction.utils.pb_utils import (
    parse_story_info,
    parse_generate_input,
    parse_story_outline,
    convert_to_script_proposal,
)


async def _do_generate_proposal(
    ctx: context.Context,
    story_info,
    generate_input,
    suggestion="暂无",
    trigger_event_extract=False,
):
    """核心策划生成逻辑（generate_proposal 和 regenerate_proposal 的公共实现）"""
    if trigger_event_extract:
        analyze = FictionAnalyzer(ctx)
        await analyze.generate_story_events(generate_input.novel_id)

    adapter = ProposalGenerator(ctx)
    role_infos, world_building = await adapter.generate_proposal(
        story_info, generate_input, suggestion=suggestion
    )

    # 转化为pb格式
    role_infos = [role_infos.content] * generate_input.season_nums
    world_building = [world_building.content] * generate_input.season_nums

    return pb.GenerateScriptProposalRsp(
        story_outline=pb.StoryOutline(
            story_outline="", role_info=role_infos, world_building=world_building
        )
    )


async def generate_proposal(
    ctx: context.Context, request: pb.GenerateScriptProposalByFictionReq
):
    """生成剧本策划（首次生成，触发故事弧事件提取）"""
    story_info = parse_story_info(ctx, request)
    generate_input = parse_generate_input(ctx, request)
    return await _do_generate_proposal(
        ctx, story_info, generate_input, trigger_event_extract=True
    )


async def regenerate_proposal(
    ctx: context.Context, request: pb.RegenerateScriptProposalByFictionReq
) -> GenerateScriptProposalRsp:
    """重新生成剧本策划（带用户建议，不重新提取事件）"""
    story_info = parse_story_info(ctx, request)
    generate_input = parse_generate_input(ctx, request)
    suggestion = request.regenerate_data.suggestion
    return await _do_generate_proposal(
        ctx, story_info, generate_input, suggestion=suggestion
    )


async def generate_point_plan(
    ctx: context.Context, request: pb.RegenerateScriptProposalByFictionReq
) -> GenerateScriptProposalRsp:
    """生成卡点规划"""
    point_planner = PointPlanner(ctx)

    story_info = parse_story_info(ctx, request)
    generate_input = parse_generate_input(ctx, request)
    world_buildings, role_infos, _ = parse_story_outline(
        ctx, request.regenerate_data.story_outline, generate_input.season_nums
    )
    suggestion = request.regenerate_data.suggestion

    proposals = await point_planner.generate_point_plan(
        story_info, generate_input, world_buildings[0], role_infos[0], suggestion
    )
    script_proposal = convert_to_script_proposal(proposals)
    return pb.GenerateScriptProposalRsp(
        story_outline=pb.StoryOutline(
            story_outline="", role_info=role_infos, world_building=world_buildings
        ),
        script_proposal=script_proposal,
    )
