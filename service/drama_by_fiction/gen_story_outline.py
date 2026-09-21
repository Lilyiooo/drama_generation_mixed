from trpc import context
from trpc.log import logger
from trpc_script_drama_operator import pb

from service.drama_by_fiction.core.generate_outline import OutlineGenerator
from service.drama_by_fiction.core.script_proposal import ProposalGenerator


from service.drama_by_fiction.utils.pb_utils import (
    parse_story_info,
    parse_generate_input,
    parse_proposals,
    check_proposals,
)


async def generate_story_outline(
    ctx: context.Context, request: pb.GenerateStoryOutlineByFictionReq
):
    generator = OutlineGenerator(ctx)
    adapter = ProposalGenerator(ctx)

    story_info = parse_story_info(ctx, request)
    generate_input = parse_generate_input(ctx, request)
    proposals = parse_proposals(ctx, request.generate_data.script_proposal)

    role_infos = request.generate_data.story_outline.role_info
    world_buildings = request.generate_data.story_outline.world_building
    season_division = proposals[0]

    check_proposals(ctx, story_info, generate_input, proposals)

    story_outlines = []
    for season_id in range(generate_input.season_nums):
        try:
            season_proposal = proposals[season_id + 1]
            chapter_from, chapter_to = adapter.get_chapter_range(
                season_division, season_id
            )
            season_story_outline = await generator.generate_story_outline(
                story_info,
                generate_input,
                season_id,
                world_buildings[0],
                role_infos[0],
                chapter_from,
                chapter_to,
                season_proposal,
            )
            story_outlines.append(season_story_outline.content)
        except Exception as e:
            logger.error(f"generate story outline failed: {e}")
            raise e

    story_outline = pb.StoryOutline(
        story_outline=story_outlines,
        role_info=role_infos,
        world_building=world_buildings,
    )

    return pb.GenerateStoryOutlineRsp(story_outline=story_outline)


async def regenerate_story_outline(
    ctx: context.Context, request: pb.RegenerateStoryOutlineByFictionReq
):
    generator = OutlineGenerator(ctx)
    adapter = ProposalGenerator(ctx)

    story_info = parse_story_info(ctx, request)
    generate_input = parse_generate_input(ctx, request)
    proposals = parse_proposals(
        ctx, request.regenerate_data.generate_data.script_proposal
    )

    role_infos = request.regenerate_data.generate_data.story_outline.role_info
    world_buildings = request.regenerate_data.generate_data.story_outline.world_building
    season_division = proposals[0]
    suggestion = request.regenerate_data.suggestion
    season_id = request.regenerate_data.season_id

    check_proposals(ctx, story_info, generate_input, proposals)

    try:
        season_proposal = proposals[season_id + 1]
        chapter_from, chapter_to = adapter.get_chapter_range(season_division, season_id)

        season_story_outline = await generator.generate_story_outline(
            story_info,
            generate_input,
            season_id,
            world_buildings[0],
            role_infos[0],
            chapter_from,
            chapter_to,
            season_proposal,
            suggestion=suggestion,
        )
    except Exception as e:
        logger.error(f"regenerate story outline failed: {e}")
        raise

    return pb.RegenerateStoryOutlineRsp(result=season_story_outline.content)
