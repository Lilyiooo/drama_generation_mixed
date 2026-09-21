from trpc import context
from trpc.log import logger
from trpc_script_drama_operator import pb

from service.shortanime_by_fiction.data_models import StoryArcs

from service.shortanime_by_fiction.data_manager import DataManager
from service.shortanime_by_fiction.core.analyze_fiction import FictionAnalyzer
from service.shortanime_by_fiction.core.story_outline.generator import StoryOutlineGenerator
from service.shortanime_by_fiction.core.proposal.proposal_generator import ProposalGenerator
from service.shortanime_by_fiction.tools.pb_utils import (
    parse_story_info,
    parse_generate_input,
    parse_proposals,
    message_to_dict,
    convert_to_drama
)
from service.shortanime_by_fiction.tools.params_check import (
    check_proposals
)


async def generate_story_outline(
    ctx: context.Context, request: pb.GenerateStoryOutlineByFictionReq
):
    generator = StoryOutlineGenerator(ctx)
    adapter = ProposalGenerator(ctx)
    data_manager = DataManager(ctx)

    story_info, generate_input = parse_story_info(ctx, request), parse_generate_input(ctx, request)
    proposals = parse_proposals(ctx, request.generate_data.script_proposal)
    role_infos = request.generate_data.story_outline.role_info
    world_buildings = request.generate_data.story_outline.world_building
    season_division = proposals[0]
    demo_drama = message_to_dict(request.generate_data.demo_drama)

    check_proposals(ctx, story_info, generate_input, proposals)

    story_outlines = []
    for season_id in range(generate_input.season_nums):
        try:
            season_proposal = proposals[season_id+1]
            season_demo_drama = demo_drama['seasons'][season_id]['episodes']
            chapter_from, chapter_to = adapter.get_chapter_range(season_division, season_id)
            
            # season_story_outline = await data_manager.load_story_outline(story_info.story_id, season_id)
            season_story_outline = await generator.generate_story_outline(
                story_info, 
                generate_input,
                season_id,
                world_buildings[0],
                role_infos[0],
                chapter_from,
                chapter_to,
                season_proposal,
                season_demo_drama
            )
            story_outlines.append(season_story_outline.content)
        except Exception as e:
            logger.error(f"generate story outline failed: {e}")
            raise e
    
    story_outline = pb.StoryOutline(
        story_outline=story_outlines,
        role_info=role_infos,
        world_building=world_buildings
    )

    return pb.GenerateStoryOutlineRsp(
        story_outline=story_outline
    )

async def regenerate_story_outline(
    ctx: context.Context, request: pb.RegenerateStoryOutlineByFictionReq
):
    generator = StoryOutlineGenerator(ctx)
    adapter = ProposalGenerator(ctx)
    data_manager = DataManager(ctx)

    story_info, generate_input = parse_story_info(ctx, request), parse_generate_input(ctx, request)
    proposals = parse_proposals(ctx, request.regenerate_data.generate_data.script_proposal)
    role_infos = request.regenerate_data.generate_data.story_outline.role_info
    world_buildings = request.regenerate_data.generate_data.story_outline.world_building
    season_division = proposals[0]
    demo_drama = message_to_dict(request.regenerate_data.generate_data.demo_drama)
    suggestion = request.regenerate_data.suggestion
    season_id = request.regenerate_data.season_id

    check_proposals(ctx, story_info, generate_input, proposals)

    try:
        season_proposal = proposals[season_id+1]
        season_demo_drama = demo_drama['seasons'][season_id]['episodes']
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
            season_demo_drama,
            suggestion=suggestion
        )
    except Exception as e:
        logger.error(f"regenerate story outline failed: {e}")
        raise e

    return pb.RegenerateStoryOutlineRsp(
        result=season_story_outline.content
    )
