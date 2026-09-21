import os
from concurrent.futures import ThreadPoolExecutor

from trpc import context
from trpc.log import logger
from trpc_script_drama_operator import pb

from service.shortanime_by_fiction.data_manager import DataManager
from service.shortanime_by_fiction.core.script.writer import ScriptWriter
from service.shortanime_by_fiction.core.proposal.proposal_generator import ProposalGenerator
from service.shortanime_by_fiction.core.episode_outline.scheduler import ProgressScheduler
from service.shortanime_by_fiction.tools.pb_utils import (
    parse_story_info,
    parse_generate_input,
    parse_proposals,
    parse_story_outline,
    parse_episode_outlines,
    parse_drama,
    convert_to_drama,
    convert_to_episode_outline
)
from service.shortanime_by_fiction.tools.params_check import (
    check_proposals
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".cache", "fiction")

async def generate_demo_scripts(
    ctx: context.Context, request: pb.GenerateDramaByFictionReq
):
    story_info, generate_input = parse_story_info(ctx, request), parse_generate_input(ctx, request)
    proposals = parse_proposals(ctx, request.generated_data_content.script_proposal)
    role_info = request.generated_data_content.story_outline.role_info[0]
    world_building = request.generated_data_content.story_outline.world_building[0]
    suggestion = request.generate_params.suggestion

    check_proposals(ctx, story_info, generate_input, proposals)

    writer = ScriptWriter(ctx)
    adapter = ProposalGenerator(ctx)
    data_manager = DataManager(ctx)
    
    drama, outline = [], []

    season_division = proposals[0]

    for season_id in range(generate_input.season_nums):
        season_proposal = proposals[season_id+1]
        try:
            if season_proposal.point_planning.points != []:
                # 前3集对应章节范围
                chapter_from, chapter_to = adapter.get_chapter_range(season_proposal, 1)
            else:
                chapter_from, chapter_to = adapter.get_chapter_range(season_division, season_id)
        except Exception as e:
            logger.info_context(ctx, f"获取项目第{season_id + 1}季章节范围失败")
            raise

        # scripts = await data_manager.load_episode_scripts(story_info.story_id, season_id, [0, 1, 2])
        scripts, outlines = await writer.generate_demo_scripts(
            story_info,
            generate_input,
            season_id,
            world_building,
            role_info,
            chapter_from,
            chapter_to,
            season_proposal,
            suggestion=suggestion
        )
        drama.append({
            "season_id": season_id,
            "episodes": scripts
        })
        outline.append({
            "season_id": season_id,
            "episodes": outlines
        })

    drama = convert_to_drama(drama)
    outline = convert_to_episode_outline(outline)

    return pb.GenerateDramaRsp(
        result=drama,
        episode_outline=outline
    )

async def generate_scripts(
    ctx: context.Context, request: pb.GenerateDramaByFictionReq
):
    story_info, generate_input = parse_story_info(ctx, request), parse_generate_input(ctx, request)
    proposals = parse_proposals(ctx, request.generated_data_content.script_proposal)
    season_division = proposals[0]

    world_buildings, role_infos, story_outlines = parse_story_outline(ctx, request.generated_data_content.story_outline, generate_input.season_nums)
    episode_outlines = parse_episode_outlines(ctx, request.generated_data_content.episode_outline, generate_input.season_nums)
    drama_data = parse_drama(ctx, request.generated_data_content.drama, generate_input.season_nums)

    suggestion = request.generate_params.suggestion

    writer = ScriptWriter(ctx)
    adapter = ProposalGenerator(ctx)

    check_proposals(ctx, story_info, generate_input, proposals)

    drama, outline = [], []

    for range in request.generate_params.select_range:

        try:
            season_id, episode_ids = range.season_id, range.episode_ids
            season_proposal = proposals[season_id+1]
            chapter_from, chapter_to = adapter.get_chapter_range(season_division, season_id)

            season_episode_outlines = episode_outlines[season_id]
            season_story_outline = story_outlines[season_id]
            season_drama = drama_data[season_id]

            season_episode_outline_dict = {outline.episode_id: outline for outline in season_episode_outlines}
            season_scripts_dict = {drama.episode_id: drama for drama in season_drama}

            scripts = await writer.generate_episode_scripts(
                story_info,
                generate_input,
                season_id,
                episode_ids,
                world_buildings[0],
                role_infos[0],
                chapter_from,
                chapter_to,
                season_proposal,
                season_story_outline,
                season_episode_outline_dict,
                season_scripts_dict,
                suggestion=suggestion
            )
        except Exception as e:
            logger.error(f"generate_scripts failed: {e}")
            raise

        drama.append({
            "season_id": season_id,
            "episodes": scripts
        })
        outline.append({
            "season_id": season_id,
            "episodes": season_episode_outlines
        })

    drama = convert_to_drama(drama)
    outline = convert_to_episode_outline(outline)

    return pb.GenerateDramaRsp(
        result=drama,
        episode_outline=outline
    )