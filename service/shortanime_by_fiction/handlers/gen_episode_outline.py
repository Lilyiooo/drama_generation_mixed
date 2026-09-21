import os
import asyncio

import re
from trpc import context
from trpc.log import logger
from trpc_script_drama_operator import pb


from service.shortanime_by_fiction.data_manager import DataManager
from service.shortanime_by_fiction.core.analyze_fiction import FictionAnalyzer
from service.shortanime_by_fiction.data_models.drama import EpisodeOutline
from service.shortanime_by_fiction.core.proposal.proposal_generator import ProposalGenerator
from service.shortanime_by_fiction.core.episode_outline.scheduler import ProgressScheduler

from service.shortanime_by_fiction.tools.pb_utils import (
    parse_story_info,
    parse_generate_input,
    parse_proposals,
    parse_story_outline,
    convert_to_drama,
    convert_to_episode_outline
)
from service.shortanime_by_fiction.tools.params_check import (
    check_proposals
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".cache", "fiction")


async def generate_episode_outline(
    ctx: context.Context, request: pb.GenerateEpisodeOutlineByFictionReq
):
    story_info, generate_input = parse_story_info(ctx, request), parse_generate_input(ctx, request)
    proposals = parse_proposals(ctx, request.generated_data.script_proposal)
    season_division = proposals[0]

    world_buildings, role_infos, story_outlines = parse_story_outline(ctx, request.generated_data.story_outline, generate_input.season_nums)
    
    write_scheduler = ProgressScheduler(ctx)
    adapter = ProposalGenerator(ctx)

    # 构建结果
    outline = []
    for season_id in range(generate_input.season_nums):
        try:
            season_proposal = proposals[season_id+1]
            chapter_from, chapter_to = adapter.get_chapter_range(season_division, season_id)
            season_story_outline = story_outlines[season_id]
        except Exception as e:
            logger.error(f"Error in parsing params in generate_episode_outline: {e}, season_id: {season_id}")
            raise e

        episode_outlines = await write_scheduler.generate_episode_outlines(
            story_info, 
            generate_input,
            season_id,
            world_buildings[0],
            role_infos[0],
            chapter_from,
            chapter_to,
            season_proposal,
            season_story_outline
        )
        
        outline.append({
            "season_id": season_id,
            "episodes": episode_outlines
        })
    
    outline = convert_to_episode_outline(outline)

    return pb.GenerateEpisodeOutlineRsp(episode_outline=outline)


async def regenerate_episode_outline_all(
    ctx: context.Context, request: pb.RegenerateEpisodeOutlineAllByFictionReq
):
    story_info, generate_input = parse_story_info(ctx, request), parse_generate_input(ctx, request)
    proposals = parse_proposals(ctx, request.generated_data.script_proposal)
    world_buildings = request.generated_data.story_outline.world_building
    role_infos = request.generated_data.story_outline.role_info
    story_outlines = request.generated_data.story_outline.story_outline
    season_division = proposals[0]
    suggestion = request.regenerate_data.suggestion
    
    write_scheduler = ProgressScheduler(ctx)
    adapter = ProposalGenerator(ctx)

    check_proposals(ctx, story_info, generate_input, proposals)

    # 构建结果
    outline = []
    for season_id in range(generate_input.season_nums):
        try:
            season_proposal = proposals[season_id+1]
            chapter_from, chapter_to = adapter.get_chapter_range(season_division, season_id)
            season_story_outline = story_outlines[season_id]
        except Exception as e:
            logger.error(f"Error in parsing params in regenerate_episode_outline_all: {e}, season_id: {season_id}")
            raise e

        # 串行执行
        episode_outlines = await write_scheduler.generate_episode_outlines(
            story_info, 
            generate_input,
            season_id,
            world_buildings[0],
            role_infos[0],
            chapter_from,
            chapter_to,
            season_proposal,
            season_story_outline,
            suggestion=suggestion
        )
        
        outline.append({
            "season_id": season_id,
            "episodes": episode_outlines
        })

    outline = convert_to_episode_outline(outline)

    return pb.GenerateEpisodeOutlineRsp(episode_outline=outline)


async def regenerate_episode_outline_single(
    ctx: context.Context, request: pb.RegenerateEpisodeOutlineSingleByFictionReq
):
    story_info, generate_input = parse_story_info(ctx, request), parse_generate_input(ctx, request)

    write_scheduler = ProgressScheduler(ctx)

    episode_outline = await write_scheduler.regenerate_episode_outline_single(
        story_info,
        generate_input,
        request.episode_outline,
        request.season_id,
        request.episode_id,
        request.suggestion
    )

    return pb.RegenerateEpisodeOutlineSingleRsp(
        episode_outline=episode_outline.content
    )