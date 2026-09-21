from trpc import context
from trpc.log import logger
from trpc_script_drama_operator import pb

from service.drama_by_fiction.core.scene_planner import ScenePlanner

from service.drama_by_fiction.utils.pb_utils import (
    check_proposals,
    check_episode_outline_content,
    parse_story_info,
    parse_generate_input,
    parse_proposals,
    parse_story_outline,
    parse_episode_outlines,
    get_single_episode_outline,
    convert_to_episode_outline,
)


async def generate_episode_outlines(
    ctx: context.Context, request: pb.GenerateEpisodeOutlineByFictionReq
) -> pb.GenerateEpisodeOutlineRsp:
    story_info = parse_story_info(ctx, request)
    generate_input = parse_generate_input(ctx, request)
    proposals = parse_proposals(ctx, request.generated_data.script_proposal)

    world_buildings, role_infos, _ = parse_story_outline(
        ctx, request.generated_data.story_outline, generate_input.season_nums
    )

    suggestion = request.generate_data.suggestion or request.generate_input.suggestion
    check_proposals(ctx, story_info, generate_input, proposals)

    scene_planner = ScenePlanner(ctx)

    outline = []
    for _range in request.generate_data.select_range:
        try:
            season_id, episode_ids = _range.season_id, _range.episode_ids
            season_proposal = proposals[season_id + 1]

            episode_outlines = await scene_planner.generate_episode_outlines(
                story_info,
                generate_input,
                season_id,
                episode_ids,
                world_buildings[0],
                role_infos[0],
                season_proposal,
                suggestion=suggestion,
            )
        except Exception as e:
            logger.error(f"generate_episode_outlines failed: {e}")
            raise

        outline.append({"season_id": season_id, "episodes": episode_outlines})
    outline = convert_to_episode_outline(outline)
    return pb.GenerateEpisodeOutlineRsp(episode_outline=outline)


async def generate_episode_scene_plan(
    ctx: context.Context, request: pb.GenerateEpisodeOutlineByFictionReq
) -> pb.GenerateEpisodeOutlineRsp:
    """
    按 select_range 生成指定季-集的分场规划
    """
    story_info = parse_story_info(ctx, request)
    generate_input = parse_generate_input(ctx, request)
    proposals = parse_proposals(ctx, request.generated_data.script_proposal)

    check_proposals(ctx, story_info, generate_input, proposals)
    check_episode_outline_content(
        ctx, request.generated_data.episode_outline, request.generate_data.select_range
    )

    world_buildings, role_infos, _ = parse_story_outline(
        ctx, request.generated_data.story_outline, generate_input.season_nums
    )

    episode_outlines = parse_episode_outlines(
        ctx, request.generated_data.episode_outline, generate_input.season_nums
    )

    suggestion = request.generate_data.suggestion or request.generate_input.suggestion

    scene_planner = ScenePlanner(ctx)

    outline = []
    for _range in request.generate_data.select_range:
        try:
            season_id, episode_ids = _range.season_id, _range.episode_ids
            season_proposal = proposals[season_id + 1]

            season_episode_outlines = episode_outlines[season_id]

            new_outlines = await scene_planner.generate_scene_plans(
                story_info,
                generate_input,
                season_id,
                episode_ids,
                world_buildings[0],
                role_infos[0],
                season_proposal,
                season_episode_outlines,
                suggestion=suggestion,
            )
        except Exception as e:
            logger.error(f"generate_episode_scene_plan failed: {e}")
            raise

        outline.append({"season_id": season_id, "episodes": new_outlines})
    outline = convert_to_episode_outline(outline)
    return pb.GenerateEpisodeOutlineRsp(episode_outline=outline)


async def generate_scene_outline(
    ctx: context.Context, request: pb.GenerateSceneOutlineByFictionReq
) -> pb.GenerateSceneOutlineRsp:
    story_info = parse_story_info(ctx, request)
    generate_input = parse_generate_input(ctx, request)
    proposals = parse_proposals(ctx, request.generated_data.script_proposal)

    world_buildings, role_infos, _ = parse_story_outline(
        ctx, request.generated_data.story_outline, generate_input.season_nums
    )

    season_id = request.generate_data.season_id
    episode_id = request.generate_data.episode_id
    scene_id = request.generate_data.scene_id
    suggestion = request.generate_data.suggestion or request.generate_input.suggestion
    episode_outline = get_single_episode_outline(
        ctx, request.generated_data.episode_outline, season_id, episode_id
    )

    scene_planner = ScenePlanner(ctx)

    check_proposals(ctx, story_info, generate_input, proposals)

    scene_plan = await scene_planner.generate_single_scene_outline(
        story_info,
        generate_input,
        season_id,
        episode_id,
        scene_id,
        world_buildings[0],
        role_infos[0],
        proposals[season_id + 1],
        episode_outline,
        suggestion=suggestion,
    )
    return pb.GenerateSceneOutlineRsp(scene_outline=scene_plan.to_dict())


async def generate_scene_outline_prompt(
    ctx: context.Context, request: pb.GenerateSceneOutlineByFictionReq
) -> pb.GeneratePromptRsp:
    """返回生成单场大纲时拼装好的完整提示词，不调用 LLM。"""
    story_info = parse_story_info(ctx, request)
    generate_input = parse_generate_input(ctx, request)
    proposals = parse_proposals(ctx, request.generated_data.script_proposal)

    world_buildings, role_infos, _ = parse_story_outline(
        ctx, request.generated_data.story_outline, generate_input.season_nums
    )

    season_id = request.generate_data.season_id
    episode_id = request.generate_data.episode_id
    scene_id = request.generate_data.scene_id
    suggestion = request.generate_data.suggestion or request.generate_input.suggestion
    episode_outline = get_single_episode_outline(
        ctx, request.generated_data.episode_outline, season_id, episode_id
    )

    check_proposals(ctx, story_info, generate_input, proposals)

    scene_planner = ScenePlanner(ctx)
    prompt = await scene_planner.build_single_scene_outline_prompt(
        story_info,
        generate_input,
        season_id,
        episode_id,
        scene_id,
        world_buildings[0],
        role_infos[0],
        proposals[season_id + 1],
        episode_outline,
        suggestion=suggestion,
    )
    return pb.GeneratePromptRsp(prompt=prompt)
