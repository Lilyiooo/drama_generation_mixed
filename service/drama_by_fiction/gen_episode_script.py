from trpc import context
from trpc.log import logger
from trpc_script_drama_operator import pb

from service.drama_by_fiction.core.script_writer import ScriptWriter
from service.drama_by_fiction.utils.pb_utils import (
    check_proposals,
    check_episode_outline_scenes,
    parse_story_info,
    parse_generate_input,
    parse_proposals,
    parse_story_outline,
    parse_episode_outlines,
    get_single_episode_outline,
    get_single_episode_script,
    convert_to_drama,
)


async def generate_scripts(ctx: context.Context, request: pb.GenerateDramaByFictionReq):

    story_info = parse_story_info(ctx, request)
    generate_input = parse_generate_input(ctx, request)
    proposals = parse_proposals(ctx, request.generate_data.script_proposal)

    check_proposals(ctx, story_info, generate_input, proposals)
    check_episode_outline_scenes(
        ctx, request.generate_data.episode_outline, request.generate_data.select_range
    )

    world_buildings, role_infos, _ = parse_story_outline(
        ctx, request.generate_data.story_outline, generate_input.season_nums
    )
    episode_outlines = parse_episode_outlines(
        ctx, request.generate_data.episode_outline, generate_input.season_nums
    )
    suggestion = request.generate_data.suggestion or request.generate_input.suggestion
    script_writer = ScriptWriter(ctx)

    drama = []
    for _range in request.generate_data.select_range:

        try:
            season_id, episode_ids = _range.season_id, _range.episode_ids
            season_proposal = proposals[season_id + 1]

            season_episode_outlines = episode_outlines[season_id]

            scripts = await script_writer.generate_episode_scripts(
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
            logger.error(f"generate_scripts failed: {e}")
            raise

        drama.append({"season_id": season_id, "episodes": scripts})

    drama = convert_to_drama(drama)
    outline = request.generate_data.episode_outline

    return pb.GenerateDramaRsp(result=drama, episode_outline=outline)


async def generate_scene_script(
    ctx: context.Context, request: pb.GenerateSceneDramaByFictionReq
) -> pb.GenerateSceneDramaRsp:
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
    existing_episode_script = get_single_episode_script(
        ctx, request.generated_data.drama, season_id, episode_id
    )

    script_writer = ScriptWriter(ctx)

    check_proposals(ctx, story_info, generate_input, proposals)

    scene_drama = await script_writer.generate_single_scene_script(
        story_info,
        generate_input,
        season_id,
        episode_id,
        scene_id,
        world_buildings[0],
        role_infos[0],
        proposals[season_id + 1],
        episode_outline,
        existing_episode_script,
        suggestion=suggestion,
    )
    return pb.GenerateSceneDramaRsp(scene_drama=scene_drama.to_dict())


async def generate_scene_script_prompt(
    ctx: context.Context, request: pb.GenerateSceneDramaByFictionReq
) -> pb.GeneratePromptRsp:
    """返回生成单场剧本时拼装好的完整提示词，不调用 LLM。"""
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
    existing_episode_script = get_single_episode_script(
        ctx, request.generated_data.drama, season_id, episode_id
    )

    check_proposals(ctx, story_info, generate_input, proposals)

    script_writer = ScriptWriter(ctx)
    prompt = await script_writer.build_single_scene_script_prompt(
        story_info,
        generate_input,
        season_id,
        episode_id,
        scene_id,
        world_buildings[0],
        role_infos[0],
        proposals[season_id + 1],
        episode_outline,
        existing_episode_script,
        suggestion=suggestion,
    )
    return pb.GeneratePromptRsp(prompt=prompt)
