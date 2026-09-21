import os
import json
import asyncio
from trpc import context
from trpc.log import logger
from trpc_script_drama_operator import pb

from service.shortdrama_by_fiction.data_model import WorldBuilding,\
    EpisodeOutline_Season, ShortDramaInfo

from service.shortdrama_by_fiction.tools.utils import get_save_drama_info
from service.shortdrama_by_fiction.core.data_manager import DataManager
from service.shortdrama_by_fiction.core.analyze_processor import ShortAnalyze
from service.shortdrama_by_fiction.data_model import EpisodeOutline

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".cache")

async def upload_episode_outlines(
    data_manager: DataManager, story_id, episode_outlines
):
    for _outline in episode_outlines:
        epi_outline = EpisodeOutline(**_outline)
        await data_manager.upload(
            epi_outline,
            story_id=story_id,
            season_id=epi_outline.season_id,
            episode_id=epi_outline.episode_id,
        )


async def generate_episode_outline(
    ctx: context.Context, request: pb.GenerateEpisodeOutlineByFictionReq
):
    story_id = request.story_info.story_id

    novel_id = request.generate_input.novel_id
    season_nums = request.generate_input.common.season_nums
    episode_cnt = request.generate_input.common.episode_nums
    # StoryOutline, story_outline: list[str]
    story_outlines = request.generated_data.story_outline.story_outline
    role_infos = request.generated_data.story_outline.role_info
    new_story_id = f"{novel_id}_{story_id}"
    # 检查参数 # season_nums != 1
    if (
        len(story_outlines) != season_nums or len(story_outlines) != len(role_infos)
    ): 
        logger.error_context(
            ctx, "story_outlines and role_infos length not equal season_nums"
        )
        raise ValueError("story_outlines\role_infos length not equal season_nums")

    analyze = ShortAnalyze(ctx, novel_id, new_story_id)
    data_manager = DataManager(ctx)

    episode_outlines = []
    #season_idx = 0
    #if True:
    for season_idx in range(1):
    
        role_info = role_infos[season_idx]
        story_outline = story_outlines[season_idx]
        #logger.info_context(ctx, role_info[:30], story_outline[:40], new_story_id, season_idx+1)
        logger.info_context(ctx, f"SE{season_idx} story outline:\n {story_outline}")
        try:
            short_drama_info = await data_manager.load_shortdrama_info(new_story_id, season_idx) 
        except Exception as e:
            logger.error_context(ctx, f"load storyinfo error: {e}")
            short_drama_info = None
        world_building_str = await analyze.generate_world_building()
        world_building = WorldBuilding(content=world_building_str)
        world_building_url = await data_manager.upload(
            world_building, story_id=new_story_id, season_id=season_idx
        )
        logger.info_context(
            ctx, f"Generated world building: {world_building_url}\n{world_building}"
        )
        
        #analyze.restore_environment(story_outline, role_info, world_building)
        #analyze.ip_part_split()
        # brief_outlines,part_outlines,eps_ids
        brief_outlines, short_drama_info = await analyze.generate_episode_outlines(short_drama_info,
            episode_cnt, role_info, world_building_str
        )
        #message Season 
        season_outline = {
            "season_id": season_idx,
            "episodes": brief_outlines,
        }
        print("getting season outline",season_outline)
        
        season_outline_info = EpisodeOutline_Season(episodes=brief_outlines, season_id=season_idx)
        season_outline_url = await data_manager.upload(season_outline_info, story_id=new_story_id,season_id=season_idx)
        logger.info_context(ctx, f"Generated se{season_idx} episode_outline uploaded \
                            to COS: {season_outline_url}")

        # 使用新的序列化方法
        #short_drama_info = ipdata_dict.to_short_drama_info()
        """ipdata_info = ShortDramaInfo(
            fname=ipdata_dict.get('fname', ''),
            outline=ipdata_dict.get('outline', {}),
            heros_str=ipdata_dict.get('heros_str', ''),
            world_str=ipdata_dict.get('world_str', ''),
            chapters=ipdata_dict.get('chapters', []),
            chapters_summary=ipdata_dict.get('chapters_summary', []),
            mainline=ipdata_dict.get('mainline', ''),
            subline=ipdata_dict.get('subline', ''),
            outline_plan_str=ipdata_dict.get('outline_plan_str', '')
        )"""
        ipdata_info = get_save_drama_info(short_drama_info)
        print("uploading:short ",short_drama_info)
        #print("uploading:short info",short_drama_info.keys())

        ipdatainfo_url = await data_manager.upload(ipdata_info, story_id=new_story_id)
        logger.info_context(ctx, f"Generated story_outline uploaded \
                            to COS: {ipdatainfo_url}")

        episode_outlines.append(season_outline)
    result = pb.EpisodeOutline(seasons=episode_outlines)

    return pb.GenerateEpisodeOutlineRsp(episode_outline=result)


async def regenerate_episode_outline_all(
    ctx: context.Context, request: pb.RegenerateEpisodeOutlineAllByFictionReq
):

    story_id = request.story_info.story_id
    novel_id = request.generate_input.novel_id
    # 用于存数据
    new_story_id = f"{novel_id}_{story_id}"
    #new_story_id = f"{novel_id}"
    data_manager = DataManager(ctx)
    generator = ShortAnalyze(ctx, novel_id, new_story_id)
    season_outlines = request.generated_data.episode_outline.seasons
    suggestion = request.regenerate_data.suggestion

    episode_outlines = []
    #for idx in range(len(season_outlines)):
    for idx in range(1):
        season_id = season_outlines[idx].season_id
        epi_outlines = season_outlines[idx].episodes

        role_info = await data_manager.load_role_info(new_story_id, season_id)
        world_building = await data_manager.load_world_building(new_story_id, season_id)
        #part_summary = await data_manager.load_shortdrama_plan(story_id, season_id)
        short_drama_info = await data_manager.load_shortdrama_info(new_story_id, season_id)
        refine_outlines, short_drama_info = await generator.regenerate_episode_outline(short_drama_info,
            role_info, world_building, epi_outlines, suggestion
        )
        #await upload_episode_outlines(data_manager, new_story_id, [refine_outlines])
        season_outline = {
            "season_id": season_id,
            "episodes": refine_outlines,
        }
        episode_outlines.append(season_outline)

        season_outline_info = EpisodeOutline_Season(episodes=refine_outlines, season_id=season_id)
        season_outline_url = await data_manager.upload(season_outline_info, story_id=new_story_id, season_id=season_id)
        logger.info_context(ctx, f"Generated se{season_id} episode_outline uploaded \
                            to COS: {season_outline_url}")
        
        
        # 使用新的序列化方法
        #short_drama_info = ipdata_dict.to_short_drama_info()
        ipdata_info = get_save_drama_info(short_drama_info)
        """ipdata_info = ShortDramaInfo(
            fname=ipdata_dict.get('fname', ''),
            outline=ipdata_dict.get('outline', {}),
            heros_str=ipdata_dict.get('heros_str', ''),
            world_str=ipdata_dict.get('world_str', ''),
            chapters=ipdata_dict.get('chapters', []),
            chapters_summary=ipdata_dict.get('chapters_summary', []),
            mainline=ipdata_dict.get('mainline', ''),
            subline=ipdata_dict.get('subline', ''),
            outline_plan_str=ipdata_dict.get('outline_plan_str', '')
        )"""
        ipdatainfo_url = await data_manager.upload(ipdata_info, story_id=new_story_id)
        logger.info_context(ctx, f"RE-Generated story_outline uploaded \
                            to COS: {ipdatainfo_url}")

    result = pb.EpisodeOutline(seasons=episode_outlines)
    
    return pb.GenerateEpisodeOutlineRsp(episode_outline=result)


async def regenerate_episode_outline_single(
    ctx: context.Context, request: pb.RegenerateEpisodeOutlineSingleByFictionReq
):


    story_id = request.story_info.story_id

    novel_id = request.generate_input.novel_id
    season_id = request.season_id
    episode_id = request.episode_id
    # 用于存数据
    new_story_id = f"{novel_id}_{story_id}"

    outline = request.episode_outline
    start = request.chapter_from
    end = request.chapter_to
    suggestion = request.suggestion
    data_manager = DataManager(ctx)
    generator = ShortAnalyze(ctx, novel_id, new_story_id)
    print("input regenerate single: season_id! then make it 0", season_id)
    season_id = 0
    if len(outline.strip()) <= 0:
        logger.warning_context(
            ctx, "regenerate_episode_outline_single outline is empty"
        )

    if len(suggestion.strip()) <= 0:
        logger.warning_context(
            ctx, "regenerate_episode_outline_single suggestion is empty"
        )

    role_info = await data_manager.load_role_info(new_story_id, season_id)
    world_building = await data_manager.load_world_building(new_story_id, season_id)
    #contents = await data_manager.load_chapter_contents(novel_id, start, end)
    ipdata_dict = await data_manager.load_shortdrama_info(new_story_id, season_id)
    if ipdata_dict:
        generator.restore_geninfo(ipdata_dict)
    season_outline, ipdata_dict = await generator.regenerate_episode_outline(
        ipdata_dict, role_info, world_building, outline, suggestion
    )

    season_outline_info = EpisodeOutline_Season(episodes=season_outline, season_id=season_id)
    season_outline_url = await data_manager.upload(season_outline_info, story_id=new_story_id, season_id=season_id)
    logger.info_context(ctx, f"Generated se{season_id} episode_outline uploaded \
                            to COS: {season_outline_url}")
    # 使用新的序列化方法
    #short_drama_info = ipdata_dict.to_short_drama_info()
    ipdata_info = get_save_drama_info(ipdata_dict)
    """ipdata_info = ShortDramaInfo(
        fname=ipdata_dict.get('fname', ''),
        outline=ipdata_dict.get('outline', {}),
        heros_str=ipdata_dict.get('heros_str', ''),
        world_str=ipdata_dict.get('world_str', ''),
        chapters=ipdata_dict.get('chapters', []),
        chapters_summary=ipdata_dict.get('chapters_summary', []),
        mainline=ipdata_dict.get('mainline', ''),
        subline=ipdata_dict.get('subline', ''),
        outline_plan_str=ipdata_dict.get('outline_plan_str', '')
    )"""
    ipdatainfo_url = await data_manager.upload(ipdata_info, new_story_id, season_id=season_id)
    logger.info_context(ctx, f"RE-Generated story_outline uploaded \
                        to COS: {ipdatainfo_url}")


    #await upload_episode_outlines(data_manager, new_story_id, [season_outline])
    return pb.RegenerateEpisodeOutlineSingleRsp(episode_outline=season_outline)