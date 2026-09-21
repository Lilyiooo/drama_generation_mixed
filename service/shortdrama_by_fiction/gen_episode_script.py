import os
import json
from trpc import context
from trpc.log import logger
from trpc_script_drama_operator import pb
from service.shortdrama_by_fiction.tools.utils import get_save_drama_info 
from service.shortdrama_by_fiction.core.data_manager import DataManager
from service.shortdrama_by_fiction.core.analyze_processor import ShortAnalyze
from service.shortdrama_by_fiction.data_model import ShortDramaInfo

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".cache")

# diff input
async def generate_episode_scripts(
    ctx: context.Context, request: pb.GenerateDramaByFictionReq
):  
    
    story_id = request.story_info.story_id
    novel_id = request.generate_input.novel_id
    # 用于存数据
    new_story_id = f"{novel_id}_{story_id}"
    #new_story_id = f"{novel_id}"
    data_manager = DataManager(ctx)
    analyze = ShortAnalyze(ctx, novel_id, new_story_id)
    # add story_outlines
    #story_outline = request.generate_data.story_outline.story_outlines
    #role_infos = request.generate_data.story_outline.role_info
    #seasons = request.generate_data.episode_outline.seasons
    select_ranges = request.generate_data.select_range
    #suggestion = request.generate_data.suggestion
    print("generate scripts: select_ranges", select_ranges) 
    total_scripts = []

    #for select_range in select_ranges:
    for runidx in range(1):
        #season_id = select_range.season_id
        #episode_ids = select_range.episode_ids
        season_id = 0
        #epi_outlines = seasons[season_id - 1].episodes
        #role_info = role_infos[season_id - 1]

        role_info = await data_manager.load_role_info(new_story_id, season_id)
        world_building = await data_manager.load_world_building(new_story_id, season_id)
        ipdata_dict = await data_manager.load_shortdrama_info(new_story_id, season_id)
        
        #refine_part_summary = await data_manager.load_shortdrama_plan(
        #   story_id, season_id)
        #analyze.restore_environment(story_outline, role_info, world_building)
        #analyze.restore_plan(refine_part_summary)

        results, ipdata_dict = await analyze.ip_part_script_gen(ipdata_dict, role_info, world_building)


        season_scripts = {
            "season_id": season_id,
            "episodes": results,
        }
        total_scripts.append(season_scripts)

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
        ipdata_info = get_save_drama_info(ipdata_dict)
        ipdatainfo_url = await data_manager.upload(ipdata_info, story_id=new_story_id)
        logger.info_context(ctx, f"Generated epsode scripts uploaded \
                            to COS: {ipdatainfo_url}")

    results = pb.Drama(seasons=total_scripts)
    return pb.GenerateDramaRsp(result=results)