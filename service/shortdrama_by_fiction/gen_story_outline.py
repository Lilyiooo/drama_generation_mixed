import os
import json
from trpc import context
from trpc.log import logger
from trpc_script_drama_operator import pb
from google.protobuf.json_format import MessageToDict

from service.shortdrama_by_fiction.tools.utils import get_save_drama_info 

from service.shortdrama_by_fiction.data_model import ShortDramaInfo, \
    EpisodeOutline, RoleInfo, ShortDramaPartPlan

from service.shortdrama_by_fiction.core.data_manager import DataManager
from service.shortdrama_by_fiction.core.analyze_processor import ShortAnalyze

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".cache")



async def generate_story_outlines_and_role_infos(
    ctx: context.Context, request: pb.GenerateStoryOutlineByFictionReq
):
    story_id = request.story_info.story_id

    novel_id = request.generate_input.novel_id
    
    season_nums = request.generate_input.common.season_nums
    episode_nums = request.generate_input.common.episode_nums

    #duration = request.generate_input.common.duration
    # 不检查 季 输入，默认只有1季
    if not ( novel_id and episode_nums):
        logger.error_context(
            ctx, f"参数不完整，无法生成故事大纲。generate_input:\n {request.generate_input} \
            story_id: {request.story_info.story_id}, 检查novel_id、episode_nums"
        )
        return [], []
    # 用于存数据
    new_story_id = f"{novel_id}_{story_id}"
    data_manager = DataManager(ctx)
    analyze = ShortAnalyze(ctx, novel_id, new_story_id)
    
    
    role_info = await data_manager.load_role_info(new_story_id, season_id=0)
    print("loading:",role_info)
    #role_info = role_info.content
    #role_info = None
    if len(role_info)<10:
        role_info, short_drama_info = await analyze.generate_heros()
        #print(f"-----genheros_str: {genheros_str}")
        role_info_pb = RoleInfo(content=role_info)
        role_info_url = await data_manager.upload(
            role_info_pb, story_id=new_story_id
        )
        logger.info_context(ctx, f"upload role info: {role_info_url}")
    else:
        analyze.set_role_info(role_info) 

    #story_outline = await data_manager.load_shortdrama_plan(new_story_id)
    story_outline = None 
    if not story_outline:
        logger.info_context(
            ctx, f"shortdrama_outline not found for novel_id: {new_story_id}, start generating..."
        )
        story_outline, short_drama_info = await analyze.generate_story_outline(episode_nums)
        #print("story_outline type:", type(story_outline), "short_drama_info keys:", short_drama_info.keys())
        story_outline_info = ShortDramaPartPlan(content=story_outline)
        story_outline_info_url = await data_manager.upload(
            story_outline_info, story_id=new_story_id
        )
        logger.info_context(ctx, f"upload story outline info: {story_outline_info_url}") 

    else:
        analyze.set_story_outline(story_outline)


    # 使用新的序列化方法
    #short_drama_info = ipdata_dict.to_short_drama_info()
    ipdata_info = get_save_drama_info(short_drama_info)
    """
    ipdata_info = ShortDramaInfo(
        fname=short_drama_info.get('fname', ''),
        outline=short_drama_info.get('outline', {}),
        heros_str=short_drama_info.get('heros_str', ''),
        world_str=short_drama_info.get('world_str', ''),
        chapters=short_drama_info.get('chapters', []),
        chapters_summary=short_drama_info.get('chapters_summary', []),
        mainline=short_drama_info.get('mainline', ''),
        subline=short_drama_info.get('subline', ''),
        outline_plan_str=short_drama_info.get('outline_plan_str', '')
    )"""
    #print("before upload ipdata_info",ipdata_info.keys())
    ipdatainfo_url = await data_manager.upload(ipdata_info, story_id=new_story_id)
    logger.info_context(ctx, f"Generated ipdata_info uploaded to COS: {ipdatainfo_url}")
        

    result = pb.StoryOutline(story_outline=[story_outline], role_info=[role_info])
    return pb.GenerateStoryOutlineRsp(story_outline=result)


async def regenerate_story_outline(
    ctx: context.Context, request: pb.RegenerateStoryOutlineByFictionReq
):

    story_id = request.story_info.story_id
    novel_id = request.generate_input.novel_id
    season_id = request.regenerate_data.season_id
    #raw_outlines = request.regenerate_data.story_outline.story_outline
    # 用于存数据
    new_story_id = f"{novel_id}_{story_id}"

    data_manager = DataManager(ctx)
    analyze = ShortAnalyze(ctx, novel_id, new_story_id)
    role_info = request.regenerate_data.generate_data.story_outline.role_info[season_id]
    outline = request.regenerate_data.generate_data.story_outline.story_outline[season_id]
    suggestion = request.regenerate_data.suggestion

    storyinfo = await data_manager.load_shortdrama_info(new_story_id, season_id) 
    # ShortDramaInfo对象直接使用content属性
    story_outline, short_drama_info = await analyze.regenerate_story_outline(storyinfo, outline, suggestion, role_info)

    # 使用新的序列化方法
    #short_drama_info = ipdata_dict.to_short_drama_info()
    ipdata_info = get_save_drama_info(short_drama_info)
    """
    shortdrama_info = ShortDramaInfo(
        fname=short_drama_info.get('fname', ''),
        outline=short_drama_info.get('outline', {}),
        heros_str=short_drama_info.get('heros_str', ''),
        world_str=short_drama_info.get('world_str', ''),
        mainline=short_drama_info.get('mainline', ''),
        subline=short_drama_info.get('subline', ''),
        outline_plan_str=short_drama_info.get('outline_plan_str', ''),
        outline_plan=short_drama_info.get('outline_plan', {}),
        num_episode=short_drama_info.get('num_episode', 0)
    )"""
    #chapters=short_drama_info.get('chapters', []),
    #chapters_summary=short_drama_info.get('chapters_summary', []),
    storyinfo_url = await data_manager.upload(ipdata_info, story_id=new_story_id, season_id=season_id)
    logger.info_context(ctx, f"RE-Generated story_outline uploaded to COS: {storyinfo_url}")

    return pb.RegenerateStoryOutlineRsp(result=story_outline)



async def regenerate_role_info(
    ctx: context.Context, request: pb.RegenerateStoryOutlineByFictionReq
):

    story_id = request.story_info.story_id
    novel_id = request.generate_input.novel_id
    season_id = request.regenerate_data.season_id
    role_info = request.regenerate_data.generate_data.story_outline.role_info[season_id]
    outline = request.regenerate_data.generate_data.story_outline.story_outline[season_id]
    suggestion = request.regenerate_data.suggestion

    # 用于存数据
    new_story_id = f"{novel_id}_{story_id}"

    short_drama_info = await data_manager.load_shortdrama_info(story_id, season_id) 
    # ShortDramaInfo对象直接使用content属性
    data_manager = DataManager(ctx)
    analyze = ShortAnalyze(ctx, novel_id, new_story_id)

    new_role_info_str, short_drama_info = await analyze.regenerate_heros(
        short_drama_info, role_info, outline, suggestion
    )

    role_info_url = await data_manager.upload(
        RoleInfo(content=new_role_info_str), story_id=new_story_id, season_id=season_id
    )
    logger.info_context(ctx, f"upload role info: {role_info_url}")

    ipdata_info = get_save_drama_info(short_drama_info)
    storyinfo_url = await data_manager.upload(ipdata_info, story_id=new_story_id, season_id=season_id)
    logger.info_context(ctx, f"RE-Generated story_outline uploaded to COS: {storyinfo_url}")

    return pb.RegenerateStoryOutlineRsp(result=role_info_url)
