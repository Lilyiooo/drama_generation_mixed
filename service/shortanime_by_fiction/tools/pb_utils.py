from google.protobuf.json_format import MessageToDict
from trpc_script_drama_operator import rpc, pb
from trpc.log import logger
from google.protobuf import json_format

from service.shortanime_by_fiction.data_models import (
    StoryInfo,
    GenerateInput,
    SeasonProposal,
    EpisodeOutline,
    EpisodeScript
)

def message_to_dict(message, str2int_convert=True):
    
    data = MessageToDict(
        message,
        preserving_proto_field_name=True,
        use_integers_for_enums=True,
        including_default_value_fields=True
    )

    def convert_string_to_int(obj):
        if isinstance(obj, dict):
            return {k: convert_string_to_int(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_string_to_int(item) for item in obj]
        elif isinstance(obj, str) and ( obj.isdigit() or obj == '-1'):
            return int(obj)
        return obj

    if str2int_convert:
        data = convert_string_to_int(data)
    return data

def parse_story_info(ctx, request):
    data_dict = MessageToDict(request.story_info, preserving_proto_field_name=True)
    story_info = StoryInfo(**data_dict)
    return story_info

def parse_generate_input(ctx, request):
    generate_input = request.generate_input
    generate_input = GenerateInput(
        novel_id=generate_input.novel_id,
        season_nums=generate_input.common.season_nums,
        episode_nums=generate_input.common.episode_nums,
        min_word_count_per_episode=generate_input.common.min_word_count_per_episode,
        max_word_count_per_episode=generate_input.common.max_word_count_per_episode,
        instruction=generate_input.common.adaptation_approach,
    )
    return generate_input

def parse_proposals(ctx, script_proposal):
    data_dict = message_to_dict(script_proposal)
    proposals = []
    for proposal in data_dict["script_proposals"]:
        for plot in proposal['plot_planning']:
            plot['status'] = '保留' if plot.get('reserve', 0) == 1 else '删除'
        if 'storylines' in proposal:
            for story_line in proposal['storylines']:
                story_line['status'] = {0: "保留", 1: "保留", 2: "精简", 3: "删除"}.get(story_line.get('reserve', 0), "保留")

        season_proposal = SeasonProposal.from_dict(proposal)
        proposals.append(season_proposal)
    
    # 后处理：处理 season_division
    last_end_plot_id = -1  # 首季从0开始，所以初始值为-1
    
    season_division = proposals[0]
    if hasattr(season_division, 'point_planning') and season_division.point_planning:
        for point in season_division.point_planning.points:
            point.start_plot_id = last_end_plot_id + 1
            last_end_plot_id = point.end_plot_id

    # start_plot_id = 0
    for season_proposal in proposals[1:]:
        if hasattr(season_proposal, 'point_planning') and season_proposal.point_planning:
            for point in season_proposal.point_planning.points:
                point.start_plot_id = 0
    
    return proposals

def parse_episode_outlines(ctx, episode_outline, season_nums):
    episode_outlines = []
    data_dict = message_to_dict(episode_outline)['seasons']

    if len(data_dict) != season_nums:
        logger.error_context(ctx, f"season_nums: {season_nums}, len of data_dict: {len(data_dict)}")
        raise ValueError(f"Episode outline length invalid: expected {season_nums}, got {len(data_dict)}")

    for season_id in range(season_nums):
        season_episode_outline = []
        for episode_outline in data_dict[season_id]['episodes']:
            season_episode_outline.append(EpisodeOutline.from_dict(episode_outline))
        episode_outlines.append(season_episode_outline)
    return episode_outlines


def parse_drama(ctx, drama, season_nums):
    dramas = []
    data_dict = message_to_dict(drama)

    seasons = data_dict.get('seasons', [])
    if len(seasons) != season_nums:
        logger.error_context(ctx, f"season_nums: {season_nums}, len of seasons: {len(seasons)}")
        raise ValueError(f"Drama seasons length invalid: expected {season_nums}, got {len(seasons)}")

    for season_id in range(season_nums):
        season_scripts = []
        for episode in seasons[season_id].get('episodes', []):
            season_scripts.append(EpisodeScript.from_dict(episode))
        dramas.append(season_scripts)
    return dramas

def parse_story_outline(ctx, story_outline, season_nums):

    world_buildings = story_outline.world_building
    role_infos = story_outline.role_info
    story_outlines = story_outline.story_outline

    if len(story_outlines) != season_nums:
        logger.error_context(ctx, f"story_outlines length: {len(story_outlines)}")
        raise ValueError(f"story_outlines length invalid, expected {season_nums}, got {len(story_outlines)}")

    if len(world_buildings) < 1:
        logger.error_context(ctx, f"world_buildings length: {len(world_buildings)}")
        raise ValueError(f"world_buildings length invalid, expected {season_nums}, got {len(world_buildings)}")

    if len(role_infos) < 1:
        logger.error_context(ctx, f"role_infos length: {len(role_infos)}")
        raise ValueError(f"role_infos length invalid, expected {season_nums}, got {len(role_infos)}")
    
    
    return world_buildings, role_infos, story_outlines

def convert_to_script_proposal(
    proposals
):
    script_proposal = pb.ScriptProposal()
    proposals = [proposal.to_dict() for proposal in proposals]

    for season_proposal in proposals:
        for plot in season_proposal['plot_planning']:
            plot["reserve"] = True if plot["status"] == "保留" else False
    for season_proposal in proposals:
        for storyline in season_proposal['storylines']:
            storyline["reserve"] = {"保留": 1, "精简": 2, "删除": 3}.get(storyline['status'], 1)

    json_format.ParseDict({"script_proposals": proposals}, script_proposal, ignore_unknown_fields=True)
    return script_proposal

def convert_to_drama(
    scripts
):
    # 转换数据格式，只保留 pb.Episode 识别的字段，排除 word_tier/tier_reason 等扩展字段
    converted_scripts = []
    for season in scripts:
        converted_season = season.copy()
        if 'episodes' in converted_season:
            episodes = []
            for script in season['episodes']:
                if hasattr(script, 'to_dict'):
                    episodes.append({
                        "season_id": script.season_id,
                        "episode_id": script.episode_id,
                        "script_content": script.script_content,
                    })
                else:
                    episodes.append(script)
            converted_season['episodes'] = episodes
        converted_scripts.append(converted_season)

    return pb.Drama(seasons=converted_scripts)

def convert_to_episode_outline(
    episode_outlines
):
    converted_outlines = []
    for season in episode_outlines:
        season_dict = {'season_id': season['season_id'], 'episodes': []}
        for episode_outline in season['episodes']:
            converted_outline = {
                "season_id": episode_outline.season_id,
                "episode_id": episode_outline.episode_id,
                "chapter_from": episode_outline.chapter_from,
                "chapter_to": episode_outline.chapter_to,
                "content": episode_outline.content
            }
            season_dict['episodes'].append(converted_outline)
            
        converted_outlines.append(season_dict)
    
    return pb.EpisodeOutline(seasons=converted_outlines)

