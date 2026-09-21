from google.protobuf.json_format import MessageToDict
from trpc_script_drama_operator import pb

from trpc.log import logger
from trpc.exceptions import NewBusinessError

from service.drama_by_fiction.error_codes import BizCode, BIZ_MSG

from google.protobuf import json_format

from service.drama_by_fiction.data_model import (
    StoryInfo,
    GenerateInput,
    SeasonProposal,
    EpisodeOutline,
    EpisodeScript,
)


def message_to_dict(message, str2int_convert=True):

    data = MessageToDict(
        message,
        preserving_proto_field_name=True,
        use_integers_for_enums=True,
        including_default_value_fields=True,
    )

    def convert_string_to_int(obj):
        if isinstance(obj, dict):
            return {k: convert_string_to_int(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_string_to_int(item) for item in obj]
        elif isinstance(obj, str) and (obj.isdigit() or obj == "-1"):
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
        duration=generate_input.common.duration,
        min_word_count_per_episode=generate_input.common.min_word_count_per_episode,
        max_word_count_per_episode=generate_input.common.max_word_count_per_episode,
        instruction=generate_input.common.adaptation_approach,
    )
    return generate_input


def parse_proposals(ctx, script_proposal):
    data_dict = message_to_dict(script_proposal)
    proposals = []
    for proposal in data_dict["script_proposals"]:
        for plot in proposal["plot_planning"]:
            plot["status"] = "保留" if plot.get("reserve", 0) == 1 else "删除"

        STATUS_MAP = {1: "保留", 2: "精简", 3: "删除"}
        if "storylines" in proposal:
            for story_line in proposal["storylines"]:
                story_line["status"] = STATUS_MAP.get(
                    story_line.get("reserve", 0), "保留"
                )

        season_proposal = SeasonProposal.from_dict(proposal)
        proposals.append(season_proposal)

    # 后处理：处理 season_division
    last_end_plot_id = -1  # 首季从0开始，所以初始值为-1

    season_division = proposals[0]
    if hasattr(season_division, "point_planning") and season_division.point_planning:
        for point in season_division.point_planning.points:
            point.start_plot_id = last_end_plot_id + 1
            last_end_plot_id = point.end_plot_id

    # start_plot_id固定为0，以end_plot_id作为卡点标记
    for season_proposal in proposals[1:]:
        if (
            hasattr(season_proposal, "point_planning")
            and season_proposal.point_planning
        ):
            for point in season_proposal.point_planning.points:
                point.start_plot_id = 0

    return proposals


def parse_episode_outlines(ctx, episode_outline, season_nums):
    episode_outlines = {idx: [] for idx in range(season_nums)}
    seasons = message_to_dict(episode_outline)["seasons"]
    for season in seasons:
        season_id = season["season_id"]
        episodes = season["episodes"]
        for outline in episodes:
            episode_outlines[season_id].append(EpisodeOutline.from_dict(outline))
    return episode_outlines


def _find_single_episode(ctx, pb_message, season_id, episode_id, model_cls, label):
    """从 proto 的多季多集嵌套结构中，按 season_id + episode_id 查找并转为 dataclass 对象。

    :param pb_message: proto message（EpisodeOutline 或 Drama）
    :param season_id: 目标季 ID
    :param episode_id: 目标集 ID
    :param model_cls: dataclass 类型（EpisodeOutline / EpisodeScript），需有 from_dict
    :param label: 日志标签，如 "分场规划" / "剧本"
    :return: 匹配的 dataclass 对象，未找到则返回空实例
    """
    episode_map = {}
    seasons = message_to_dict(pb_message)["seasons"]
    for season in seasons:
        for episode in season["episodes"]:
            _id = f'{season["season_id"]}_{episode["episode_id"]}'
            episode_map[_id] = model_cls.from_dict(episode)

    this_id = f"{season_id}_{episode_id}"
    if this_id not in episode_map:
        logger.warning_context(ctx, f"未找到{this_id}的{label}，默认为空")
        return model_cls()
    return episode_map[this_id]


def get_single_episode_outline(
    ctx, episode_outline, season_id, episode_id
) -> EpisodeOutline:
    return _find_single_episode(
        ctx, episode_outline, season_id, episode_id, EpisodeOutline, "分场规划"
    )


def get_single_episode_script(ctx, drama, season_id, episode_id) -> EpisodeScript:
    return _find_single_episode(
        ctx, drama, season_id, episode_id, EpisodeScript, "剧本"
    )


def parse_story_outline(ctx, story_outline, season_nums):

    world_buildings = story_outline.world_building
    role_infos = story_outline.role_info
    story_outlines = story_outline.story_outline

    # if len(story_outlines) != season_nums:
    #     logger.error_context(ctx, f"story_outlines length: {len(story_outlines)}")
    #     raise NewBusinessError(
    #         BizCode.PARAM_INVALID,
    #         f"{BIZ_MSG[BizCode.PARAM_INVALID]}: story_outlines length invalid, expected {season_nums}, got {len(story_outlines)}",
    #     )

    if len(world_buildings) < 1:
        logger.error_context(ctx, f"world_buildings length: {len(world_buildings)}")
        raise NewBusinessError(
            BizCode.PARAM_INVALID,
            f"{BIZ_MSG[BizCode.PARAM_INVALID]}: world_buildings length invalid, expected {season_nums}, got {len(world_buildings)}",
        )

    if len(role_infos) < 1:
        logger.error_context(ctx, f"role_infos length: {len(role_infos)}")
        raise NewBusinessError(
            BizCode.PARAM_INVALID,
            f"{BIZ_MSG[BizCode.PARAM_INVALID]}: role_infos length invalid, expected {season_nums}, got {len(role_infos)}",
        )

    return world_buildings, role_infos, story_outlines


def convert_to_script_proposal(proposals):
    script_proposal = pb.ScriptProposal()
    proposals = [proposal.to_dict() for proposal in proposals]

    for season_proposal in proposals:
        for plot in season_proposal["plot_planning"]:
            plot["reserve"] = True if plot["status"] == "保留" else False

    STATUS_MAP = {"保留": 1, "精简": 2, "删除": 3}
    for season_proposal in proposals:
        for storyline in season_proposal["storylines"]:
            storyline["reserve"] = STATUS_MAP.get(storyline["status"], 1)

    json_format.ParseDict(
        {"script_proposals": proposals}, script_proposal, ignore_unknown_fields=True
    )
    return script_proposal


def convert_to_drama(scripts):
    # 转换数据格式
    converted_scripts = []
    for season in scripts:
        converted_season = season.copy()
        if "episodes" in converted_season:
            converted_season["episodes"] = [
                script.to_dict() if hasattr(script, "to_dict") else script
                for script in season["episodes"]
            ]
        converted_scripts.append(converted_season)

    return pb.Drama(seasons=converted_scripts)


def convert_to_episode_outline(episode_outlines):
    converted_outlines = []
    for season in episode_outlines:
        season_dict = {"season_id": season["season_id"], "episodes": []}
        for episode_outline in season["episodes"]:
            # scenes 是 data_model.Scene 对象列表，需要转为 dict 才能被 protobuf 接受
            scenes = [
                scene.to_dict() if hasattr(scene, "to_dict") else scene
                for scene in (episode_outline.scenes or [])
            ]
            converted_outline = {
                "season_id": episode_outline.season_id,
                "episode_id": episode_outline.episode_id,
                "chapter_from": episode_outline.chapter_from,
                "chapter_to": episode_outline.chapter_to,
                "content": episode_outline.content,
                "scenes": scenes,
            }
            season_dict["episodes"].append(converted_outline)

        converted_outlines.append(season_dict)

    return pb.EpisodeOutline(seasons=converted_outlines)


def check_proposals(ctx, story_info, generate_input, proposals):
    # 检查参数
    if len(proposals) != generate_input.season_nums + 1:
        err_msg = f"策划案数量({len(proposals)})与预期({generate_input.season_nums + 1})不一致"
        logger.error_context(ctx, err_msg)
        raise NewBusinessError(
            BizCode.PARAM_INVALID, f"{BIZ_MSG[BizCode.PARAM_INVALID]}: {err_msg}"
        )


def _build_episode_map(episode_outline):
    """从 proto EpisodeOutline 构建 {season_id-episode_id: episode} 映射。"""
    episode_map = {}
    for season in episode_outline.seasons:
        for episode in season.episodes:
            episode_map[f"{season.season_id}-{episode.episode_id}"] = episode
    return episode_map


def check_episode_outline_content(ctx, episode_outline, select_range):
    """生成分场规划前：检查 select_range 中每集的集大纲 content 是否存在。"""
    episode_map = _build_episode_map(episode_outline)
    existing_ids = sorted(
        episode_map.keys(), key=lambda x: tuple(int(i) for i in x.split("-"))
    )

    for _range in select_range:
        for eid in _range.episode_ids:
            this_id = f"{_range.season_id}-{eid}"
            ep = episode_map.get(this_id)
            if not ep or not ep.content or ep.content == "暂无":
                err_msg = (
                    f"未找到 {this_id} 的集大纲内容，传入的集大纲为：{existing_ids}"
                )
                logger.error_context(ctx, err_msg)
                raise NewBusinessError(
                    BizCode.PARAM_INVALID,
                    f"{BIZ_MSG[BizCode.PARAM_INVALID]}: {err_msg}",
                )


def check_episode_outline_scenes(ctx, episode_outline, select_range):
    """生成剧本前：检查 select_range 中每集的分场规划 scenes 是否存在且有具体内容。"""
    episode_map = _build_episode_map(episode_outline)
    existing_ids = sorted(
        episode_map.keys(), key=lambda x: tuple(int(i) for i in x.split("-"))
    )

    for _range in select_range:
        for eid in _range.episode_ids:
            this_id = f"{_range.season_id}-{eid}"
            ep = episode_map.get(this_id)
            if not ep:
                err_msg = (
                    f"未找到 {this_id} 的分场规划，传入的分场规划为：{existing_ids}"
                )
            elif not ep.scenes:
                err_msg = f"{this_id} 的分场规划 scenes 为空"
            else:
                err_msg = None
            if err_msg:
                logger.error_context(ctx, err_msg)
                raise NewBusinessError(
                    BizCode.PARAM_INVALID,
                    f"{BIZ_MSG[BizCode.PARAM_INVALID]}: {err_msg}",
                )
            for idx, scene in enumerate(ep.scenes):
                if not scene.content:
                    err_msg = f"{this_id} 第{idx+1}场的分场内容为空"
                    logger.warning_context(ctx, err_msg)
