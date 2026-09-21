import re
from copy import deepcopy

import cn2an
import json_repair
from trpc.log import logger

from service.drama_by_fiction.data_model import (
    FictionConfig,
    SeasonProposal,
)

from service.drama_by_fiction.core.data_manager import DataManager
from service.drama_by_fiction.utils import (
    _query_llm_with_postprocess,
    Task,
)


class AnimeScriptDivider:
    def __init__(self, ctx):
        self.ctx = ctx
        self.data_loader = DataManager(ctx)
        self.cfg = FictionConfig()

    def _process_points(self, json_data, season_nums, max_plot_id):
        """处理卡点规划"""
        season_details = json_data["season_details"]
        assert (
            len(season_details) == season_nums
        ), f"season_details 数量({len(season_details)})与预期季数({season_nums})不一致"

        desc_parts = [f"### 划分依据\n{json_data['division_basis']}"]
        points = []

        for i, season in enumerate(season_details):
            point_name = f"第{cn2an.an2cn(i + 1)}季"
            start_plot_id = 0 if i == 0 else points[i - 1]["end_plot_id"] + 1
            end_plot_id = season["end_plot_id"]

            # 校验 plot_id 范围
            for label, pid in [("start", start_plot_id), ("end", end_plot_id)]:
                if pid > max_plot_id:
                    raise ValueError(
                        f"第{i + 1}季 {label}_plot_id({pid}) 超出上限({max_plot_id})"
                    )

            _point = {
                "point_id": i,
                "point_name": f"{point_name}结局",
                "start_plot_id": start_plot_id,
                "end_plot_id": end_plot_id,
            }
            points.append(_point)

            season_title = season["season_title"]
            # 如果 season_title 中已包含"第x季"（中文数字或阿拉伯数字）则不再补充前缀
            has_season_label = bool(
                re.search(r"第[一二三四五六七八九十百千万\d]+季", season_title)
            )
            if has_season_label:
                desc_parts.append(f"### [{season_title}]\n{season['content']}")
            else:
                desc_parts.append(
                    f"### {point_name} [{season_title}]\n{season['content']}"
                )

        return {"points": points, "description": "\n\n".join(desc_parts)}

    def _process_season_division(self, response, plot_list, season_nums):
        json_data = json_repair.loads(response)
        max_plot_id = max(p["plot_id"] for p in plot_list) if plot_list else 0
        point_planning = self._process_points(json_data, season_nums, max_plot_id)
        result = {
            "title": "整体卡点规划",
            "season_id": -1,
            "plot_planning": deepcopy(plot_list),
            "point_planning": point_planning,
        }
        return SeasonProposal.from_dict(result)

    def _build_single_season_division(self, plot_list):
        """单季场景：跳过 LLM，直接构造 SeasonProposal，与后续流程兼容。"""
        if not plot_list:
            raise ValueError("单季构造失败: plot_list 不能为空")
        max_plot_id = max(p["plot_id"] for p in plot_list)

        point_planning = {
            "points": [
                {
                    "point_id": 0,
                    "point_name": "第一季结局",
                    "start_plot_id": 0,
                    "end_plot_id": max_plot_id,
                }
            ],
            "description": "### 划分依据\n仅规划单季，全部情节归入第一季。",
        }
        result = {
            "title": "整体卡点规划",
            "season_id": -1,
            "plot_planning": deepcopy(plot_list),
            "point_planning": point_planning,
        }
        season_division = SeasonProposal.from_dict(result)
        logger.info_context(
            self.ctx, f"Single season, skip LLM division: {season_division}"
        )
        return season_division

    async def generate_season_division(
        self,
        generate_input,
        adaptation,
        plot_list,
        chapter_content,
        suggestion="暂无",
    ):
        # 单季时无需 LLM 分季，直接构造结果
        if generate_input.season_nums == 1:
            return self._build_single_season_division(plot_list)

        chapter_cnt = await self.data_loader.get_max_chapter_idx(
            generate_input.novel_id
        )

        max_plot_id = len(plot_list) - 1

        variables = {
            "SeasonCnt": generate_input.season_nums,
            "EpisodeCnt": generate_input.episode_nums,
            "ChapterCnt": chapter_cnt,
            "MAX_PLOT_ID": max_plot_id,
            "Adaptation": adaptation,
            "Suggestion": suggestion,
            "ChapterContent": chapter_content,
        }

        season_division = _query_llm_with_postprocess(
            ctx=self.ctx,
            task=Task(
                stage="divide",
                prompt_name="season_division",
                post_process_func=self._process_season_division,
                post_process_kwargs={
                    "plot_list": plot_list,
                    "season_nums": generate_input.season_nums,
                },
                variables=variables,
                debug_info="generate season division",
            ),
            model_name=self.cfg.model_name,
            max_retries=self.cfg.retry_cnt,
        )

        logger.info_context(self.ctx, f"season division: {season_division}")
        return season_division
