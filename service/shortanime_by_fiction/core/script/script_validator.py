import re
import json_repair
from trpc.log import logger

from service.shortanime_by_fiction.llms import query_llm, Task


MAX_FIX_ATTEMPTS = 3
_SCENE_HEADER_RE = re.compile(r'^\d+-\d+\.')
_EPISODE_END_RE = re.compile(r'^（第\d+集完）')


class WordCountValidator:
    def __init__(self, ctx, cfg):
        self.ctx = ctx
        self.cfg = cfg

    def validate(self, content: str, min_word: int, max_word: int) -> bool:
        actual = len(content)
        return (min_word - 20) <= actual <= (max_word + 20)

    def _dialogue_stats_per_scene(self, content: str) -> dict:
        """返回每场的 (dialogue_count, description_count)，key 为场头行文本。"""
        scenes = {}
        current_scene = None
        current_stats = [0, 0]
        for raw in content.splitlines():
            line = raw.strip()
            if not line:
                continue
            if _SCENE_HEADER_RE.match(line):
                if current_scene is not None:
                    scenes[current_scene] = tuple(current_stats)
                current_scene = line
                current_stats = [0, 0]
                continue
            if '【闪回' in line:
                continue
            if _EPISODE_END_RE.match(line):
                continue
            if line.startswith('△'):
                current_stats[1] += 1
            else:
                current_stats[0] += 1
        if current_scene is not None:
            scenes[current_scene] = tuple(current_stats)
        return scenes

    def _dialogue_stats(self, content: str) -> tuple:
        """返回 (dialogue_count, description_count)，过滤场头行、闪回行、末尾行、空行。"""
        dialogue, description = 0, 0
        for raw in content.splitlines():
            line = raw.strip()
            if not line:
                continue
            if _SCENE_HEADER_RE.match(line):
                continue
            if '【闪回' in line:
                continue
            if _EPISODE_END_RE.match(line):
                continue
            if line.startswith('△'):
                description += 1
            else:
                dialogue += 1
        return dialogue, description

    def validate_dialogue_ratio(self, content: str, min_ratio: float = 0.6, label: str = "") -> bool:
        """校验台词行占比不低于 min_ratio（默认 6:4）。校验失败时记 warning。"""
        prefix = f"[{label}] " if label else ""
        dialogue, description = self._dialogue_stats(content)
        total = dialogue + description
        if total == 0:
            logger.warning_context(self.ctx, f"{prefix}台词比例校验：有效行为0，跳过")
            return True
        ratio = dialogue / total
        passed = ratio >= min_ratio
        if not passed:
            logger.warning_context(
                self.ctx,
                f"{prefix}台词比例不达标：台词={dialogue}行，描述={description}行，"
                f"比例={ratio:.2%}，要求>={min_ratio:.0%}"
            )
        return passed

    def _build_issues(self, content: str, min_word: int, max_word: int, min_ratio: float, fix_ratio: float, scene_min_ratio: float = 0.4, scene_fix_ratio: float = 0.5) -> str:
        """构造全量状态报告（达标和不达标项均列出），用于日志和 fix prompt。"""
        lines = []
        actual = len(content)
        if not self.validate(content, min_word, max_word):
            direction = "偏多" if actual > max_word else "偏少"
            diff = actual - max_word if actual > max_word else min_word - actual
            lines.append(f"- 字数：当前{actual}字，目标{min_word}-{max_word}字（{direction}{abs(diff)}字）")
        else:
            lines.append(f"- 字数：当前{actual}字，目标{min_word}-{max_word}字（达标）")
        dialogue, description = self._dialogue_stats(content)
        total = dialogue + description
        if total > 0:
            ratio = dialogue / total
            if ratio < min_ratio:
                lines.append(
                    f"- 台词占比（整集）：当前{ratio:.0%}（台词{dialogue}行/描述{description}行），"
                    f"目标≥{fix_ratio:.0%}（当前偏低{fix_ratio - ratio:.0%}）"
                )
            else:
                lines.append(
                    f"- 台词占比（整集）：当前{ratio:.0%}（台词{dialogue}行/描述{description}行），"
                    f"过关线{min_ratio:.0%}（达标）"
                )

        for scene_id, (d, desc) in self._dialogue_stats_per_scene(content).items():
            t = d + desc
            if t == 0:
                continue
            r = d / t
            if r < scene_min_ratio:
                lines.append(
                    f"- 台词占比（{scene_id}）：当前{r:.0%}（台词{d}行/描述{desc}行），"
                    f"目标≥{scene_fix_ratio:.0%}（当前偏低{scene_fix_ratio - r:.0%}）"
                )
            else:
                lines.append(
                    f"- 台词占比（{scene_id}）：当前{r:.0%}（台词{d}行/描述{desc}行），"
                    f"过关线{scene_min_ratio:.0%}（达标）"
                )
        return "\n".join(lines)

    def _has_issues(self, content: str, min_word: int, max_word: int, min_ratio: float, scene_min_ratio: float = 0.4) -> bool:
        """快速判断是否存在不达标项。"""
        if not self.validate(content, min_word, max_word):
            return True
        d, desc = self._dialogue_stats(content)
        t = d + desc
        if t > 0 and d / t < min_ratio:
            return True
        for _, (d, desc) in self._dialogue_stats_per_scene(content).items():
            t = d + desc
            if t > 0 and d / t < scene_min_ratio:
                return True
        return False

    def fix(self, script_content: str, issues: str, prev_script: str = "暂无", follow_script: str = "暂无") -> str:
        """调用 word_count prompt 按问题描述对剧本进行一次微调。"""
        response = query_llm(
            ctx=self.ctx,
            task=Task(
                stage="check",
                prompt_name="script_fix",
                variables={
                    "Script": script_content,
                    "ValidationIssues": issues,
                    "PreviousEpisodeScript": prev_script,
                    "FollowEpisodeScript": follow_script,
                },
                domain="COMMON",
                debug_info="fix script issues",
            ),
            model_name=self.cfg.model_name,
            max_retries=1,
        )
        data = json_repair.loads(response)
        return data.get("content", "").strip()

    def apply_fix(self, content: str, min_word: int, max_word: int, min_ratio: float = 0.6, fix_ratio: float = 0.7, scene_min_ratio: float = 0.4, scene_fix_ratio: float = 0.5, label: str = "", prev_script: str = "暂无", follow_script: str = "暂无") -> str:
        """字数和台词比例联合校验修正，最多 MAX_FIX_ATTEMPTS 次，超限后软降级。
        min_ratio/fix_ratio: 整集过关线/修正目标；scene_min_ratio/scene_fix_ratio: 单场过关线/修正目标。
        """
        if not self.cfg.enable_script_fix:
            return content

        prefix = f"[{label}] " if label else ""
        current = content
        for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
            report = self._build_issues(current, min_word, max_word, min_ratio, fix_ratio, scene_min_ratio, scene_fix_ratio)
            if not self._has_issues(current, min_word, max_word, min_ratio, scene_min_ratio):
                if attempt == 1:
                    logger.info_context(self.ctx, f"{prefix}校验通过，无需修正\n{report}")
                else:
                    logger.info_context(self.ctx, f"{prefix}第{attempt - 1}次修正后校验通过\n{report}")
                return current
            logger.info_context(self.ctx, f"{prefix}第{attempt}次修正（整集过关线{min_ratio:.0%}/单场过关线{scene_min_ratio:.0%}），状态：\n{report}")
            current = self.fix(current, report, prev_script, follow_script)

        report = self._build_issues(current, min_word, max_word, min_ratio, fix_ratio, scene_min_ratio, scene_fix_ratio)
        if self._has_issues(current, min_word, max_word, min_ratio, scene_min_ratio):
            logger.warning_context(
                self.ctx,
                f"{prefix}修正达到上限（{MAX_FIX_ATTEMPTS}次）仍不达标，直接使用最后结果。状态：\n{report}"
            )
        else:
            logger.info_context(self.ctx, f"{prefix}第{MAX_FIX_ATTEMPTS}次修正后校验通过\n{report}")
        return current
