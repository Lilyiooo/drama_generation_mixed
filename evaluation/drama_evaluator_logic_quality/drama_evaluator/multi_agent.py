from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import statistics
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .llm import OpenAIChatClient
from .pipeline import EvaluationConfig, EvaluationPipeline

PROMPT_VERSION = "logic-quality-ledger-v2.4-shared-module-baseline-20260921"
DIMENSIONS = ("logic", "quality")
LABELS = {"logic": "剧本逻辑", "quality": "剧本质量"}
TOP_SCORE_LABELS = {
    "logic": "剧本逻辑总分",
    "quality": "剧本质量最终总分",
}
UNITS = {
    "logic": {
        "causality": ("因果链路", 0.18),
        "character_state": ("角色状态", 0.18),
        "knowledge_state": ("认知状态", 0.14),
        "resource_state": ("资源状态", 0.14),
        "rules_spacetime": ("规则时空", 0.13),
        "foreshadowing": ("伏笔闭环", 0.13),
        "repetition": ("重复检测", 0.10),
    },
    "quality": {
        "character_main": ("人物质量/主角", None),
        "character_support": ("人物质量/配角与角色功能", None),
        "character_relationship": ("人物质量/人物关系", None),
        "hooks_opening": ("戏剧吸引力/开篇叙事吸引力", None),
        "hooks_conflict": ("戏剧吸引力/矛盾建立", None),
        "hooks_continuity": ("戏剧吸引力/集间承接与兑现", None),
        "plot_scenes": ("情节规划/桥段质量", None),
        "plot_emotion": ("情节规划/情绪节奏", None),
        "plot_structure": ("情节规划/情节结构", None),
        "episode_dialogue": ("分集剧本/台词质量", None),
        "episode_visual": ("分集剧本/视觉化质量", None),
        "setting": ("设定质量", None),
    },
}
QUALITY_MODULES = {
    "character": ("人物质量", (
        "character_main", "character_support", "character_relationship",
    )),
    "hooks": ("戏剧吸引力", (
        "hooks_opening", "hooks_conflict", "hooks_continuity",
    )),
    "plot": ("情节规划质量", (
        "plot_scenes", "plot_emotion", "plot_structure",
    )),
    "episode": ("分集剧本质量", (
        "episode_dialogue", "episode_visual",
    )),
    "setting": ("设定质量", ("setting",)),
}
PLOT_BEHAVIOR_CRITERIA = {
    "plot_scenes": ("事件/桥段非重复性", "对手与阻力的差异化", "场景递进结构"),
    "plot_emotion": ("开篇阶段节奏", "中后段节奏", "目标情绪曲线", "连续期待分布"),
    "plot_structure": ("情节量与信息分布", "停滞与注水", "副线整合", "主线推进与阶段兑现"),
}

STANDARD_FILES = {
    "logic": ("剧本逻辑.md",),
    "quality": (
        "剧本质量/人物质量.md",
        "剧本质量/卡点质量.md",
        "剧本质量/情节规划质量.md",
        "剧本质量/分集剧本质量.md",
        "剧本质量/设定质量.md",
    ),
}
VALID_SEVERITIES = frozenset({"S", "A", "B"})
VALID_KINDS = frozenset({"issue", "highlight"})
SUBDIMENSION_FUSION_WEIGHT = 0.5


@dataclass(slots=True)
class MultiAgentEvaluationConfig:
    script_path: Path
    output_dir: Path
    model: str = "gpt-5.6-sol"
    review_workers: int = 3
    audit_workers: int = 3
    arbitration_workers: int = 3
    max_output_tokens: int = 16000
    timeout: float = 600.0
    retries: int = 3
    parse_retries: int = 2
    context_mode: str = "auto"
    direct_char_limit: int = 300000
    chunk_chars: int = 100000
    restart: bool = False

    def validate(self) -> None:
        if not self.script_path.is_file():
            raise ValueError(f"剧本文件不存在：{self.script_path}")
        if not self.model.strip():
            raise ValueError("模型名称不能为空")
        for name in ("review_workers", "audit_workers", "arbitration_workers"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} 必须大于等于 1")
        if self.max_output_tokens < 4000:
            raise ValueError("max_output_tokens 不应低于 4000")
        if self.parse_retries < 1:
            raise ValueError("parse_retries 必须大于等于 1")
        if self.context_mode not in {"auto", "direct", "evidence"}:
            raise ValueError("context_mode 只能是 auto、direct 或 evidence")


class MultiAgentEvaluationPipeline:
    """逻辑、质量分别建立审核台账，并各自独立完成一次子维度整体评分。"""

    def __init__(self, config: MultiAgentEvaluationConfig) -> None:
        config.validate()
        self.config = config
        self.root = Path(__file__).resolve().parent.parent
        self.prompt_root = self.root / "prompts"
        self.artifact_dir = config.output_dir / "multi_agent"
        self.client: OpenAIChatClient | None = None
        self._errors: list[dict[str, str]] = []
        self.standards = self._load_standards()

    @staticmethod
    def _read_text(path: Path) -> str:
        return path.read_text(encoding="utf-8-sig").strip()

    def _load_standards(self) -> dict[str, str]:
        standards: dict[str, str] = {}
        for dimension, paths in STANDARD_FILES.items():
            contents = []
            for relative in paths:
                path = self.prompt_root / relative
                if not path.is_file():
                    raise ValueError(f"缺少评分标准：{path}")
                contents.append(f"\n\n===== {relative} =====\n\n{self._read_text(path)}")
            standards[dimension] = "".join(contents)
        return standards

    @staticmethod
    def _sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                file.write(content)
                if not content.endswith("\n"):
                    file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    def _write_json(self, path: Path, payload: object) -> None:
        self._atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2))

    @staticmethod
    def _read_json(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    def _manifest(self, script: str) -> dict[str, object]:
        return {
            "pipeline": "dimension_ledger_multi_agent",
            "version": PROMPT_VERSION,
            "script_path": str(self.config.script_path.resolve()),
            "script_sha256": self._sha256(script),
            "model": self.config.model,
            "reviewers_per_dimension": 3,
            "context_mode": self.config.context_mode,
            "direct_char_limit": self.config.direct_char_limit,
            "chunk_chars": self.config.chunk_chars,
            "standard_sha256": {key: self._sha256(value) for key, value in self.standards.items()},
        }

    def _prepare_output(self, manifest: dict[str, object]) -> None:
        manifest_path = self.artifact_dir / "manifest.json"
        if self.config.restart:
            if self.artifact_dir.exists():
                shutil.rmtree(self.artifact_dir)
            for name in ("00_多Agent评估报告.md", "multi_agent_result.json", "scores.json"):
                (self.config.output_dir / name).unlink(missing_ok=True)
        elif manifest_path.exists():
            old = self._read_json(manifest_path)
            if {key: old.get(key) for key in manifest} != manifest:
                raise RuntimeError("输出目录中的剧本、模型、配置或评分标准已变化；请换目录或使用 --restart")
        elif (self.config.output_dir / "scores.json").exists():
            raise RuntimeError("输出目录已有其他评测结果，请换目录或使用 --restart")
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(manifest_path, manifest)

    def _create_client(self) -> None:
        self.client = OpenAIChatClient(
            model=self.config.model,
            max_output_tokens=self.config.max_output_tokens,
            timeout=self.config.timeout,
            retries=self.config.retries,
        )

    @staticmethod
    def _extract_json(text: str) -> Any:
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
        decoder = json.JSONDecoder()
        for index, char in enumerate(candidate):
            if char not in "{[":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
                return value
            except json.JSONDecodeError:
                pass
        raise ValueError("模型输出中未找到合法 JSON")

    def _request_json(self, system: str, user: str, *, label: str) -> Any:
        assert self.client is not None
        last_error: Exception | None = None
        for attempt in range(1, self.config.parse_retries + 1):
            suffix = "" if attempt == 1 else "\n\n上次响应格式错误。请以一个完整合法的 JSON 对象重新提交同一任务结果。"
            raw = self.client.text(system, user + suffix, label=f"{label} JSON#{attempt}")
            try:
                return self._extract_json(raw)
            except ValueError as error:
                last_error = error
        raise RuntimeError(f"{label} 未返回合法 JSON：{last_error}")

    @staticmethod
    def _text(value: object, limit: int = 1600) -> str:
        return value.strip()[:limit] if isinstance(value, str) else ""

    @staticmethod
    def _score(value: object, field: str) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{field} 必须是数值")
        result = float(value)
        if not math.isfinite(result) or not 0 <= result <= 100:
            raise ValueError(f"{field} 必须在 0-100")
        return round(result, 2)

    @staticmethod
    def _severity(value: object) -> str:
        severity = str(value or "B").upper().strip().replace("级", "")
        return severity if severity in VALID_SEVERITIES else "B"

    @staticmethod
    def _behavior_rating(value: object, *, required: bool = False) -> int | None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            rating = int(value)
            if float(value) == rating and 1 <= rating <= 5:
                return rating
        if required:
            raise ValueError("情节规划条目必须提供 1-5 的 behavior_rating")
        return None

    @staticmethod
    def _point_anchor(dimension: str, kind: str, severity: str) -> float:
        if kind == "issue":
            return {"S": 10.0, "A": 6.0, "B": 4.0}[severity]
        return {"S": 3.0, "A": 1.0, "B": 0.0}[severity]

    @classmethod
    def _normalize_points(cls, dimension: str, kind: str, severity: str, value: object) -> float:
        # 最终台账严格按照 S/A/B 锚点计分，避免出现 B/20、B级亮点加3分等矛盾。
        # 模型原始 points 仅用于解释，不作为程序计分依据。
        return cls._point_anchor(dimension, kind, severity)

    def _validate_plot_behavior_ratings(self, payload: dict[str, Any], dimension: str) -> list[dict[str, Any]]:
        if dimension != "quality":
            return []
        raw = payload.get("plot_behavior_ratings")
        if not isinstance(raw, list):
            raise ValueError("剧本质量评审缺少 plot_behavior_ratings")
        expected = {
            (unit, criterion)
            for unit, criteria in PLOT_BEHAVIOR_CRITERIA.items()
            for criterion in criteria
        }
        ratings: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            unit = str(item.get("unit", "")).strip()
            criterion = self._text(item.get("criterion"), 120)
            key = (unit, criterion)
            if key not in expected or key in seen:
                continue
            seen.add(key)
            evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
            ratings.append({
                "unit": unit,
                "criterion": criterion,
                "rating": self._behavior_rating(item.get("rating"), required=True),
                "reason": self._text(item.get("reason"), 800),
                "evidence": {
                    "episode": self._text(evidence.get("episode"), 80),
                    "scene": self._text(evidence.get("scene"), 120),
                    "quote": self._text(evidence.get("quote"), 500),
                },
            })
        missing = expected - seen
        if missing:
            raise ValueError(f"plot_behavior_ratings 缺少：{sorted(missing)}")
        return ratings

    def _validate_review(self, payload: Any, dimension: str, reviewer_id: str) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("评审结果必须是对象")
        raw_entries = payload.get("ledger_entries")
        if not isinstance(raw_entries, list):
            raise ValueError("缺少 ledger_entries")
        entries: list[dict[str, Any]] = []
        for index, item in enumerate(raw_entries, start=1):
            if not isinstance(item, dict):
                continue
            unit = str(item.get("unit", "")).strip()
            kind = str(item.get("kind", "issue")).strip().lower()
            if unit not in UNITS[dimension] or kind not in VALID_KINDS:
                continue
            # 不信任模型生成的 ID，由程序保证跨评审全局唯一且断点稳定。
            entry_id = f"{dimension}-{reviewer_id}-E{index:03d}"
            severity = self._severity(item.get("severity"))
            behavior_rating = self._behavior_rating(
                item.get("behavior_rating"),
                required=dimension == "quality" and unit.startswith("plot_"),
            )
            evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
            entries.append({
                "entry_id": entry_id,
                "dimension": dimension,
                "reviewer_id": reviewer_id,
                "unit": unit,
                "kind": kind,
                "category": self._text(item.get("category"), 120),
                "severity": severity,
                "behavior_rating": behavior_rating,
                "points": self._normalize_points(dimension, kind, severity, item.get("points")),
                "claim": self._text(item.get("claim")),
                "reason": self._text(item.get("reason")),
                "confidence": self._text(item.get("confidence"), 20) or "中",
                "evidence": {
                    "episode": self._text(evidence.get("episode"), 80),
                    "scene": self._text(evidence.get("scene"), 120),
                    "quote": self._text(evidence.get("quote"), 500),
                    "contrast_episode": self._text(evidence.get("contrast_episode"), 80),
                    "contrast_scene": self._text(evidence.get("contrast_scene"), 120),
                    "contrast_quote": self._text(evidence.get("contrast_quote"), 500),
                },
            })
        return {
            "dimension": dimension,
            "reviewer_id": reviewer_id,
            "rationale": self._text(payload.get("rationale")),
            "plot_behavior_ratings": self._validate_plot_behavior_ratings(payload, dimension),
            "ledger_entries": entries,
        }

    def _review_prompt(self, dimension: str, reviewer_id: str, source_name: str, source: str) -> tuple[str, str]:
        units = {key: label for key, (label, _) in UNITS[dimension].items()}
        behavior_template = [
            {
                "unit": unit,
                "criterion": criterion,
                "rating": 3,
                "reason": "对照1-5行为锚点的理由",
                "evidence": {"episode": "第X集", "scene": "场次", "quote": "逐字原文"},
            }
            for unit, criteria in PLOT_BEHAVIOR_CRITERIA.items()
            for criterion in criteria
        ] if dimension == "quality" else []
        system = f"""你是 {LABELS[dimension]} 的独立评审 {reviewer_id}。另有两名同维度评审独立工作。你的评估范围是当前大维度。

下面的 <DETAILED_STANDARD> 是本任务的详细评分标准。请逐项使用其中的 S/A/B 定义、情节规划1-5行为映射、计分锚点和边界，以具体文本证据建立评分台账。

{self.standards[dimension]}

统一结构化要求：
1. 罗列所有具有文本证据且实际成立的 S/A/B 问题与亮点。
2. 每条 entry 归入以下 unit 之一：{json.dumps(units, ensure_ascii=False)}。
3. 每条提供逐字原文 quote；前后矛盾与重复类同时提供 contrast_quote 和两处位置。
4. severity 严格按详细标准；points 填正数绝对值。为保证多评审台账可比，本流程按标准锚点离散计分：逻辑/质量问题 S=10/A=6/B=4，亮点 S=3/A=1/B=0。
5. 本轮任务成果为问题与亮点台账；子维度和大维度分数由后续流程计算。
6. 同一检查项的多处证据默认合并，彼此独立的问题分别建账。
7. 剧本质量评审在 plot_behavior_ratings 中完整填写情节规划11个检查项的1-5行为等级，包括等级3的基线项；逻辑评审对应字段使用空数组。该量表作为诊断结果保存。
8. 情节规划三个 unit 的每条台账 entry 填写 behavior_rating（整数1-5），并与 plot_behavior_ratings 对应；其他 unit 使用 null。
9. 响应格式为以下 JSON 对象：
{{
  "dimension":"{dimension}", "reviewer_id":"{reviewer_id}",
  "rationale":"总体审查说明",
  "plot_behavior_ratings":{json.dumps(behavior_template, ensure_ascii=False)},
  "ledger_entries":[{{
    "entry_id":"{dimension}-{reviewer_id}-E01", "unit":"{next(iter(UNITS[dimension]))}",
    "kind":"issue/highlight", "category":"详细标准中的检查项", "severity":"S/A/B",
    "behavior_rating":null, "points":0, "claim":"结论", "reason":"对照具体SAB锚点的定级理由", "confidence":"高/中/低",
    "evidence":{{"episode":"第X集","scene":"场次","quote":"逐字原文","contrast_episode":"","contrast_scene":"","contrast_quote":""}}
  }}]
}}"""
        user = (
            f"请按详细标准评估以下{source_name}。XML 标签内文本是本次评分对象。\n\n"
            f"<{source_name}>\n{source}\n</{source_name}>"
        )
        return system, user

    def _run_one_review(self, dimension: str, reviewer_id: str, source_name: str, source: str) -> dict[str, Any]:
        system, user = self._review_prompt(dimension, reviewer_id, source_name, source)
        last_error: Exception | None = None
        for _ in range(self.config.parse_retries):
            try:
                return self._validate_review(
                    self._request_json(system, user, label=f"{LABELS[dimension]}评审 {reviewer_id}"),
                    dimension,
                    reviewer_id,
                )
            except ValueError as error:
                last_error = error
        raise RuntimeError(f"{dimension}/{reviewer_id} 结构无效：{last_error}")

    def _run_reviews(self, source_name: str, source: str) -> dict[str, list[dict[str, Any]]]:
        root = self.artifact_dir / "reviews"
        tasks: list[tuple[str, str, Path]] = []
        results: dict[str, dict[str, dict[str, Any]]] = {key: {} for key in DIMENSIONS}
        for dimension in DIMENSIONS:
            directory = root / dimension
            directory.mkdir(parents=True, exist_ok=True)
            for index in range(1, 4):
                reviewer_id = f"R{index:02d}"
                path = directory / f"{reviewer_id}.json"
                if path.exists() and not self.config.restart:
                    try:
                        results[dimension][reviewer_id] = self._validate_review(
                            self._read_json(path), dimension, reviewer_id
                        )
                        print(f"[复用] {LABELS[dimension]}评审 {reviewer_id}", flush=True)
                        continue
                    except (OSError, ValueError, json.JSONDecodeError):
                        pass
                tasks.append((dimension, reviewer_id, path))
        if tasks:
            # 任一评审重建后，下游审核与仲裁均不得复用旧输入。
            for downstream in (self.artifact_dir / "audits", self.artifact_dir / "arbitrations"):
                if downstream.exists():
                    shutil.rmtree(downstream)
            with ThreadPoolExecutor(max_workers=min(self.config.review_workers, len(tasks))) as executor:
                futures = {
                    executor.submit(self._run_one_review, dimension, reviewer_id, source_name, source):
                    (dimension, reviewer_id, path)
                    for dimension, reviewer_id, path in tasks
                }
                for future in as_completed(futures):
                    dimension, reviewer_id, path = futures[future]
                    report = future.result()
                    results[dimension][reviewer_id] = report
                    self._write_json(path, report)
        ordered: dict[str, list[dict[str, Any]]] = {}
        for dimension in DIMENSIONS:
            ordered[dimension] = [results[dimension][key] for key in sorted(results[dimension])]
            if len(ordered[dimension]) != 3:
                raise RuntimeError(f"{LABELS[dimension]}需要完整的 3 份评审结果")
        return ordered

    @staticmethod
    def _scored_subdimensions(dimension: str) -> dict[str, str]:
        if dimension == "quality":
            return {key: value[0] for key, value in QUALITY_MODULES.items()}
        return {key: value[0] for key, value in UNITS[dimension].items()}

    def _holistic_score_prompt(
        self, dimension: str, source_name: str, source: str
    ) -> tuple[str, str]:
        subdimensions = self._scored_subdimensions(dimension)
        system = f"""你是专门负责{LABELS[dimension]}子维度整体评分的严格评委。请通读完整剧本，依据详细标准，对每个指定子维度分别给出 0-100 分。

{self.standards[dimension]}

评分要求：
1. 评分对象仅限：{json.dumps(subdimensions, ensure_ascii=False)}。
2. 每个分数综合该子维度在全剧中的整体表现、问题数量、严重程度、覆盖范围与亮点。
3. 本轮独立阅读剧本并完成子维度整体评分；程序会在审核完成后融合分数。
4. 剧本质量评分对象为五个模块，结果字段对应这五个模块。
5. 五个模块分别独立评分，外层融合由程序执行。
6. 响应格式为以下 JSON 对象：
{{
  "dimension":"{dimension}",
  "subdimension_scores":{json.dumps({key: 0 for key in subdimensions}, ensure_ascii=False)},
  "rationales":{json.dumps({key: "评分依据" for key in subdimensions}, ensure_ascii=False)}
}}"""
        user = (
            f"请对以下{source_name}进行独立的子维度整体评分。XML 标签内文本是本次评分对象。\n\n"
            f"<{source_name}>\n{source}\n</{source_name}>"
        )
        return system, user

    def _validate_holistic_score(self, payload: Any, dimension: str) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("整体评分结果必须是对象")
        expected = self._scored_subdimensions(dimension)
        raw_scores = payload.get("subdimension_scores")
        if not isinstance(raw_scores, dict):
            raise ValueError("缺少 subdimension_scores")
        if set(raw_scores) != set(expected):
            raise ValueError(f"subdimension_scores 必须且只能包含：{', '.join(expected)}")
        raw_rationales = payload.get("rationales")
        if not isinstance(raw_rationales, dict):
            raw_rationales = {}
        return {
            "dimension": dimension,
            "subdimension_scores": {
                key: self._score(raw_scores[key], f"subdimension_scores.{key}") for key in expected
            },
            "rationales": {key: self._text(raw_rationales.get(key), 1200) for key in expected},
        }

    def _run_one_holistic_score(
        self, dimension: str, source_name: str, source: str
    ) -> dict[str, Any]:
        system, user = self._holistic_score_prompt(dimension, source_name, source)
        last_error: Exception | None = None
        for _ in range(self.config.parse_retries):
            try:
                return self._validate_holistic_score(
                    self._request_json(system, user, label=f"{LABELS[dimension]}子维度整体评分"),
                    dimension,
                )
            except ValueError as error:
                last_error = error
        raise RuntimeError(f"{dimension} 子维度整体评分结构无效：{last_error}")

    def _run_holistic_scores(self, source_name: str, source: str) -> dict[str, dict[str, Any]]:
        directory = self.artifact_dir / "holistic_scores"
        directory.mkdir(parents=True, exist_ok=True)
        results: dict[str, dict[str, Any]] = {}
        pending: list[tuple[str, Path]] = []
        for dimension in DIMENSIONS:
            path = directory / f"{dimension}.json"
            if path.exists() and not self.config.restart:
                try:
                    results[dimension] = self._validate_holistic_score(self._read_json(path), dimension)
                    print(f"[复用] {LABELS[dimension]}子维度整体评分", flush=True)
                    continue
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
            pending.append((dimension, path))
        if pending:
            with ThreadPoolExecutor(max_workers=min(self.config.audit_workers, len(pending))) as executor:
                futures = {
                    executor.submit(self._run_one_holistic_score, dimension, source_name, source):
                    (dimension, path)
                    for dimension, path in pending
                }
                for future in as_completed(futures):
                    dimension, path = futures[future]
                    result = future.result()
                    results[dimension] = result
                    self._write_json(path, result)
        return {dimension: results[dimension] for dimension in DIMENSIONS}

    @staticmethod
    def _lookup(reviews: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {entry["entry_id"]: entry for review in reviews for entry in review["ledger_entries"]}

    def _audit_prompt(
        self, dimension: str, reviews: list[dict[str, Any]], source_name: str, source: str
    ) -> tuple[str, str]:
        bundle = json.loads(json.dumps(reviews, ensure_ascii=False))
        system = f"""你是 {LABELS[dimension]} 的台账审核 Agent。你的任务是逐条审核三份评审意见中的全部 S/A/B 加扣分点，并依据原剧本与详细标准完成归并和纠错。审核范围由输入台账项确定。

{self.standards[dimension]}

审核规则：
1. 覆盖输入中的每个 source_entry_id；同一事实/同一检查项重复报告时合并，并列出全部来源 ID。
2. 将内容错误、范围越界、证据不足、基线误报亮点或重复计分的条目标为 rejected。
3. 判断成立且 kind/unit/severity/points 均正确的标 confirmed；若分类、SAB或分值有错，纠正后 confirmed，并在 reason 说明。
4. 若成立性或正确分级存在无法消除的实质争议，标 needs_arbitration，并把双方理由、冲突证据和待裁问题写清楚。
5. 审核每条 S/A/B 的成立性与分级合理性，包括 unit、kind、severity、claim 与详细标准的匹配度，以及重复计分和基线误报亮点的情况。
6. 以证据质量作为判断依据，并保留证据充分的少数评审意见。
7. points 为正数绝对值，并严格使用 S/A/B 对应锚点；程序会按 severity 重新确定分值。
8. 情节规划 unit 必须审核并输出1-5整数 behavior_rating，其他 unit 输出 null。
9. 响应格式为以下 JSON 对象：
{{"dimension":"{dimension}","audited_entries":[{{
 "audit_id":"{dimension}-A01","source_entry_ids":["..."],"status":"confirmed/rejected/needs_arbitration",
 "unit":"{next(iter(UNITS[dimension]))}","kind":"issue/highlight","category":"...","severity":"S/A/B","behavior_rating":null,"points":0,
 "claim":"...","evidence":{{"episode":"...","scene":"...","quote":"...","contrast_episode":"","contrast_scene":"","contrast_quote":""}},
 "reason":"逐条审核及纠错理由","arbitration_question":"仅争议时填写"
}}]}}"""
        user = (
            f"<INDEPENDENT_REVIEWS>\n{json.dumps(bundle, ensure_ascii=False)}\n</INDEPENDENT_REVIEWS>\n\n"
            f"<{source_name}>\n{source}\n</{source_name}>"
        )
        return system, user

    def _validate_audit(self, payload: Any, dimension: str, reviews: list[dict[str, Any]]) -> dict[str, Any]:
        if not isinstance(payload, dict) or not isinstance(payload.get("audited_entries"), list):
            raise ValueError("审核结果缺少 audited_entries")
        lookup = self._lookup(reviews)
        output: list[dict[str, Any]] = []
        covered: set[str] = set()
        seen_audit_ids: set[str] = set()
        for index, item in enumerate(payload["audited_entries"], start=1):
            if not isinstance(item, dict):
                raise ValueError("audited_entries 中存在非对象条目")
            source_ids = [str(value) for value in item.get("source_entry_ids", []) if str(value) in lookup]
            source_ids = list(dict.fromkeys(source_ids))
            if not source_ids:
                raise ValueError("审核条目缺少有效 source_entry_ids")
            duplicated = covered.intersection(source_ids)
            if duplicated:
                raise ValueError(f"源条目被重复归并：{sorted(duplicated)}")
            covered.update(source_ids)
            audit_id = f"{dimension}-A{index:03d}"
            if audit_id in seen_audit_ids:
                raise ValueError(f"重复 audit_id：{audit_id}")
            seen_audit_ids.add(audit_id)
            status = str(item.get("status", "rejected")).lower()
            if status not in {"confirmed", "rejected", "needs_arbitration"}:
                status = "rejected"
            unit = str(item.get("unit", lookup[source_ids[0]]["unit"]))
            kind = str(item.get("kind", lookup[source_ids[0]]["kind"])).lower()
            if unit not in UNITS[dimension]:
                unit = lookup[source_ids[0]]["unit"]
            if kind not in VALID_KINDS:
                kind = lookup[source_ids[0]]["kind"]
            severity = self._severity(item.get("severity"))
            behavior_rating = self._behavior_rating(
                item.get("behavior_rating", lookup[source_ids[0]].get("behavior_rating")),
                required=dimension == "quality" and unit.startswith("plot_"),
            )
            evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
            output.append({
                "audit_id": audit_id,
                "dimension": dimension,
                "source_entry_ids": source_ids,
                "status": status,
                "unit": unit,
                "kind": kind,
                "category": self._text(item.get("category"), 120),
                "severity": severity,
                "behavior_rating": behavior_rating,
                "points": self._normalize_points(dimension, kind, severity, item.get("points")),
                "claim": self._text(item.get("claim")),
                "evidence": {
                    "episode": self._text(evidence.get("episode"), 80),
                    "scene": self._text(evidence.get("scene"), 120),
                    "quote": self._text(evidence.get("quote"), 500),
                    "contrast_episode": self._text(evidence.get("contrast_episode"), 80),
                    "contrast_scene": self._text(evidence.get("contrast_scene"), 120),
                    "contrast_quote": self._text(evidence.get("contrast_quote"), 500),
                },
                "reason": self._text(item.get("reason")),
                "arbitration_question": self._text(item.get("arbitration_question")),
            })
        missing = set(lookup) - covered
        if missing:
            raise ValueError(f"审核未覆盖全部源条目，缺少：{sorted(missing)}")
        return {
            "dimension": dimension,
            "audited_entries": output,
            "source_count": len(lookup),
            "covered_count": len(covered),
        }

    def _run_one_audit(
        self, dimension: str, reviews: list[dict[str, Any]], source_name: str, source: str
    ) -> dict[str, Any]:
        system, user = self._audit_prompt(dimension, reviews, source_name, source)
        last_error: Exception | None = None
        for attempt in range(1, self.config.parse_retries + 1):
            suffix = "" if attempt == 1 else "\n\n上次审核未逐条且唯一覆盖所有 source_entry_id。请完整重做，确保每个源 ID 恰好出现一次。"
            try:
                payload = self._request_json(system, user + suffix, label=f"{LABELS[dimension]}台账审核")
                return self._validate_audit(payload, dimension, reviews)
            except ValueError as error:
                last_error = error
        raise RuntimeError(f"{LABELS[dimension]}台账审核结构无效：{last_error}")

    def _run_audits(
        self, reviews: dict[str, list[dict[str, Any]]], source_name: str, source: str
    ) -> dict[str, dict[str, Any]]:
        directory = self.artifact_dir / "audits"
        directory.mkdir(parents=True, exist_ok=True)
        results: dict[str, dict[str, Any]] = {}
        pending: list[str] = []
        for dimension in DIMENSIONS:
            path = directory / f"{dimension}.json"
            if path.exists() and not self.config.restart:
                try:
                    results[dimension] = self._validate_audit(self._read_json(path), dimension, reviews[dimension])
                    print(f"[复用] {LABELS[dimension]}台账审核", flush=True)
                    continue
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
            pending.append(dimension)
        if pending:
            arbitration_dir = self.artifact_dir / "arbitrations"
            for dimension in pending:
                (arbitration_dir / f"{dimension}.json").unlink(missing_ok=True)
            with ThreadPoolExecutor(max_workers=min(self.config.audit_workers, len(pending))) as executor:
                futures = {
                    executor.submit(self._run_one_audit, dimension, reviews[dimension], source_name, source): dimension
                    for dimension in pending
                }
                for future in as_completed(futures):
                    dimension = futures[future]
                    result = future.result()
                    results[dimension] = result
                    self._write_json(directory / f"{dimension}.json", result)
        return results

    def _arbitration_prompt(
        self, dimension: str, candidates: list[dict[str, Any]], reviews: list[dict[str, Any]], source_name: str, source: str
    ) -> tuple[str, str]:
        lookup = self._lookup(reviews)
        sources = {
            item["audit_id"]: [lookup[source_id] for source_id in item["source_entry_ids"] if source_id in lookup]
            for item in candidates
        }
        system = f"""你是 {LABELS[dimension]} 的有限仲裁 Agent。你的裁决范围是审核 Agent 提交的争议点。

{self.standards[dimension]}

逐条阅读争议说明、正反意见、原文和详细 S/A/B 标准。每项选择 confirmed 或 rejected；confirmed 时给出正确的 unit、kind、severity、behavior_rating 和 points。情节规划 unit 的 behavior_rating 使用1-5整数，其他 unit 使用 null。以证据质量完成裁决，证据达到标准的条目标为 confirmed，其余条目标为 rejected。响应格式为以下 JSON 对象：
{{"dimension":"{dimension}","decisions":[{{"audit_id":"...","status":"confirmed/rejected","unit":"...","kind":"issue/highlight","severity":"S/A/B","behavior_rating":null,"points":0,"reason":"裁决理由"}}]}}"""
        user = (
            f"<DISPUTES>\n{json.dumps(candidates, ensure_ascii=False)}\n</DISPUTES>\n\n"
            f"<SOURCE_REVIEWS>\n{json.dumps(sources, ensure_ascii=False)}\n</SOURCE_REVIEWS>\n\n"
            f"<{source_name}>\n{source}\n</{source_name}>"
        )
        return system, user

    def _run_one_arbitration(
        self, dimension: str, audit: dict[str, Any], reviews: list[dict[str, Any]], source_name: str, source: str
    ) -> dict[str, Any]:
        candidates = [item for item in audit["audited_entries"] if item["status"] == "needs_arbitration"]
        system, user = self._arbitration_prompt(dimension, candidates, reviews, source_name, source)
        payload = self._request_json(system, user, label=f"{LABELS[dimension]}有限仲裁")
        raw = payload.get("decisions", []) if isinstance(payload, dict) else []
        valid = {item["audit_id"]: item for item in candidates}
        decisions = []
        seen = set()
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, dict) or item.get("audit_id") not in valid:
                continue
            audit_id = str(item["audit_id"])
            seen.add(audit_id)
            base = valid[audit_id]
            status = str(item.get("status", "rejected")).lower()
            if status not in {"confirmed", "rejected"}:
                status = "rejected"
            unit = str(item.get("unit", base["unit"]))
            kind = str(item.get("kind", base["kind"])).lower()
            if unit not in UNITS[dimension]: unit = base["unit"]
            if kind not in VALID_KINDS: kind = base["kind"]
            severity = self._severity(item.get("severity"))
            behavior_rating = self._behavior_rating(
                item.get("behavior_rating", base.get("behavior_rating")),
                required=dimension == "quality" and unit.startswith("plot_"),
            )
            decisions.append({
                "audit_id": audit_id, "status": status, "unit": unit, "kind": kind,
                "severity": severity,
                "behavior_rating": behavior_rating,
                "points": self._normalize_points(dimension, kind, severity, item.get("points")),
                "reason": self._text(item.get("reason")),
            })
        for audit_id, base in valid.items():
            if audit_id not in seen:
                decisions.append({
                    "audit_id": audit_id, "status": "rejected", "unit": base["unit"],
                    "kind": base["kind"], "severity": base["severity"],
                    "behavior_rating": base.get("behavior_rating"), "points": base["points"],
                    "reason": "仲裁未覆盖该争议，按证据不足拒绝。",
                })
        return {"dimension": dimension, "status": "llm", "decisions": decisions}

    def _run_arbitrations(
        self, audits: dict[str, dict[str, Any]], reviews: dict[str, list[dict[str, Any]]], source_name: str, source: str
    ) -> dict[str, dict[str, Any]]:
        directory = self.artifact_dir / "arbitrations"
        directory.mkdir(parents=True, exist_ok=True)
        results: dict[str, dict[str, Any]] = {}
        pending: list[str] = []
        for dimension in DIMENSIONS:
            candidates = [item for item in audits[dimension]["audited_entries"] if item["status"] == "needs_arbitration"]
            path = directory / f"{dimension}.json"
            if not candidates:
                result = {"dimension": dimension, "status": "not_needed", "decisions": []}
                results[dimension] = result
                self._write_json(path, result)
                continue
            if path.exists() and not self.config.restart:
                cached = self._read_json(path)
                cached_ids = {item.get("audit_id") for item in cached.get("decisions", [])}
                if cached_ids == {item["audit_id"] for item in candidates}:
                    results[dimension] = cached
                    print(f"[复用] {LABELS[dimension]}有限仲裁", flush=True)
                    continue
            pending.append(dimension)
        if pending:
            with ThreadPoolExecutor(max_workers=min(self.config.arbitration_workers, len(pending))) as executor:
                futures = {
                    executor.submit(self._run_one_arbitration, dimension, audits[dimension], reviews[dimension], source_name, source): dimension
                    for dimension in pending
                }
                for future in as_completed(futures):
                    dimension = futures[future]
                    result = future.result()
                    results[dimension] = result
                    self._write_json(directory / f"{dimension}.json", result)
        return results

    @staticmethod
    def _quality_module_scores(entries: list[dict[str, Any]]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for module, (_, units) in QUALITY_MODULES.items():
            selected = [item for item in entries if item["unit"] in units]
            penalties = sum(item["points"] for item in selected if item["kind"] == "issue")
            bonuses = sum(item["points"] for item in selected if item["kind"] == "highlight")
            scores[module] = round(
                max(0.0, min(100.0, 90.0 - min(90.0, penalties) + min(10.0, bonuses))),
                2,
            )
        return scores

    @staticmethod
    def _unit_entry_counts(
        dimension: str, entries: list[dict[str, Any]]
    ) -> dict[str, dict[str, int]]:
        return {
            unit: {
                "issues": sum(item["kind"] == "issue" for item in entries if item["unit"] == unit),
                "highlights": sum(item["kind"] == "highlight" for item in entries if item["unit"] == unit),
                "total": sum(item["unit"] == unit for item in entries),
            }
            for unit in UNITS[dimension]
        }

    @staticmethod
    def _aggregate_plot_behavior_ratings(reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for unit, criteria in PLOT_BEHAVIOR_CRITERIA.items():
            for criterion in criteria:
                samples = []
                for review in reviews:
                    match = next(
                        item
                        for item in review["plot_behavior_ratings"]
                        if item["unit"] == unit and item["criterion"] == criterion
                    )
                    samples.append({
                        "reviewer_id": review["reviewer_id"],
                        "rating": match["rating"],
                        "reason": match["reason"],
                        "evidence": match["evidence"],
                    })
                values = [item["rating"] for item in samples]
                output.append({
                    "unit": unit,
                    "unit_label": UNITS["quality"][unit][0],
                    "criterion": criterion,
                    "mean": round(statistics.mean(values), 2),
                    "minimum": min(values),
                    "maximum": max(values),
                    "samples": samples,
                })
        return output

    @staticmethod
    def _rating(score: float) -> str:
        return "S" if score >= 90 else "A" if score >= 75 else "B" if score >= 60 else "C"

    def _score_units(self, dimension: str, entries: list[dict[str, Any]]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for unit in UNITS[dimension]:
            selected = [item for item in entries if item["unit"] == unit]
            penalties = sum(item["points"] for item in selected if item["kind"] == "issue")
            bonuses = sum(item["points"] for item in selected if item["kind"] == "highlight")
            scores[unit] = round(max(0.0, min(100.0, 90.0 - min(90.0, penalties) + min(10.0, bonuses))), 2)
        return scores

    def _aggregate(
        self,
        reviews: dict[str, list[dict[str, Any]]],
        audits: dict[str, dict[str, Any]],
        arbitrations: dict[str, dict[str, Any]],
        holistic_scores: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        dimensions: dict[str, Any] = {}
        all_final_entries: list[dict[str, Any]] = []
        for dimension in DIMENSIONS:
            decisions = {item["audit_id"]: item for item in arbitrations[dimension].get("decisions", [])}
            final_entries: list[dict[str, Any]] = []
            for audited in audits[dimension]["audited_entries"]:
                status = audited["status"]
                final = dict(audited)
                if status == "needs_arbitration":
                    decision = decisions.get(audited["audit_id"])
                    if decision:
                        final.update({
                            key: decision[key]
                            for key in ("status", "unit", "kind", "severity", "behavior_rating", "points")
                        })
                        final["final_reason"] = decision["reason"]
                    else:
                        final["status"] = "rejected"
                        final["final_reason"] = "缺少有效仲裁，按证据不足拒绝。"
                else:
                    final["final_reason"] = audited["reason"]
                final_entries.append(final)

            confirmed = [item for item in final_entries if item["status"] == "confirmed"]
            if dimension == "quality":
                ledger_subdimension_scores = self._quality_module_scores(confirmed)
            else:
                ledger_subdimension_scores = self._score_units(dimension, confirmed)

            holistic_subdimension_scores = holistic_scores[dimension]["subdimension_scores"]
            expected = self._scored_subdimensions(dimension)
            if set(ledger_subdimension_scores) != set(expected):
                raise RuntimeError(f"{LABELS[dimension]}台账子维度与整体评分子维度不一致")
            fused_subdimension_scores = {
                key: round(
                    SUBDIMENSION_FUSION_WEIGHT * ledger_subdimension_scores[key]
                    + (1 - SUBDIMENSION_FUSION_WEIGHT) * holistic_subdimension_scores[key],
                    2,
                )
                for key in expected
            }
            ledger_mean = round(statistics.mean(ledger_subdimension_scores.values()), 2)
            holistic_mean = round(statistics.mean(holistic_subdimension_scores.values()), 2)
            final_score = round(statistics.mean(fused_subdimension_scores.values()), 2)
            dimensions[dimension] = {
                "label": LABELS[dimension],
                "ledger_subdimension_scores": ledger_subdimension_scores,
                "holistic_subdimension_scores": holistic_subdimension_scores,
                "holistic_rationales": holistic_scores[dimension]["rationales"],
                "fused_subdimension_scores": fused_subdimension_scores,
                "ledger_subdimension_mean": ledger_mean,
                "holistic_subdimension_mean": holistic_mean,
                "final_score": final_score,
                "rating": self._rating(final_score),
                "audited_unit_entry_counts": self._unit_entry_counts(dimension, confirmed),
                "confirmed_entry_count": len(confirmed),
                "rejected_entry_count": len(final_entries) - len(confirmed),
                "score_method": (
                    "剧本质量五个模块各以90分为基线，模块内确认台账项直接加扣；每个模块融合分=50%台账分+50%独立整体评分；剧本质量得分=五个模块融合分的算术平均"
                    if dimension == "quality"
                    else "每个逻辑子维度以90分为基线并按确认台账项加扣；每个子维度融合分=50%台账分+50%独立整体评分；剧本逻辑得分=七个子维度融合分的算术平均"
                ),
            }
            all_final_entries.extend(final_entries)
        return {
            "version": PROMPT_VERSION,
            "script": str(self.config.script_path.resolve()),
            "model": self.config.model,
            "architecture": "2 dimensions × 3 ledger reviewers + 2 dimension audits + up to 2 dimension arbitrations + 2 independent holistic subdimension scorers",
            "dimensions": dimensions,
            "plot_behavior_ratings": self._aggregate_plot_behavior_ratings(reviews["quality"]),
            "final_ledger_entries": all_final_entries,
            "call_policy": {
                "reviews": 6,
                "audits": 2,
                "arbitrations_max": 2,
                "holistic_subdimension_scores": 2,
            },
        }

    @staticmethod
    def _md(value: object, limit: int = 240) -> str:
        text = str(value or "").replace("|", "\\|").replace("\n", " ").strip()
        return text[:limit] + ("…" if len(text) > limit else "")

    def _render_report(self, result: dict[str, Any]) -> str:
        top: list[str] = []
        children: list[str] = []
        unit_details: list[str] = []
        for dimension in DIMENSIONS:
            item = result["dimensions"][dimension]
            top.append(
                f"| {item['label']} | {item['ledger_subdimension_mean']:.1f} | "
                f"{item['holistic_subdimension_mean']:.1f} | {item['final_score']:.1f} | "
                f"{item['rating']} | {item['confirmed_entry_count']} |"
            )
            labels = self._scored_subdimensions(dimension)
            for key, label in labels.items():
                children.append(
                    f"| {item['label']} | {label} | {item['ledger_subdimension_scores'][key]:.1f} | "
                    f"{item['holistic_subdimension_scores'][key]:.1f} | "
                    f"{item['fused_subdimension_scores'][key]:.1f} | "
                    f"{self._md(item['holistic_rationales'].get(key, ''), 200)} |"
                )
            if dimension == "quality":
                for key, counts in item["audited_unit_entry_counts"].items():
                    unit_details.append(
                        f"| {UNITS['quality'][key][0]} | {counts['issues']} | "
                        f"{counts['highlights']} | {counts['total']} |"
                    )

        plot_ratings = [
            f"| {item['unit_label']} | {item['criterion']} | {item['mean']:.2f} | "
            f"{item['minimum']}–{item['maximum']} |"
            for item in result["plot_behavior_ratings"]
        ]

        entries: list[str] = []
        for item in result["final_ledger_entries"]:
            evidence = item["evidence"]
            quote = f"[{evidence['episode']}/{evidence['scene']}] {evidence['quote']}"
            if evidence["contrast_quote"]:
                quote += f" ⇄ [{evidence['contrast_episode']}/{evidence['contrast_scene']}] {evidence['contrast_quote']}"
            entries.append(
                f"| {item['audit_id']} | {LABELS[item['dimension']]} | {UNITS[item['dimension']][item['unit']][0]} | "
                f"{item['kind']} | {item['severity']} | {item.get('behavior_rating') or '-'} | "
                f"{item['points']:.1f} | {item['status']} | "
                f"{self._md(item['claim'])} | {self._md(quote, 320)} | {self._md(item['final_reason'])} |"
            )
        return "\n".join([
            "# 剧本逻辑与质量多 Agent 台账评估报告", "",
            f"- 输入剧本：`{result['script']}`", f"- 评测模型：`{result['model']}`",
            "- 架构：逻辑、质量各3个独立台账评审 → 每维度1个审核 → 每维度按需1个仲裁 → 每个大维度各1次独立子维度整体评分。",
            "- 计分：剧本质量五个模块各以90分为基线，模块内全部确认问题和亮点直接加扣；每个模块融合分 = 50% 台账分 + 50% 独立整体评分；剧本质量得分为五个模块融合分的算术平均。", "",
            "## 大维度分数", "",
            "| 大维度 | 台账子维度均分 | 整体评分子维度均分 | 最终得分 | 评级 | 确认台账项 |",
            "|---|---:|---:|---:|:---:|---:|", *top, "",
            "## 模块与子维度融合分数", "",
            "| 大维度 | 模块或子维度 | 审核后台账分 | 独立整体评分 | 50/50融合分 | 整体评分依据 |",
            "|---|---|---:|---:|---:|---|", *children, "",
            "## 剧本质量台账的12个证据分类", "",
            "这12类用于定位证据所属方面；同一模块内各类确认条目共同作用于该模块的一份90分基线。", "",
            "| 证据分类 | 确认问题 | 确认亮点 | 合计 |", "|---|---:|---:|---:|", *unit_details, "",
            "## 情节规划1–5行为量表诊断", "",
            "该表汇总3名质量评审的独立行为等级，不直接参与当前台账加扣分或50/50融合。", "",
            "| 底层单元 | 检查项 | 三评审均值 | 范围 |",
            "|---|---|---:|:---:|", *plot_ratings, "",
            "## 全部审核台账", "",
            "| ID | 大维度 | 证据分类 | 类型 | SAB | 1-5行为量表 | 分值 | 最终状态 | 判断 | 原文证据 | 审核/仲裁理由 |",
            "|---|---|---|---|:---:|:---:|---:|---|---|---|---|",
            *(entries or ["| - | - | - | - | - | - | - | - | 无台账项 | - | - |"]), "",
            "机器可读完整结果见 `multi_agent_result.json`；评审、审核、仲裁与独立整体评分原始结果位于 `multi_agent/`。",
        ])

    def workload(self) -> dict[str, object]:
        script = self._read_text(self.config.script_path)
        use_evidence = self.config.context_mode == "evidence" or (
            self.config.context_mode == "auto" and len(script) > self.config.direct_char_limit
        )
        chunks = EvaluationPipeline._split_script(script, self.config.chunk_chars) if use_evidence else []
        return {
            "script": str(self.config.script_path.resolve()), "characters": len(script),
            "model": self.config.model, "output_dir": str(self.config.output_dir.resolve()),
            "context_mode": "evidence" if use_evidence else "direct",
            "core_model_calls": {"reviews": 6, "audits": 2, "holistic_subdimension_scores": 2, "arbitrations": "0-2", "min": 10, "max": 12},
            "evidence_extraction_calls": len(chunks),
            "total_calls_excluding_retries": {"min": 10 + len(chunks), "max": 12 + len(chunks)},
            "parallelism": {"review_workers": self.config.review_workers, "audit_workers": self.config.audit_workers, "arbitration_workers": self.config.arbitration_workers},
        }

    def run(self) -> Path:
        script = self._read_text(self.config.script_path)
        if not script:
            raise ValueError("剧本文件为空")
        self._prepare_output(self._manifest(script))
        self._create_client()
        source_pipeline = EvaluationPipeline(EvaluationConfig(
            script_path=self.config.script_path, output_dir=self.artifact_dir,
            model=self.config.model, workers=self.config.review_workers,
            max_output_tokens=self.config.max_output_tokens, timeout=self.config.timeout,
            retries=self.config.retries, context_mode=self.config.context_mode,
            direct_char_limit=self.config.direct_char_limit, chunk_chars=self.config.chunk_chars,
            restart=self.config.restart,
        ))
        source_pipeline.client = self.client
        source_name, source = source_pipeline._evaluation_source(script)
        reviews = self._run_reviews(source_name, source)
        audits = self._run_audits(reviews, source_name, source)
        arbitrations = self._run_arbitrations(audits, reviews, source_name, source)
        holistic_scores = self._run_holistic_scores(source_name, source)
        result = self._aggregate(reviews, audits, arbitrations, holistic_scores)
        result["artifacts"] = {
            "reviews": str((self.artifact_dir / "reviews").resolve()),
            "audits": str((self.artifact_dir / "audits").resolve()),
            "arbitrations": str((self.artifact_dir / "arbitrations").resolve()),
            "holistic_scores": str((self.artifact_dir / "holistic_scores").resolve()),
        }
        self._write_json(self.config.output_dir / "multi_agent_result.json", result)
        self._write_json(self.config.output_dir / "scores.json", {
            "script": result["script"], "model": result["model"],
            "evaluation_mode": "logic_quality_shared_module_baseline_fusion_multi_agent_6_2_2_2",
            "scores": {
                TOP_SCORE_LABELS[key]: {
                    "final_score": result["dimensions"][key]["final_score"],
                    "ledger_subdimension_mean": result["dimensions"][key]["ledger_subdimension_mean"],
                    "holistic_subdimension_mean": result["dimensions"][key]["holistic_subdimension_mean"],
                    "subdimensions": {
                        unit: {
                            "ledger_score": result["dimensions"][key]["ledger_subdimension_scores"][unit],
                            "holistic_score": result["dimensions"][key]["holistic_subdimension_scores"][unit],
                            "fused_score": result["dimensions"][key]["fused_subdimension_scores"][unit],
                        }
                        for unit in result["dimensions"][key]["fused_subdimension_scores"]
                    },
                    "audited_unit_entry_counts": result["dimensions"][key]["audited_unit_entry_counts"],
                    "formula": result["dimensions"][key]["score_method"],
                } for key in DIMENSIONS
            },
            "plot_behavior_ratings": result["plot_behavior_ratings"],
        })
        report = self.config.output_dir / "00_多Agent评估报告.md"
        self._atomic_write(report, self._render_report(result))
        self._write_json(self.artifact_dir / "errors.json", {"errors": self._errors})
        return report
