"""Optional NMF state lifecycle memory and hybrid retrieval for script generation."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from uuid import uuid4

import jinja2

from drama_local.runtime import atomic_json
from .local_llm import LLM


REPO_ROOT = Path(__file__).resolve().parents[2]
TENCENT_ROOT = Path(
    os.environ.get("DRAMA_TENCENT_ROOT", REPO_ROOT / "vendor/tencent_drama")
).resolve()
EXTRACTION_RULES = """你是连续剧状态维护器。只根据上一状态和本集最终剧本抽取变化，不补写、不预测。
只输出一个合法 JSON 对象，不要 Markdown。
state_delta 必须包含九个字段，每个字段都是字符串数组：character_state、relationship_state、
known_information、unknown_information、confirmed_facts、unresolved_threads、resources_and_evidence、current_goals、timeline。
只记录本集真正新增或改变的状态，不复制上一状态、不复述人物履历、不照抄大纲或作者要求。
角色状态写明主体，关系状态写明双方；每个变化主体最多两项，其余每个字段最多四项，每项尽量不超过35字。
区分角色知道、观众知道和角色误以为；未证实的推测不能记为 confirmed_facts。
relationship_state 只记录实际信任、冲突、依赖及关系边界变化，不记录临时任务分工。
resolved_goals、resolved_unknown、retracted_facts 都必须是字符串数组，逐字引用上一状态条目。
resolved_goals：本集已经得出确定答案或完成具体结果的目标。建立计划、得到线索、部分推进不算完成。
resolved_unknown：本集直接回答了该疑问本身；只有相关线索但没有明确答案时不能清除。
retracted_facts：本集被明确推翻的旧事实，用于清除 confirmed_facts 或 known_information 中的对应条目。
拿不准时保留；已明确解决的小目标不必等主线结局才删除。没有变化或清除时使用空数组。
不输出义务记忆、经验卡、策略、写作建议或评分。
输出格式：{"state_delta":{"character_state":[],"relationship_state":[],"known_information":[],"unknown_information":[],"confirmed_facts":[],"unresolved_threads":[],"resources_and_evidence":[],"current_goals":[],"timeline":[]},"resolved_goals":[],"resolved_unknown":[],"retracted_facts":[]}
"""

INITIAL_STATE_RULES = """你是连续剧初始状态整理器。读取本次生成流程实际产出的世界观、人设、总纲和第一集大纲，建立第一集开场之前已经成立的九类状态。
世界观和人设是已有设定；总纲与第一集大纲是作者计划，只用于辨认开场时点，不能当作已经发生的剧情。
禁止提前写入未来相识、结婚、身份揭露、死亡、资源获取、目标完成或谜底。
特别注意人设中的“从……到……”“后来”“最终”“将会”等描述，后半段可能是未来弧光，不是初始事实。
人物不知道的隐藏身份不得写成该人物已知的信息。状态应写明主体和时间边界，不必填满所有字段。
原始资产的 world_view 和 role_info 已拆成带编号的原文片段 evidence_catalog。每条状态用 evidence_ids 引用支持它的编号，不要抄写、改写或拼接 quote。
大纲只能辅助判断，不能作为已发生事实的证据。引用真实编号不代表事实已经发生，仍需排除来源中的未来弧光。
只输出 JSON：{"entries":[{"field":"character_state","text":"某角色开场前的状态","evidence_ids":["R0001"]}]}。
field 只能是 character_state、relationship_state、known_information、unknown_information、confirmed_facts、unresolved_threads、resources_and_evidence、current_goals、timeline。
最多32条，每条最多80字、引用1至4个编号；同一事实不要跨字段重复。不必穷举外貌、服装和履历，优先记录影响开场行动的状态。
不得新增来源中没有的人物和设定，不输出 initial_state、evidence、quote 或解释文字。
"""


def enabled():
    return os.environ.get("DRAMA_STATE_LIFECYCLE_HYBRID", "0").lower() in {"1", "true", "yes"}


def digest(value):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def initial_evidence_catalog(assets):
    catalog = {}
    for source, prefix in (("world_view", "W"), ("role_info", "R")):
        count = 0
        for match in re.finditer(r"[^。！？；\n]+[。！？；]?", assets[source]):
            quote = match.group().strip()
            if quote:
                count += 1
                evidence_id = f"{prefix}{count:04d}"
                catalog[evidence_id] = {"source": source, "quote": quote}
    return catalog


def assets_from_request(request):
    story = request.generate_data.story_outline
    return {
        "world_view": "\n\n".join(story.world_building) or request.generate_input.world_view,
        "role_info": "\n\n".join(story.role_info) or request.generate_input.role_setting,
        "story_outline": "\n\n".join(story.story_outline),
        "episode_outlines": [episode.to_dict() for season in request.generate_data.episode_outline.seasons
                             for episode in season.episodes],
    }


def preserve_generation_assets(assets, output_root):
    path = Path(output_root) / "generation_assets.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != assets:
        raise ValueError("Generated story assets changed within this run")
    atomic_json(path, assets)


@lru_cache(maxsize=1)
def dependencies():
    package = TENCENT_ROOT / "nmf_v2_7_package"
    sys.path.insert(0, str(package))
    modules = []
    for name, path in (
        ("narrative_memory_feedback_v2_7_self_evolving.scriptpipeline_state_engine", TENCENT_ROOT / "local_pipeline/gated_engine.py"),
        ("scriptpipeline_hybrid_retrieval", TENCENT_ROOT / "local_pipeline/state_retrieval.py"),
    ):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return tuple(modules)


class StateLifecycleMemory:
    def __init__(self, story_id, output_root=None):
        self.engine, self.retrieval = dependencies()
        self.output_root = Path(output_root or os.environ["DRAMA_OUTPUT_DIR"])
        self.root = self.output_root / "state_lifecycle" / story_id
        self.initial_state = {field: [] for field in self.engine.STATE_FIELDS}
        initial_path = self.root / "initial_state.json"
        self.initialized = initial_path.exists()
        self.assets = None
        if self.initialized:
            initial = json.loads(initial_path.read_text(encoding="utf-8"))
            self.assets = json.loads((self.root / "story_assets.json").read_text(encoding="utf-8"))
            if initial["assets_sha256"] != digest(self.assets) or initial["state_sha256"] != digest(initial["initial_state"]):
                raise ValueError("Initial state or its source assets changed")
            self.initial_state = initial["initial_state"]
            self.engine.validate_state(self.initial_state)
        self.records = []
        self.state = self.initial_state
        for index, path in enumerate(sorted((self.root / "updates").glob("E*.json"))):
            if not self.initialized:
                raise ValueError("State updates without an initialized story bible; use the v2 experiment")
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["episode_id"] != f"E{index + 1:02d}" or record["previous_state_hash"] != digest(self.state):
                raise ValueError(f"Broken state checkpoint chain: {path}")
            self.engine.validate_state(record["next_state"])
            if record["next_state_hash"] != digest(record["next_state"]):
                raise ValueError(f"State checkpoint checksum mismatch: {path}")
            self.records.append(record)
            self.state = record["next_state"]

    def validate_initial_result(self, result, assets):
        if not isinstance(result, dict) or set(result) != {"initial_state", "evidence"}:
            raise ValueError("Initial state requires initial_state and evidence")
        state = result["initial_state"]
        self.engine.validate_state(state)
        expected = set()
        for field, values in state.items():
            if not isinstance(values, list) or any(not isinstance(item, str) or not item.strip() for item in values):
                raise ValueError(f"initial_state.{field} must contain strings")
            expected.update((field, item) for item in values)
        if not expected or not isinstance(result["evidence"], list):
            raise ValueError("Initial state must contain supported opening facts")
        supported = set()
        for item in result["evidence"]:
            if not isinstance(item, dict):
                raise ValueError("Initial-state evidence must be objects")
            source, quote = item.get("source"), item.get("quote")
            key = (item.get("field"), item.get("text"))
            if source not in {"world_view", "role_info"} or not isinstance(quote, str) or not quote.strip():
                raise ValueError("Initial-state evidence must cite generated world_view or role_info")
            if quote not in assets[source]:
                raise ValueError(f"Initial-state quote is invalid for {source}: {quote[:100]!r}")
            if key not in expected:
                raise ValueError(f"Initial-state evidence references an invalid state entry: {key!r}")
            supported.add(key)
        if supported != expected:
            raise ValueError("Each initial-state item requires source evidence")

    def resolve_initial_entries(self, result, catalog):
        if not isinstance(result, dict) or set(result) != {"entries"}:
            raise ValueError("Return only entries with field, text and evidence_ids")
        entries = result["entries"]
        if not isinstance(entries, list) or not 1 <= len(entries) <= 32:
            raise ValueError("Initial entries must contain 1 to 32 concise items")
        state = {field: [] for field in self.engine.STATE_FIELDS}
        evidence = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict) or set(entry) != {"field", "text", "evidence_ids"}:
                raise ValueError(f"entries[{index}] requires field, text and evidence_ids only")
            field, text, references = entry["field"], entry["text"], entry["evidence_ids"]
            if not isinstance(field, str) or field not in state:
                raise ValueError(f"entries[{index}] has invalid field {field!r}")
            if not isinstance(text, str) or not text.strip() or len(text) > 80:
                raise ValueError(f"entries[{index}].text must contain 1 to 80 characters")
            if not isinstance(references, list) or not 1 <= len(references) <= 4:
                raise ValueError(f"entries[{index}] requires 1 to 4 evidence_ids")
            if any(not isinstance(reference, str) for reference in references):
                raise ValueError(f"entries[{index}].evidence_ids must contain strings")
            if text in state[field]:
                raise ValueError(f"entries[{index}] duplicates a state entry")
            state[field].append(text)
            for reference in dict.fromkeys(references):
                if reference not in catalog:
                    raise ValueError(f"entries[{index}] has unknown evidence_id {reference!r}; copy an ID from evidence_catalog")
                evidence.append({"field": field, "text": text, "evidence_id": reference, **catalog[reference]})
        return {"initial_state": state, "evidence": evidence}

    async def extract_initial_state(self, ctx, assets):
        catalog = initial_evidence_catalog(assets)
        initial_context = {"evidence_catalog": catalog, "story_outline": assets["story_outline"]}
        initial_context["opening_outline"] = assets["episode_outlines"][0]["content"]
        prompt = INITIAL_STATE_RULES + "\n本次生成的原始资产：\n" + json.dumps(initial_context, ensure_ascii=False)
        attempt_root = self.root / "initialization_attempts" / uuid4().hex
        atomic_json(attempt_root / "evidence_catalog.json", {"assets_sha256": digest(assets), "catalog": catalog})
        client = LLM(model="Qwen3.6-27B", system_prompt="从实际生成资产建立开场状态，未来大纲不等于历史事实。",
                     temperature=0.0, top_p=1.0, top_k=50, max_length=131072,
                     max_new_tokens=4096, template=jinja2.Template("{{ Prompt }}"))
        client.thinking = False
        last_error = ""
        for attempt in range(3):
            client.max_tokens = 4096 if attempt == 0 else 8192
            request_prompt = prompt + (f"\n校验错误：{last_error}。请重新整理并逐条引用证据。" if last_error else "")
            response_text = None
            client.last_output = None
            try:
                response = await client.request(ctx, {"Prompt": request_prompt}, f"initial_state_{attempt + 1}")
                response_text = response.response
                result = self.resolve_initial_entries(self.engine.extract_json_value(response_text), catalog)
                self.validate_initial_result(result, assets)
                atomic_json(attempt_root / f"attempt_{attempt + 1}.json", {"response": response_text, "result": result})
                return result
            except (ValueError, TypeError, KeyError, RuntimeError) as error:
                last_error = str(error)
                atomic_json(attempt_root / f"attempt_{attempt + 1}.json",
                            {"error": last_error, "response": response_text or getattr(client, "last_output", None)})
        raise RuntimeError(f"Initial state extraction failed; rerun to resume: {last_error}")

    async def initialize(self, ctx, assets):
        if any(not isinstance(assets.get(key), str) or assets[key].strip() in {"", "暂无"}
               for key in ("world_view", "role_info", "story_outline")) or not assets.get("episode_outlines"):
            raise ValueError("State initialization requires this run's generated world, roles and outlines")
        if self.initialized:
            if self.assets != assets:
                raise ValueError("State memory is bound to a different generation asset snapshot")
            return
        path = self.root / "story_assets.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != assets:
            raise ValueError("Initialization input changed during retry")
        atomic_json(path, assets)
        result = await self.extract_initial_state(ctx, assets)
        self.validate_initial_result(result, assets)
        atomic_json(self.root / "initial_state.json", {
            **result, "assets_sha256": digest(assets), "state_sha256": digest(result["initial_state"]),
        })
        self.assets = assets
        self.initial_state = result["initial_state"]
        self.state = self.initial_state
        self.initialized = True

    def context(self, episode_id, episode_outline):
        if not self.initialized:
            raise ValueError("Initialize state from generated story assets before retrieval")
        if len(self.records) != episode_id:
            raise ValueError(f"Episode {episode_id + 1} requires {episode_id} state updates, found {len(self.records)}")
        plan = {"episode_id": f"E{episode_id + 1:02d}", "episode_goal": episode_outline,
                "hard_anchors": [], "open_decisions": []}
        view, selection = self.retrieval.select_state(
            self.state, self.initial_state, self.records, plan, "hybrid"
        )
        record = {"episode_id": plan["episode_id"], "previous_state_hash": digest(self.state),
                  "outline_hash": digest(episode_outline), "view": view, "selection": selection}
        path = self.root / "retrieval" / f"{plan['episode_id']}.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != record:
            raise ValueError(f"Retrieval changed during resume: {path}")
        atomic_json(path, record)
        return (
            "\n\n## 状态生命周期记忆（截至上一集的 hybrid 检索视图）\n"
            "以下来自开场设定及此前剧本的状态，用于连续性约束；未检索到不代表不存在，不要把它改写成说明性对白。\n"
            + self.retrieval.compact(view)
        )

    def validate_extraction(self, result):
        if not isinstance(result, dict) or set(result) != {
            "state_delta", "resolved_goals", "resolved_unknown", "retracted_facts"
        }:
            raise ValueError("Expected state_delta and three lifecycle resolution arrays")
        self.engine.validate_state(result["state_delta"])
        for field, values in result["state_delta"].items():
            if not isinstance(values, list) or any(not isinstance(item, str) or not item.strip() for item in values):
                raise ValueError(f"state_delta.{field} must contain nonempty strings")
        for field in ("resolved_goals", "resolved_unknown", "retracted_facts"):
            values = result[field]
            if not isinstance(values, list) or any(not isinstance(item, str) or not item.strip() for item in values):
                raise ValueError(f"{field} must contain strings")
        self.engine.validate_state_delta_semantics(result, {"hard_anchors": []})

    async def extract(self, ctx, episode_id, script):
        client = LLM(model="Qwen3.6-27B", system_prompt="只抽取剧本中实际发生的状态变化。",
                     temperature=0.0, top_p=1.0, top_k=50, max_length=131072,
                     max_new_tokens=4096, template=jinja2.Template("{{ Prompt }}"))
        client.thinking = False
        prompt = (EXTRACTION_RULES + f"\n当前集：第{episode_id + 1}集\n上一完整状态：\n"
                  + json.dumps(self.state, ensure_ascii=False) + "\n本集最终剧本：\n" + script)
        last_error = ""
        for attempt in range(3):
            client.max_tokens = 4096 if attempt == 0 else 8192
            request_prompt = prompt + (f"\n上次校验错误：{last_error}。请重新提取完整 JSON。" if last_error else "")
            try:
                response = await client.request(ctx, {"Prompt": request_prompt, "EpisodeNumber": str(episode_id + 1)},
                                                f"state_E{episode_id + 1:02d}_{attempt + 1}")
                result = self.engine.extract_json_value(response.response)
                self.validate_extraction(result)
                return result
            except (ValueError, TypeError, KeyError, RuntimeError) as error:
                last_error = str(error)
                atomic_json(self.root / "failures" / f"E{episode_id + 1:02d}_{attempt + 1}.json",
                            {"error": last_error, "response": getattr(client, "last_output", None),
                             "script_sha256": digest(script)})
        raise RuntimeError(f"State extraction failed for episode {episode_id + 1}; saved script can be resumed: {last_error}")

    async def update_from_episode(self, ctx, episode_id, episode_outline, script):
        if not self.initialized:
            raise ValueError("Initialize state from generated story assets before episode updates")
        if episode_id < len(self.records):
            record = self.records[episode_id]
            if record["script_sha256"] != digest(script) or record["outline_hash"] != digest(episode_outline):
                raise ValueError("Completed state update belongs to different script or outline")
            return record
        if episode_id != len(self.records):
            raise ValueError("State extraction must run in episode order")
        extraction = await self.extract(ctx, episode_id, script)
        delta = self.engine.normalize_delta(extraction)
        merged = self.engine.merge_state(self.state, delta)
        next_state, removed = self.engine.apply_lifecycle(
            merged, extraction["resolved_goals"], extraction["resolved_unknown"], extraction["retracted_facts"]
        )
        record = {"episode_id": f"E{episode_id + 1:02d}", "script_sha256": digest(script),
                  "outline_hash": digest(episode_outline), "previous_state_hash": digest(self.state),
                  "state_delta": delta, "extraction": extraction, "lifecycle_removed": removed,
                  "next_state": next_state, "next_state_hash": digest(next_state)}
        atomic_json(self.root / "updates" / f"E{episode_id + 1:02d}.json", record)
        self.records.append(record)
        self.state = next_state
        return record
