"""逐集剧本使用的 Future Map 推动记忆与世界状态。"""
from __future__ import annotations

import asyncio
import copy
import fcntl
import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple

import jinja2
from .local_llm import LLM
from drama_local.runtime import logger

from .llm_inference import llm_inference_json
from .utils import get_model_config


FUTURE_MAP_UPDATE_PROMPT = r"""
你是长篇连续剧的 Future Map 推动信息更新器。请根据本集最终剧本，提取本集中具有未来推动能力的事件、细节、线索或伏笔，并将它们关联到只读 Future Map 中真实存在的 FM 节点。不得修改 Future Map 本身。

## FD 类型定义（必须严格区分）
1. event：本集中已经明确发生、直接支撑或实现某个 FM 节点叙事目标的关键事件；它应包含清晰的行动、变化或结果，而不是一句信息或静态描述。
2. detail：本集中已经明确呈现、非常关键且未来不可忘记的叙事细节；它本身未必直接推动情节，但若后续忽略，会造成连续性、人物认知、物品归属、规则或情节逻辑错误。
3. clue：本集中出现的具体线索、证据、异常信息或可追查指向；它指向未来的发现、真相、行动方向或冲突发展，但当前尚未完整揭晓或兑现。
4. foreshadowing：专门为其指向的 FM 节点想要实现的未来叙事情节埋下的伏笔或铺垫；当前只建立期待、条件或预兆，尚未直接实现该 FM 节点。

## 提取与关联原则
1. 只提取剧本中明确发生、明确说出或明确呈现的内容，不猜测，不虚构。
2. 只保留具有较高主线叙事价值的信息；不要随机摘取台词，不要概括无关紧要的气氛描写、路人反应、龙套背景、重复表达或只服务情绪表达的内容。
3. 每个 FD 节点只能选择一种最准确的 kind，不要把同一内容以不同类型重复输出。
4. future_driving_nodes 只保留明确、重要、对未来目标有实质作用，或者能潜在推动、埋下伏笔线索的内容；普通动作、仅主题相似、宽泛呼应或牵强联想不要输出。
5. fm_links.target 只能引用给定 Future Map 中真实存在的 FM 节点；contribution 必须具体解释该 FD 如何服务对应目标。
6. 如果某个事件会推动若干 FM 节点的情节叙事，或者其影响较为长远，可以在同一个 FD 的 fm_links 中与多个 FM 节点产生关联；但每条关联都必须有明确、可解释的依据。
7. contribution_type 只能是 setup|advance|turn|complete|payoff：setup 表示建立前提或铺垫，advance 表示直接推进，turn 表示造成关键转折，complete 表示完成目标，payoff 表示兑现或回收既有伏笔。
8. long_term_importance 表示该 FD 对 Future Map 上未来的长期情节发展、甚至最终目标结局的影响或重要性（0~1）：数值越高，说明该内容越可能在后续多集、未来情节乃至最终结局中持续发挥作用、被回收或产生深远后果；只在本集短暂生效、对未来走向影响甚微的内容给低分。importance 侧重本集内的叙事分量，两者不要混为一谈。

## 当前集
第 {{EpisodeNumber}} 集

## 当前集大纲
{{EpisodeOutline}}

## 前五集摘要（若不足五集则按实际提供）
{{RecentEpisodeSummaries}}

## 只读 Future Map（不得修改；只可引用其中的 FM 节点）
{{FutureMap}}

## 本集最终剧本
{{EpisodeScript}}

## 输出格式
只输出一个 JSON 对象：
{
  "future_driving_nodes": [{
    "temp_id": "D1",
    "kind": "event|detail|clue|foreshadowing",
    "title": "不超过20字",
    "summary": "本集已经明确呈现的关键内容，请用1-2句话完整概述。",
    "participants": ["人物名"],
    "entities": ["线索、物品、组织或地点"],
    "evidence": "剧本中的简短原文证据",
    "importance": 0.0,
    "long_term_importance": 0.0,
    "fm_links": [{
      "target": "FM001",
      "contribution_type": "setup|advance|turn|complete|payoff",
      "contribution": "该内容具体如何服务这个未来叙事目标",
      "confidence": 0.0
    }]
  }]
}
"""

WORLD_STATE_REWRITE_PROMPT = r"""
你是长篇连续剧的世界状态维护器。请根据当前 world state 和本集最终剧本，只输出本集中发生改变的状态（新增或更新），以及需要删除的既有状态；未发生变化的状态不要输出。

## 核心规则
1. 只将当前集中的重要状态变化或者新增状态记录，对后续叙事没有明显推动作用的信息不必记录；记录的状态信息必须明确、不含糊。
2. 只输出本集发生改变的状态：新增状态 state_id 留空（由程序分配）；对既有状态的更新，沿用原 state_id，直接更新 value/status/evidence，不要删除旧状态再新增一个同类状态。
3. 本集没有变化、但未来仍可能有效的既有状态，不要输出，程序会默认保留不变。
4. 所有截至当前集大概率已经过时的前序状态信息，或者本集明确失效、已被取代或不再有叙事价值的状态，都应当写进 deleted_states 并写明理由；更新不算删除，对已有节点的更新不要再写进 deleted_states。
5. 更新某个已有节点时，若其历史状态信息（旧值）仍有保留价值，可在 value 中体现变化过程，例如写成“由（旧值）转变为（新值）”，避免旧信息丢失。
6. 避免同义重复状态；同一主体的当前位置等互斥属性只能保留本集结束时有效的一条。
7. 角色认知必须区分角色知道、观众知道和角色误以为，不能把观众信息赋给角色。
8. 只输出当前快照，不要输出 history 或 previous_value。
9. summary 必须概括本集最终实际发生的关键剧情、关系变化、状态变化与悬念收束/新钩子，便于后续剧本生成直接阅读；不要写评价，不要写“本集讲述了”，不要虚构未发生内容。
10. summary 控制在 120-220 字，尽量信息密度高、指代清晰，可直接作为下一集输入上下文。

## 当前集
第 {{EpisodeNumber}} 集

## 当前集大纲
{{EpisodeOutline}}

## 前五集摘要（若不足五集则按实际提供）
{{RecentEpisodeSummaries}}

## 当前集开始前的 world state（只读参考，用于判断哪些状态发生了变化）
{{CompleteWorldState}}

## 本集最终剧本
{{EpisodeScript}}

## 输出格式
只输出一个 JSON 对象：
{
  "summary": "120-220字的本集最终摘要",
  "world_state": [{
    "state_id": "更新沿用原S编号；新增留空字符串",
    "category": "character_knowledge|character_status|relationship|object|clue|identity|goal|location|world_fact",
    "entity": "主体", "attribute": "状态属性", "value": "本集结束时的当前值",
    "status": "active|resolved|lost|destroyed|unknown",
    "evidence": "支持新增或变化的本集证据",
    "confidence": 0.0
  }],
  "deleted_states": [{"state_id": "真正删除的已有S编号", "reason": "为何已失效且不应继续保留"}]
}

注意：world_state 里只写本集新增或更新的状态；未写出的既有状态默认保留不变；只有真正删除的状态才写进 deleted_states，更新不要在 deleted_states 里重复声明。
"""

_ALLOWED_STATE_CATEGORIES = {
    "character_knowledge", "character_status", "relationship", "object",
    "clue", "identity", "goal", "location", "world_fact",
}
_ALLOWED_STATE_STATUSES = {"active", "resolved", "lost", "destroyed", "unknown"}
_ALLOWED_FUTURE_NODE_KINDS = {"event", "detail", "clue", "foreshadowing"}
_ALLOWED_FUTURE_CONTRIBUTION_TYPES = {"setup", "advance", "turn", "complete", "payoff"}
_ATTRIBUTE_ALIASES = {
    "当前持有人": "持有人", "当前持有者": "持有人", "持有者": "持有人",
    "所处位置": "位置", "当前地点": "位置", "所在地": "位置",
    "认知": "知道的信息", "已知信息": "知道的信息", "知情状态": "知道的信息",
    "真实身份": "身份", "当前身份": "身份",
    "人物关系": "关系", "当前关系": "关系",
}


def _canonical_attribute(value: Any) -> str:
    attribute = re.sub(r"[\s：:，,。]+", "", str(value or ""))
    attribute = _ATTRIBUTE_ALIASES.get(attribute, attribute)
    # 去掉常见助词，减少“对江别月的认知/对江别月认知”这类措辞抖动导致的 key 分裂。
    attribute = re.sub(r"[的之]+", "", attribute)
    return attribute


def _safe_name(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_.\-\u4e00-\u9fff]+", "_", value or "story")
    return value[:120] or "story"


def _clip(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _number(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _normalize(text: Any) -> str:
    return re.sub(r"\s+", "", str(text or "").lower())


def _atomic_json_dump(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".memory_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2, default=str)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


class NarrativeMemory:
    """Future Map 推动记忆、世界状态及其检索的本地持久化门面。"""

    def __init__(self, story_id: str):
        self.story_id = story_id
        output_root = os.environ.get("DRAMA_OUTPUT_DIR") or os.path.join(os.getcwd(), "output")
        self.output_root = os.path.abspath(output_root)
        configured_root = os.environ.get("DRAMA_MEMORY_DIR")
        self.root = os.path.abspath(configured_root or os.path.join(self.output_root, "narrative_memory"))
        self.story_dir = os.path.join(self.root, _safe_name(story_id))
        self.state_path = os.path.join(self.story_dir, "memory_state.json")
        self.future_map_path = os.path.join(self.output_root, "04_future_map", "future_map.json")
        self.episode_contributions_path = os.path.join(
            self.output_root, "04_future_map", "episode_contributions.json"
        )
        self.future_driving_path = os.path.join(self.output_root, "04_future_map", "future_driving_graph.json")
        self.future_driving_updates_dir = os.path.join(
            self.output_root, "04_future_map", "future_driving_updates"
        )
        self.data = self._load()

    def _load_future_map(self) -> Tuple[str, Set[str], Dict[str, Any]]:
        """读取只读 Future Map；缺失或损坏时关闭未来关联提取，不影响 World State。"""
        try:
            with open(self.future_map_path, "r", encoding="utf-8") as file:
                future_map = json.load(file)
            raw_nodes = _as_list(future_map.get("nodes")) if isinstance(future_map, dict) else []
            nodes = []
            valid_ids: Set[str] = set()
            for raw in raw_nodes:
                if not isinstance(raw, dict):
                    continue
                node_id = str(raw.get("node_id", "")).strip().upper()
                if not re.fullmatch(r"FM\d+", node_id) or node_id in valid_ids:
                    continue
                valid_ids.add(node_id)
                nodes.append({
                    "node_id": node_id,
                    "parent_id": raw.get("parent_id"),
                    "depth": raw.get("depth"),
                    "goal": _clip(raw.get("goal"), 500),
                    "requirement": _clip(raw.get("requirement"), 500),
                })
            if not nodes:
                raise ValueError("nodes 为空")
            compact = {
                "map_type": future_map.get("map_type", "backward_requirement_tree"),
                "root_id": future_map.get("root_id", "FM001"),
                "nodes": nodes,
            }
            return json.dumps(compact, ensure_ascii=False, indent=2), valid_ids, compact
        except (OSError, ValueError, json.JSONDecodeError) as error:
            logger.error(f"Future Map 不可用，将跳过 FD 关联更新并继续更新 World State：{error}")
            return "暂无可用 Future Map；future_driving_nodes 必须输出空数组。", set(), {}

    def _load_future_driving_graph(self) -> Dict[str, Any]:
        empty = {
            "schema_version": 1,
            "story_id": self.story_id,
            "future_map_path": self.future_map_path,
            "nodes": [],
            "edges": [],
            "last_updated_episode": -1,
            "updated_at": "",
        }
        if not os.path.exists(self.future_driving_path):
            return empty
        try:
            with open(self.future_driving_path, "r", encoding="utf-8") as file:
                loaded = json.load(file)
            if not isinstance(loaded, dict):
                return empty
            if loaded.get("story_id") not in (None, self.story_id):
                return empty
            loaded["nodes"] = _as_list(loaded.get("nodes"))
            loaded["edges"] = _as_list(loaded.get("edges"))
            loaded["schema_version"] = 1
            loaded["story_id"] = self.story_id
            loaded["future_map_path"] = self.future_map_path
            return loaded
        except (OSError, ValueError, json.JSONDecodeError) as error:
            logger.error(f"未来推动图读取失败，将从空图继续：{error}")
            return empty

    def _load_episode_contributions(self) -> Dict[int, List[Dict[str, Any]]]:
        """读取按集 Future Map 贡献标注，返回内部使用的 0-based episode_id 索引。"""
        if not os.path.exists(self.episode_contributions_path):
            return {}
        try:
            with open(self.episode_contributions_path, "r", encoding="utf-8") as file:
                loaded = json.load(file)
            episodes = _as_list(loaded.get("episodes")) if isinstance(loaded, dict) else []
            contributions_by_episode: Dict[int, List[Dict[str, Any]]] = {}
            for episode in episodes:
                if not isinstance(episode, dict):
                    continue
                try:
                    zero_based_episode_id = int(episode.get("episode_id")) - 1
                except (TypeError, ValueError):
                    continue
                if zero_based_episode_id < 0:
                    continue
                cleaned_contributions = []
                seen_node_ids: Set[str] = set()
                for contribution in _as_list(episode.get("contributions")):
                    if not isinstance(contribution, dict):
                        continue
                    node_id = str(contribution.get("node_id", "")).strip().upper()
                    if not re.fullmatch(r"FM\d+", node_id) or node_id in seen_node_ids:
                        continue
                    seen_node_ids.add(node_id)
                    cleaned_contributions.append({
                        "node_id": node_id,
                        "contribution_type": str(contribution.get("contribution_type", "")).strip(),
                        "importance": str(contribution.get("importance", "")).strip(),
                        "contribution": _clip(contribution.get("contribution"), 240),
                    })
                contributions_by_episode[zero_based_episode_id] = cleaned_contributions
            return contributions_by_episode
        except (OSError, ValueError, json.JSONDecodeError) as error:
            logger.error(f"Episode contributions 读取失败，将跳过 Future Map 定向检索：{error}")
            return {}

    @staticmethod
    def _collect_future_map_subtree(root_id: str, children_by_parent: Dict[str, List[str]]) -> Set[str]:
        subtree_ids: Set[str] = set()
        stack = [root_id]
        while stack:
            current = stack.pop()
            if current in subtree_ids:
                continue
            subtree_ids.add(current)
            stack.extend(children_by_parent.get(current, []))
        return subtree_ids

    def _empty(self) -> Dict[str, Any]:
        return {
            "schema_version": 4,
            "story_id": self.story_id,
            "last_updated_episode": -1,
            "world_state": [],
            "world_state_archive": [],
            "episode_summaries": [],
            "next_state_index": 1,
            "updated_at": "",
        }

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.state_path):
            return self._empty()
        try:
            with open(self.state_path, "r", encoding="utf-8") as file:
                loaded = json.load(file)
            if not isinstance(loaded, dict) or loaded.get("story_id") != self.story_id:
                raise ValueError("memory story_id 不匹配")
            for key in ("world_state", "world_state_archive"):
                if not isinstance(loaded.get(key), list):
                    loaded[key] = []
            loaded.pop("events", None)
            loaded.pop("edges", None)
            loaded["schema_version"] = 4
            # 为既有世界状态补齐稳定编号，供后续更新时精确指认，避免仅靠措辞归并。
            next_state_index = 1
            for state in loaded["world_state"]:
                if not isinstance(state, dict):
                    continue
                existing_id = str(state.get("state_id") or "")
                match = re.match(r"^S(\d+)$", existing_id)
                if match:
                    next_state_index = max(next_state_index, int(match.group(1)) + 1)
                else:
                    state["state_id"] = f"S{next_state_index:03d}"
                    next_state_index += 1
            for archived in loaded["world_state_archive"]:
                if not isinstance(archived, dict):
                    continue
                archived_state = archived.get("state") if isinstance(archived.get("state"), dict) else {}
                state_id = str(archived_state.get("state_id", ""))
                match = re.match(r"^S(\d+)$", state_id)
                if match:
                    next_state_index = max(next_state_index, int(match.group(1)) + 1)
            loaded["next_state_index"] = max(
                next_state_index, int(loaded.get("next_state_index", 1) or 1)
            )
            return loaded
        except Exception as error:
            backup = self.state_path + ".broken"
            try:
                os.replace(self.state_path, backup)
            except OSError:
                pass
            logger.error(f"叙事记忆读取失败，已创建空记忆: {error}")
            return self._empty()

    @contextmanager
    def _update_lock(self):
        os.makedirs(self.story_dir, exist_ok=True)
        lock_path = os.path.join(self.story_dir, ".memory.lock")
        with open(lock_path, "a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _save(self) -> None:
        self.data["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _atomic_json_dump(self.state_path, self.data)
        try:
            _atomic_json_dump(os.path.join(self.story_dir, "world_state.json"), {
                "story_id": self.story_id,
                "states": self.data["world_state"],
            })
            _atomic_json_dump(os.path.join(self.story_dir, "episode_summaries.json"), {
                "story_id": self.story_id,
                "summaries": self.data.get("episode_summaries", []),
            })
        except OSError as error:
            logger.error(f"叙事记忆主状态已保存，但可读投影文件刷新失败：{error}")

    def prepare_for_episode(self, episode_id: int, persist: bool = True) -> None:
        """重生成旧集时截断该集及之后的派生记忆。"""
        last_episode = int(self.data.get("last_updated_episode", -1))
        if episode_id > last_episode:
            return
        self.data["episode_summaries"] = [
            item for item in _as_list(self.data.get("episode_summaries"))
            if int(item.get("episode_id", -1)) < episode_id
        ]
        retained_states = []
        retained_archive = []
        state_candidates = list(self.data["world_state"])
        for archived in self.data.get("world_state_archive", []):
            if not isinstance(archived, dict) or not isinstance(archived.get("state"), dict):
                continue
            deletion_episode = int(archived.get("deleted_episode", -1))
            if deletion_episode >= episode_id:
                state_candidates.append(archived["state"])
            else:
                retained_archive.append(archived)
        seen_state_ids = set()
        for state in state_candidates:
            state_id = state.get("state_id")
            if state_id in seen_state_ids:
                continue
            history = [item for item in _as_list(state.get("history")) if int(item.get("episode_id", -1)) < episode_id]
            if not history:
                continue
            latest = history[-1]
            restored = copy.deepcopy(state)
            restored.update({
                "value": latest.get("value", ""),
                "status": latest.get("status", "active"),
                "episode_id": latest.get("episode_id", -1),
                "evidence": latest.get("evidence", ""),
                "confidence": latest.get("confidence", 0.5),
                "history": history,
            })
            retained_states.append(restored)
            seen_state_ids.add(state_id)
        self.data["world_state"] = retained_states
        self.data["world_state_archive"] = retained_archive
        self.data["last_updated_episode"] = episode_id - 1
        if persist:
            self._save()

    def _world_state_before(self, episode_id: int) -> List[Dict[str, Any]]:
        """返回当前集开始前的世界状态快照，不修改持久化记忆。"""
        snapshot = []
        states = list(self.data["world_state"])
        for archived in self.data.get("world_state_archive", []):
            if not isinstance(archived, dict) or not isinstance(archived.get("state"), dict):
                continue
            if int(archived.get("deleted_episode", -1)) >= episode_id:
                states.append(archived["state"])
        seen_state_ids = set()
        for state in states:
            state_id = state.get("state_id")
            if state_id in seen_state_ids:
                continue
            history = [
                item for item in _as_list(state.get("history"))
                if int(item.get("episode_id", -1)) < episode_id
            ]
            if not history:
                continue
            latest = history[-1]
            restored = copy.deepcopy(state)
            restored.update(latest)
            restored["history"] = history
            snapshot.append(restored)
            seen_state_ids.add(state_id)
        return snapshot

    def retrieve(self, episode_outline: str, episode_id: int) -> Tuple[str, Dict[str, Any]]:
        """按当前集直接指向的 FM 节点及其子树，分类检索此前已发生的 FD。"""
        # 默认保持原有 top 10；独立实验可设置为 0，保留全部子树非伏笔记忆。
        subtree_limit = int(os.environ.get("DRAMA_SUBTREE_MEMORY_LIMIT", "10"))
        if subtree_limit < 0:
            raise ValueError("DRAMA_SUBTREE_MEMORY_LIMIT 必须是非负整数，0 表示不限条数")
        _, _, future_map_snapshot = self._load_future_map()
        contributions_by_episode = self._load_episode_contributions()
        future_driving_graph = self._load_future_driving_graph()

        node_by_id: Dict[str, Dict[str, Any]] = {}
        children_by_parent: Dict[str, List[str]] = {}
        for raw_node in _as_list(future_map_snapshot.get("nodes")):
            if not isinstance(raw_node, dict):
                continue
            node_id = str(raw_node.get("node_id", "")).strip().upper()
            if not re.fullmatch(r"FM\d+", node_id):
                continue
            node_by_id[node_id] = raw_node
        for node_id, raw_node in node_by_id.items():
            parent_id = str(raw_node.get("parent_id") or "").strip().upper()
            if parent_id and parent_id in node_by_id:
                children_by_parent.setdefault(parent_id, []).append(node_id)

        direct_node_ids: List[str] = []
        for contribution in contributions_by_episode.get(episode_id, []):
            node_id = str(contribution.get("node_id", "")).strip().upper()
            if node_id in node_by_id and node_id not in direct_node_ids:
                direct_node_ids.append(node_id)
        direct_node_id_set = set(direct_node_ids)

        descendant_node_ids: Set[str] = set()
        for node_id in direct_node_ids:
            descendant_node_ids.update(self._collect_future_map_subtree(node_id, children_by_parent))
        descendant_node_ids.difference_update(direct_node_id_set)

        direct_items: Dict[str, List[Dict[str, Any]]] = {
            "event": [], "detail": [], "clue": [], "foreshadowing": [],
        }
        subtree_items: List[Dict[str, Any]] = []
        seen_direct: Set[str] = set()
        seen_subtree: Set[str] = set()
        future_nodes = sorted(
            (node for node in _as_list(future_driving_graph.get("nodes")) if isinstance(node, dict)),
            key=lambda node: (int(node.get("episode_id", -1)), str(node.get("id", ""))),
        )
        for future_node in future_nodes:
            node_episode_id = int(future_node.get("episode_id", -1))
            if node_episode_id < 0 or node_episode_id >= episode_id:
                continue
            kind = str(future_node.get("kind", "event")).strip().lower()
            if kind not in _ALLOWED_FUTURE_NODE_KINDS:
                kind = "event"
            linked_targets = {
                str(link.get("target", "")).strip().upper()
                for link in _as_list(future_node.get("fm_links"))
                if isinstance(link, dict)
            }
            item = {
                "id": str(future_node.get("id", "")),
                "episode_id": node_episode_id,
                "kind": kind,
                "summary": _clip(future_node.get("summary") or future_node.get("title"), 220),
                "long_term_importance": _number(future_node.get("long_term_importance"), 0.0),
            }
            if not item["summary"]:
                continue
            item_id = item["id"] or f"{node_episode_id}:{item['summary']}"
            if linked_targets & direct_node_id_set:
                if item_id not in seen_direct:
                    direct_items[kind].append(item)
                    seen_direct.add(item_id)
                continue
            if kind != "foreshadowing" and linked_targets & descendant_node_ids and item_id not in seen_subtree:
                subtree_items.append(item)
                seen_subtree.add(item_id)

        # 仅限制子树上的 event/detail/clue，直接关联的记忆仍全部保留。
        if subtree_limit and len(subtree_items) > subtree_limit:
            subtree_items = sorted(
                subtree_items,
                key=lambda item: _number(item.get("long_term_importance"), 0.0),
                reverse=True,
            )[:subtree_limit]

        payload = {
            "episode_id": episode_id,
            "query": episode_outline,
            "direct_node_ids": direct_node_ids,
            "descendant_node_ids": sorted(descendant_node_ids),
            "direct_items": direct_items,
            "subtree_items": subtree_items,
            "subtree_memory_limit": subtree_limit,
        }
        context = self._format_context(payload)
        payload["formatted_context"] = context
        _atomic_json_dump(
            os.path.join(self.story_dir, "retrieval", f"episode_{episode_id + 1:03d}.json"),
            payload,
        )
        return context, payload

    def _format_context(self, payload: Dict[str, Any]) -> str:
        direct_items = payload.get("direct_items", {})
        subtree_items = _as_list(payload.get("subtree_items"))
        sections = [
            ("与当前集最为直接相关的event", _as_list(direct_items.get("event"))),
            ("detail", _as_list(direct_items.get("detail"))),
            ("clue", _as_list(direct_items.get("clue"))),
            ("与当前集拥有类似叙事目标的伏笔（需要尽量收束这些伏笔）", _as_list(direct_items.get("foreshadowing"))),
            ("其他可能具有叙事相关性的事件", subtree_items),
        ]
        lines: List[str] = []
        for title, items in sections:
            lines.append(f"{title}：")
            if items:
                for item in items:
                    lines.append(
                        f"- 第{int(item.get('episode_id', -1)) + 1}集："
                        f"{_clip(item.get('summary', ''), 220)}"
                    )
            else:
                lines.append("- 暂无")
        return "\n".join(lines)

    @staticmethod
    def _normalize_state_id(value: Any) -> str:
        """提取模型返回的状态编号，容忍展示格式（如 [S001]）。"""
        text = str(value or "").strip()
        match = re.search(r"\b(S\d+)\b", text, re.IGNORECASE)
        return match.group(1).upper() if match else ""

    def _next_state_id(self) -> str:
        """为新建状态分配下一个稳定编号。"""
        max_index = 0
        for state in self.data["world_state"]:
            match = re.match(r"^S(\d+)$", str(state.get("state_id", "")))
            if match:
                max_index = max(max_index, int(match.group(1)))
        return f"S{max_index + 1:03d}"

    def _format_complete_world_state_for_update(self, episode_id: int) -> str:
        """向状态维护模型提供全量当前快照及最后更新时间。"""
        states = sorted(
            self._world_state_before(episode_id),
            key=lambda state: (state.get("category", ""), state.get("entity", ""), state.get("attribute", "")),
        )
        lines = [
            "以下是当前集开始前的全部 world state，仅供只读参考。只输出本集发生改变的状态（新增或更新）和真正删除的状态，其余默认保留。",
            "last_updated_episode 表示该状态最后一次发生变化的集数，可用于谨慎判断过时信息。",
        ]
        if states:
            for state in states:
                lines.append(json.dumps({
                    "state_id": state.get("state_id", ""),
                    "category": state.get("category", "world_fact"),
                    "entity": state.get("entity", ""),
                    "attribute": state.get("attribute", ""),
                    "value": state.get("value", ""),
                    "status": state.get("status", "active"),
                    "last_updated_episode": int(state.get("episode_id", -1)) + 1,
                    "evidence": state.get("evidence", ""),
                    "confidence": _number(state.get("confidence"), 0.6),
                }, ensure_ascii=False))
        else:
            lines.append("暂无世界状态")
        return "\n".join(lines)

    def _format_recent_episode_summaries(self, episode_id: int, limit: int = 5) -> str:
        """返回当前集之前最近若干集 summary，供剧本生成与记忆更新共用。"""
        summaries = [
            item for item in _as_list(self.data.get("episode_summaries"))
            if int(item.get("episode_id", -1)) < episode_id and str(item.get("summary", "")).strip()
        ]
        summaries.sort(key=lambda item: int(item.get("episode_id", -1)))
        selected = summaries[-limit:]
        if not selected:
            return "暂无前情摘要"
        lines = []
        for item in selected:
            lines.append(
                f"- 第{int(item.get('episode_id', -1)) + 1}集：{_clip(item.get('summary', ''), 400)}"
            )
        return "\n".join(lines)

    def _upsert_episode_summary(self, episode_id: int, summary: Any) -> str:
        """写入或覆盖本集 summary，并返回清洗后的文本。"""
        cleaned = _clip(summary, 500)
        filtered = [
            item for item in _as_list(self.data.get("episode_summaries"))
            if int(item.get("episode_id", -1)) != episode_id
        ]
        if cleaned:
            filtered.append({"episode_id": episode_id, "summary": cleaned})
            filtered.sort(key=lambda item: int(item.get("episode_id", -1)))
        self.data["episode_summaries"] = filtered
        return cleaned

    def _build_update_service(self, prompt: str) -> LLM:
        auxiliary_config = get_model_config("auxiliary_model")
        return LLM(
            model=auxiliary_config["model"],
            system_prompt=auxiliary_config["system_prompt"],
            temperature=min(float(auxiliary_config.get("temperature", 0.4)), 0.3),
            top_p=auxiliary_config["top_p"],
            top_k=auxiliary_config["top_k"],
            max_length=auxiliary_config["max_length"],
            max_new_tokens=auxiliary_config["max_new_tokens"],
            template=jinja2.Template(prompt),
        )

    async def update_from_episode(
        self,
        ctx,
        episode_id: int,
        episode_outline: str,
        final_script: str,
    ) -> Dict[str, Any]:
        """并发更新 Future Map 推动关联和全量 World State，再在锁内原子提交。"""
        complete_world_state = self._format_complete_world_state_for_update(episode_id)
        recent_episode_summaries = self._format_recent_episode_summaries(episode_id)
        future_map_text, valid_future_map_ids, future_map_snapshot = self._load_future_map()
        future_service = self._build_update_service(FUTURE_MAP_UPDATE_PROMPT)
        state_service = self._build_update_service(WORLD_STATE_REWRITE_PROMPT)

        async def update_future_map() -> Dict[str, Any]:
            try:
                result = await llm_inference_json(
                    params={
                        "EpisodeNumber": str(episode_id + 1),
                        "EpisodeOutline": episode_outline,
                        "RecentEpisodeSummaries": recent_episode_summaries,
                        "FutureMap": future_map_text,
                        "EpisodeScript": final_script,
                    },
                    llm_service=future_service,
                    task_name=f"更新第{episode_id + 1}集Future Map推动关联",
                    ctx=ctx,
                )
                return result if isinstance(result, dict) else {}
            except Exception as error:
                logger.error_context(ctx, f"第{episode_id + 1}集Future Map推动关联更新失败：{error}")
                return {}

        async def rewrite_world_state() -> Tuple[Dict[str, Any], bool]:
            try:
                result = await llm_inference_json(
                    params={
                        "EpisodeNumber": str(episode_id + 1),
                        "EpisodeOutline": episode_outline,
                        "RecentEpisodeSummaries": recent_episode_summaries,
                        "CompleteWorldState": complete_world_state,
                        "EpisodeScript": final_script,
                    },
                    llm_service=state_service,
                    task_name=f"重写第{episode_id + 1}集完整世界状态",
                    ctx=ctx,
                )
                if (
                    isinstance(result, dict)
                    and isinstance(result.get("world_state"), list)
                    and isinstance(result.get("deleted_states"), list)
                ):
                    return result, True
                logger.error_context(ctx, f"第{episode_id + 1}集完整世界状态格式无效，保留原状态")
            except Exception as error:
                logger.error_context(ctx, f"第{episode_id + 1}集完整世界状态重写失败，保留原状态：{error}")
            return {}, False

        future_extraction, (state_rewrite, state_rewrite_succeeded) = await asyncio.gather(
            update_future_map(), rewrite_world_state()
        )

        # 两个模型都结束后，在同一文件锁中重载、回滚并原子提交，避免部分写入。
        with self._update_lock():
            self.data = self._load()
            original_world_state = copy.deepcopy(self.data["world_state"])
            original_world_state_archive = copy.deepcopy(self.data.get("world_state_archive", []))
            original_next_state_index = int(self.data.get("next_state_index", 1) or 1)
            self.prepare_for_episode(episode_id, persist=False)
            future_driving_graph = self._load_future_driving_graph()
            if valid_future_map_ids:
                retained_future_nodes = [
                    node for node in future_driving_graph["nodes"]
                    if int(node.get("episode_id", -1)) < episode_id
                ]
                retained_future_ids = {node.get("id") for node in retained_future_nodes}
                future_driving_graph["nodes"] = retained_future_nodes
                future_driving_graph["edges"] = [
                    edge for edge in future_driving_graph["edges"]
                    if edge.get("source") in retained_future_ids
                ]
                future_applied = self._apply_future_driving_extraction(
                    episode_id=episode_id,
                    extraction=future_extraction,
                    valid_future_map_ids=valid_future_map_ids,
                    future_driving_graph=future_driving_graph,
                )
            else:
                future_applied = {"nodes": [], "edges": [], "preserved_without_future_map": True}
            if state_rewrite_succeeded:
                state_applied = self._replace_world_state(
                    episode_id,
                    state_rewrite.get("world_state", []),
                    state_rewrite.get("deleted_states", []),
                )
                if state_applied.get("preserved_after_invalid_rewrite"):
                    self.data["world_state"] = original_world_state
                    self.data["world_state_archive"] = original_world_state_archive
                    self.data["next_state_index"] = original_next_state_index
                    state_applied["world_state"] = copy.deepcopy(original_world_state)
            else:
                self.data["world_state"] = original_world_state
                self.data["world_state_archive"] = original_world_state_archive
                self.data["next_state_index"] = original_next_state_index
                state_applied = {
                    "world_state": copy.deepcopy(original_world_state),
                    "created": [], "updated": [], "deleted_state_ids": [], "preserved_after_failure": True,
                }
            episode_summary = self._upsert_episode_summary(episode_id, state_rewrite.get("summary"))
            applied = {
                "future_driving_nodes": future_applied["nodes"],
                "future_map_edges": future_applied["edges"],
                "episode_summary": episode_summary,
                "world_state_updates": state_applied.get("created", []) + state_applied.get("updated", []),
                "world_state_rewrite": state_applied,
            }
            self.data["last_updated_episode"] = max(int(self.data.get("last_updated_episode", -1)), episode_id)
            self._save()
            try:
                if valid_future_map_ids:
                    future_driving_graph["last_updated_episode"] = episode_id
                    future_driving_graph["updated_at"] = datetime.now().isoformat(timespec="seconds")
                    future_driving_graph["future_map_root_id"] = future_map_snapshot.get("root_id")
                _atomic_json_dump(self.future_driving_path, future_driving_graph)
                _atomic_json_dump(
                    os.path.join(
                        self.future_driving_updates_dir,
                        f"episode_{episode_id + 1:03d}.json",
                    ),
                    {
                        "episode_id": episode_id,
                        "raw_future_driving_nodes": _as_list(future_extraction.get("future_driving_nodes")),
                        "applied": future_applied,
                    },
                )
            except OSError as error:
                logger.error_context(
                    ctx,
                    f"第{episode_id + 1}集未来推动图保存失败，但 World State 已正常保存：{error}",
                )
            _atomic_json_dump(
                os.path.join(self.story_dir, "updates", f"episode_{episode_id + 1:03d}.json"),
                {
                    "raw_future_map_update": future_extraction,
                    "raw_world_state_rewrite": state_rewrite,
                    "world_state_rewrite_succeeded": state_rewrite_succeeded,
                    "applied": applied,
                },
            )
        return applied

    def _apply_future_driving_extraction(
        self,
        episode_id: int,
        extraction: Dict[str, Any],
        valid_future_map_ids: Set[str],
        future_driving_graph: Dict[str, Any],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """校验 Future Driving 节点并追加到独立关联图，不修改 Future Map。"""
        if not valid_future_map_ids:
            return {"nodes": [], "edges": []}

        new_nodes: List[Dict[str, Any]] = []
        new_edges: List[Dict[str, Any]] = []
        for index, raw in enumerate(_as_list(extraction.get("future_driving_nodes")), start=1):
            if not isinstance(raw, dict):
                continue
            summary = _clip(raw.get("summary"), 500)
            title = _clip(raw.get("title"), 40)
            if not summary or not title:
                continue
            kind = str(raw.get("kind", "event")).strip().lower()
            if kind not in _ALLOWED_FUTURE_NODE_KINDS:
                kind = "event"

            normalized_links: List[Dict[str, Any]] = []
            seen_targets: Set[str] = set()
            for raw_link in _as_list(raw.get("fm_links")):
                if not isinstance(raw_link, dict):
                    continue
                match = re.search(r"\b(FM\d+)\b", str(raw_link.get("target", "")), re.IGNORECASE)
                target = match.group(1).upper() if match else ""
                contribution_type = str(raw_link.get("contribution_type", "")).strip().lower()
                contribution = _clip(raw_link.get("contribution"), 500)
                if (
                    target not in valid_future_map_ids
                    or target in seen_targets
                    or contribution_type not in _ALLOWED_FUTURE_CONTRIBUTION_TYPES
                    or not contribution
                ):
                    continue
                seen_targets.add(target)
                normalized_links.append({
                    "target": target,
                    "contribution_type": contribution_type,
                    "contribution": contribution,
                    "confidence": _number(raw_link.get("confidence"), 0.6),
                })
            if not normalized_links:
                continue

            node_id = f"FD{episode_id + 1:03d}_{len(new_nodes) + 1:02d}"
            node = {
                "id": node_id,
                "episode_id": episode_id,
                "kind": kind,
                "title": title,
                "summary": summary,
                "participants": [_clip(item, 40) for item in _as_list(raw.get("participants"))[:12]],
                "entities": [_clip(item, 60) for item in _as_list(raw.get("entities"))[:16]],
                "evidence": _clip(raw.get("evidence"), 240),
                "importance": _number(raw.get("importance"), 0.7),
                "long_term_importance": _number(raw.get("long_term_importance"), 0.0),
                "fm_links": normalized_links,
            }
            new_nodes.append(node)
            for link in normalized_links:
                new_edges.append({
                    "source": node_id,
                    "target": link["target"],
                    "type": "serves",
                    "contribution_type": link["contribution_type"],
                    "contribution": link["contribution"],
                    "confidence": link["confidence"],
                    "created_episode": episode_id,
                })

        future_driving_graph["nodes"].extend(new_nodes)
        future_driving_graph["edges"].extend(new_edges)
        return {"nodes": new_nodes, "edges": new_edges}

    def _replace_world_state(
        self,
        episode_id: int,
        raw_states: List[Any],
        raw_deletions: List[Any],
    ) -> Dict[str, Any]:
        """校验模型给出的状态变化（增量），继承历史与编号后增量应用到 world state。"""
        old_states = copy.deepcopy(self.data["world_state"])
        old_by_id = {
            state.get("state_id"): state for state in old_states if state.get("state_id")
        }
        old_by_key = {state.get("key"): state for state in old_states if state.get("key")}
        next_state_index = max(1, int(self.data.get("next_state_index", 1) or 1))
        declared_deletions: Dict[str, str] = {}
        for raw in raw_deletions:
            if not isinstance(raw, dict):
                continue
            state_id = self._normalize_state_id(raw.get("state_id"))
            reason = _clip(raw.get("reason"), 240)
            if state_id in old_by_id and reason:
                declared_deletions[state_id] = reason

        replacement_by_key: Dict[str, Dict[str, Any]] = {}
        key_order: List[str] = []
        created_ids: Set[str] = set()
        updated_ids: Set[str] = set()
        invalid_response = False

        for raw in raw_states:
            if not isinstance(raw, dict):
                invalid_response = True
                continue
            requested_id = self._normalize_state_id(raw.get("state_id"))
            old_state = old_by_id.get(requested_id)
            category = str(raw.get("category", "world_fact"))
            if category not in _ALLOWED_STATE_CATEGORIES:
                category = "world_fact"
            entity = _clip(raw.get("entity"), 80)
            attribute = _clip(_canonical_attribute(raw.get("attribute")), 100)
            value = _clip(raw.get("value"), 500)
            if not (entity and attribute and value):
                invalid_response = True
                continue
            proposed_key = f"{category}:{_normalize(entity)}:{_normalize(attribute)}"
            if old_state is None:
                old_state = old_by_key.get(proposed_key)
            if old_state is not None:
                state_id = str(old_state.get("state_id", ""))
                key = str(old_state.get("key", proposed_key))
                category = str(old_state.get("category", category))
                entity = str(old_state.get("entity", entity))
                attribute = str(old_state.get("attribute", attribute))
            else:
                state_id = f"S{next_state_index:03d}"
                next_state_index += 1
                key = proposed_key

            status = str(raw.get("status") or "active")
            if status not in _ALLOWED_STATE_STATUSES:
                status = "active"
            evidence = _clip(raw.get("evidence"), 240)
            confidence = _number(raw.get("confidence"), 0.6)

            if old_state is not None:
                unchanged = (
                    value == old_state.get("value")
                    and status == old_state.get("status", "active")
                    and evidence == str(old_state.get("evidence", ""))
                )
                if unchanged:
                    state = copy.deepcopy(old_state)
                else:
                    state = copy.deepcopy(old_state)
                    history_item = {
                        "episode_id": episode_id,
                        "value": value,
                        "previous_value": old_state.get("value", ""),
                        "status": status,
                        "evidence": evidence,
                        "confidence": confidence,
                    }
                    state.update(history_item)
                    state.setdefault("history", []).append(history_item)
                    updated_ids.add(state_id)
            else:
                history_item = {
                    "episode_id": episode_id,
                    "value": value,
                    "previous_value": "",
                    "status": status,
                    "evidence": evidence,
                    "confidence": confidence,
                }
                state = {
                    "key": key,
                    "state_id": state_id,
                    "category": category,
                    "entity": entity,
                    "attribute": attribute,
                    "history": [history_item],
                    **history_item,
                }
                created_ids.add(state_id)

            # 重复 key 时保留靠后的（其值更可能是最新更新）。
            if key not in replacement_by_key:
                key_order.append(key)
            replacement_by_key[key] = state

        changes = [replacement_by_key[key] for key in key_order]
        created = [copy.deepcopy(state) for state in changes if state["state_id"] in created_ids]
        updated = [copy.deepcopy(state) for state in changes if state["state_id"] in updated_ids]

        # 更新优先：本集既被更新又声明删除的状态，忽略其删除声明。
        changed_state_ids = {state["state_id"] for state in changes}
        conflicting_ids = changed_state_ids & set(declared_deletions)
        declared_deletions = {
            state_id: reason for state_id, reason in declared_deletions.items()
            if state_id not in conflicting_ids
        }
        # 增量模式下，未出现在变化列表中的旧状态默认保留，不做隐式删除。
        if invalid_response:
            return {
                "world_state": old_states,
                "created": [], "updated": [], "deleted_state_ids": [],
                "preserved_after_invalid_rewrite": True,
                "conflicting_state_ids": sorted(conflicting_ids),
            }

        deleted_state_ids = sorted(set(declared_deletions))
        archive = self.data.setdefault("world_state_archive", [])
        archived_ids = {
            str(item.get("state", {}).get("state_id", ""))
            for item in archive if isinstance(item, dict) and isinstance(item.get("state"), dict)
        }
        for state_id in deleted_state_ids:
            if state_id not in archived_ids:
                reason = declared_deletions.get(state_id, "已删除")
                archive.append({
                    "state": copy.deepcopy(old_by_id[state_id]),
                    "deleted_episode": episode_id,
                    "reason": reason,
                })

        # 增量应用：旧状态中删除的去掉，更新的替换，未变的保留，新建的追加。
        changes_by_id = {state["state_id"]: state for state in changes}
        old_ids = {state.get("state_id") for state in old_states}
        new_world_state: List[Dict[str, Any]] = []
        for old_state in old_states:
            state_id = old_state.get("state_id")
            if state_id in deleted_state_ids:
                continue
            if state_id in changes_by_id:
                new_world_state.append(changes_by_id[state_id])
            else:
                new_world_state.append(old_state)
        for state in changes:
            if state.get("state_id") not in old_ids:
                new_world_state.append(state)

        self.data["world_state"] = new_world_state
        self.data["next_state_index"] = next_state_index
        return {
            "world_state": copy.deepcopy(new_world_state),
            "created": created,
            "updated": updated,
            "deleted_state_ids": deleted_state_ids,
        }
