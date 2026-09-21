"""
data_models/cpg.py

因果情节图（Causal Plot Graph，CPG）数据模型：
  - CausalPlot       — Plot 层：故事弧（对应原 StoryArc，两层结构的上层）
  - CausalEvent      — Event 层：CPG 节点，叙事事件（场粒度，可跨章或一章多个）
  - CausalEdge       — CPG 边：因果关系（dataclass，可序列化）
  - CausalPlotGraph  — 完整的因果情节图，使用 NetworkX DiGraph 作为存储后端

两层结构：
  Plot (PXX)  —包含→  Event (PXX-EXX)
  Event 节点间有有向因果边，构成 DAG。

存储：pickle 序列化后存 COS，本地临时缓存在 /tmp/cpg_cache/
"""

from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import networkx as nx
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class CausalPlot:
    """Plot 层：故事弧，对应原著的一段连续情节。

    Attributes:
        plot_id:            Plot 唯一标识（如 "P00"、"P01"）
        start_chapter_num:  起始章节编号（含）
        end_chapter_num:    结束章节编号（含）
        title:              故事弧标题
        description:        故事弧内容摘要（200字左右）
    """
    plot_id: str = ""
    start_chapter_num: int = 0
    end_chapter_num: int = 0
    title: str = ""
    description: str = ""


@dataclass_json
@dataclass
class CausalEvent:
    """Event 层：CPG 节点，代表故事中的一个叙事事件单元（场粒度）。

    粒度说明：一章可包含多个事件，也可多章共享一个事件，由叙事逻辑决定。

    Attributes:
        id:                 事件唯一标识，格式 "PXX-EXX"（如 "P03-E07"）
        plot_id:            所属 Plot 的 ID（如 "P03"）
        title:              事件标题（简短描述，10字内）
        description:        完整事件描述（自然语言，200字以内）
        synopsis:           事件简述（50字以内，供分集大纲使用）
        beat_type:          戏剧节拍类型：hook / rising / climax / resolution / transition
        drama_weight:       戏剧权重，0.0~1.0，表示该事件的情感冲击力
        importance:         主线重要程度，0.0~1.0，用于删减时的影响评估
        estimated_minutes:  估算所需时长（分钟）
        assigned_episode:   调度后分配到的集号（0-indexed，-1 表示未分配）
        characters:         涉及角色名称列表
        locations:          涉及场景/地点列表
        narrative_time:     故事内时间描述（如"第三年春"），非现实时间
        source_chapters:    来源原著章节列表，如 [12, 13, 14]
        effects:            此事件对世界状态的改变（自然语言列表）
        subplot_tags:       所属支线标签列表，如 ["黛玉-宝玉支线", "家族主线"]
        source:             来源：original（原著）或 adapted（改编新增）
        status:             状态：active / deleted / needs_revision / broken
    """
    id: str = ""
    plot_id: str = ""
    title: str = ""
    description: str = ""
    synopsis: str = ""
    beat_type: str = ""
    drama_weight: float = 0.0
    importance: float = 0.5
    estimated_minutes: float = 0.0
    assigned_episode: int = -1
    characters: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    narrative_time: str = ""
    source_chapters: List[int] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    subplot_tags: List[str] = field(default_factory=list)
    source: str = "original"
    status: str = "active"


@dataclass_json
@dataclass
class CausalEdge:
    """CPG 边：表示两个叙事事件之间的因果关系。

    Attributes:
        src_id:          源事件 ID
        dst_id:          目标事件 ID
        causal_strength: 因果强度，0.0~1.0（1.0 = 强因果，0.0 = 弱相关）
        edge_type:       关系类型：
                           "causes"       — A 直接导致 B
                           "enables"      — A 使 B 成为可能（非必然）
                           "motivates"    — A 为 B 提供动机
                           "contrasts"    — A 与 B 形成对比/反转
                           "foreshadows"  — A 为 B 埋下伏笔
    """
    src_id: str = ""
    dst_id: str = ""
    causal_strength: float = 1.0
    edge_type: str = "causes"
    reason: str = ""


class CausalPlotGraph:
    """完整的因果情节图，使用 NetworkX DiGraph 作为存储后端。

    两层结构：
      - plots:  有序 Plot 列表（故事弧，上层）
      - _graph: Event 节点 + 因果边构成的 DAG（下层）

    G₀ 为从原著提取的初始图；G* 为经过改编操作后的目标图。

    NetworkX 存储约定：
      - 节点：node_id (str，格式 "PXX-EXX")，节点属性 "event" = CausalEvent 对象
      - 边：(src_id, dst_id)，边属性 "edge" = CausalEdge 对象
    """

    def __init__(self, story_id: str = ""):
        self.story_id: str = story_id
        self.plots: List[CausalPlot] = []          # Plot 层（有序）
        self._graph: nx.DiGraph = nx.DiGraph()     # Event 层 DAG
        self.adaptation_ops: List[dict] = []

    # ------------------------------------------------------------------
    # Plot 层操作
    # ------------------------------------------------------------------

    def add_plot(self, plot: CausalPlot) -> None:
        """添加 Plot。若 plot_id 已存在则覆盖。"""
        for i, p in enumerate(self.plots):
            if p.plot_id == plot.plot_id:
                self.plots[i] = plot
                return
        self.plots.append(plot)

    def get_plot(self, plot_id: str) -> Optional[CausalPlot]:
        """按 plot_id 查询 Plot。"""
        for p in self.plots:
            if p.plot_id == plot_id:
                return p
        return None

    def events_for_plot(self, plot_id: str) -> List[CausalEvent]:
        """返回属于指定 Plot 的所有事件（按拓扑顺序）。"""
        return [e for e in self.topological_order() if e.plot_id == plot_id]

    # ------------------------------------------------------------------
    # 节点操作
    # ------------------------------------------------------------------

    def add_event(self, event: CausalEvent) -> None:
        """添加事件节点。若 id 已存在则覆盖。"""
        self._graph.add_node(event.id, event=event)

    def remove_event(self, event_id: str) -> None:
        """删除事件节点及其所有关联边。"""
        self._graph.remove_node(event_id)

    def get_event(self, event_id: str) -> Optional[CausalEvent]:
        """按 id 查询事件节点。"""
        if event_id in self._graph:
            return self._graph.nodes[event_id]["event"]
        return None

    def has_event(self, event_id: str) -> bool:
        return event_id in self._graph

    @property
    def nodes(self) -> Dict[str, CausalEvent]:
        """返回 {event_id: CausalEvent} 字典（兼容旧接口）。"""
        return {nid: data["event"] for nid, data in self._graph.nodes(data=True)}

    # ------------------------------------------------------------------
    # 边操作
    # ------------------------------------------------------------------

    def add_edge(self, edge: CausalEdge) -> None:
        """添加因果边。若边已存在则覆盖。"""
        self._graph.add_edge(edge.src_id, edge.dst_id, edge=edge)

    def remove_edge(self, src_id: str, dst_id: str) -> None:
        """删除指定因果边。"""
        if self._graph.has_edge(src_id, dst_id):
            self._graph.remove_edge(src_id, dst_id)

    def get_edge(self, src_id: str, dst_id: str) -> Optional[CausalEdge]:
        """查询指定因果边。"""
        if self._graph.has_edge(src_id, dst_id):
            return self._graph[src_id][dst_id]["edge"]
        return None

    @property
    def edges(self) -> List[CausalEdge]:
        """返回所有 CausalEdge 列表（兼容旧接口）。"""
        return [data["edge"] for _, _, data in self._graph.edges(data=True)]

    # ------------------------------------------------------------------
    # 图查询
    # ------------------------------------------------------------------

    def get_predecessors(self, event_id: str) -> List[CausalEvent]:
        """返回所有前驱事件（指向 event_id 的节点）。"""
        return [
            self._graph.nodes[nid]["event"]
            for nid in self._graph.predecessors(event_id)
            if nid in self._graph
        ]

    def get_successors(self, event_id: str) -> List[CausalEvent]:
        """返回所有后继事件（从 event_id 出发的节点）。"""
        return [
            self._graph.nodes[nid]["event"]
            for nid in self._graph.successors(event_id)
            if nid in self._graph
        ]

    def get_in_edges(self, event_id: str) -> List[CausalEdge]:
        """返回所有以 event_id 为目标的入边。"""
        return [
            self._graph[src][event_id]["edge"]
            for src in self._graph.predecessors(event_id)
        ]

    def get_out_edges(self, event_id: str) -> List[CausalEdge]:
        """返回所有以 event_id 为源的出边。"""
        return [
            self._graph[event_id][dst]["edge"]
            for dst in self._graph.successors(event_id)
        ]

    def topological_order(self) -> List[CausalEvent]:
        """按因果依赖拓扑排序，返回有序的 CausalEvent 列表。"""
        ordered_ids = list(nx.topological_sort(self._graph))
        return [self._graph.nodes[nid]["event"] for nid in ordered_ids]

    def unassigned_events(self) -> List[CausalEvent]:
        """返回尚未分配集号的事件列表。"""
        return [e for e in self.nodes.values() if e.assigned_episode == -1]

    def events_for_episode(self, episode_id: int) -> List[CausalEvent]:
        """返回已分配到指定集号的事件列表（按拓扑顺序）。"""
        return [e for e in self.topological_order() if e.assigned_episode == episode_id]

    def is_valid_dag(self) -> bool:
        """检查图是否为有效的 DAG（无环）。"""
        return nx.is_directed_acyclic_graph(self._graph)

    # ------------------------------------------------------------------
    # 改编操作记录
    # ------------------------------------------------------------------

    def record_op(self, op: dict) -> None:
        """记录一次改编操作，用于溯源。"""
        self.adaptation_ops.append(op)

    # ------------------------------------------------------------------
    # 序列化（pickle，用于 COS 存储）
    # ------------------------------------------------------------------

    def to_bytes(self) -> bytes:
        """序列化为 bytes（pickle），用于 COS 存储。"""
        return pickle.dumps(self)

    @classmethod
    def from_bytes(cls, data: bytes) -> "CausalPlotGraph":
        """从 bytes（pickle）反序列化。"""
        return pickle.loads(data)

    def deepcopy(self) -> "CausalPlotGraph":
        """深拷贝，用于改编操作前保留原图。"""
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # 调试
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CausalPlotGraph(story_id={self.story_id!r}, "
            f"plots={len(self.plots)}, nodes={len(self._graph)}, "
            f"edges={self._graph.number_of_edges()})"
        )

    def to_dot(self, output_path: str = "/tmp/cpg.png") -> str:
        """用 Graphviz 渲染因果图，输出 PNG/SVG/PDF（由 output_path 后缀决定）。

        节点按 Plot 分组（subgraph cluster），颜色区分 beat_type。
        返回实际输出路径。

        依赖：pip install graphviz
        """
        try:
            from graphviz import Digraph
        except ImportError:
            raise ImportError("请先安装 graphviz：pip install graphviz")

        # beat_type → 节点填充色
        BEAT_COLOR = {
            "hook":       "#AED6F1",  # 蓝
            "rising":     "#A9DFBF",  # 绿
            "climax":     "#F1948A",  # 红
            "resolution": "#F9E79F",  # 黄
            "transition": "#D7DBDD",  # 灰
        }
        EDGE_COLOR = {
            "causes":      "#E74C3C",
            "enables":     "#E67E22",
            "motivates":   "#8E44AD",
            "contrasts":   "#1ABC9C",
            "foreshadows": "#95A5A6",
        }

        fmt = output_path.rsplit(".", 1)[-1] if "." in output_path else "png"
        dot = Digraph(
            name="CPG",
            format=fmt,
            graph_attr={"rankdir": "TB", "splines": "true", "fontname": "Arial"},
            node_attr={"shape": "box", "style": "filled", "fontname": "Arial", "fontsize": "10"},
            edge_attr={"fontname": "Arial", "fontsize": "9"},
        )

        # 按 Plot 分组画 subgraph cluster
        from collections import defaultdict
        plot_events: dict = defaultdict(list)
        for nid, event in self.nodes.items():
            plot_events[event.plot_id].append(event)

        for plot in self.plots:
            pid = plot.plot_id
            with dot.subgraph(name=f"cluster_{pid}") as sub:
                sub.attr(
                    label=f"{pid} {plot.title}\n§{plot.start_chapter_num}-§{plot.end_chapter_num}",
                    style="rounded,filled",
                    fillcolor="#F8F9FA",
                    fontsize="11",
                    fontname="Arial Bold",
                )
                for event in plot_events.get(pid, []):
                    color = BEAT_COLOR.get(event.beat_type, "#FFFFFF")
                    label = (
                        f"{event.id}\n{event.title}\n"
                        f"beat={event.beat_type}  w={event.drama_weight:.1f}"
                    )
                    sub.node(event.id, label=label, fillcolor=color)

        # 画边
        for edge in self.edges:
            color = EDGE_COLOR.get(edge.edge_type, "#999999")
            label = f"{edge.edge_type}\n{edge.causal_strength:.2f}"
            dot.edge(
                edge.src_id, edge.dst_id,
                label=label,
                color=color,
                fontcolor=color,
                penwidth=str(0.5 + edge.causal_strength * 2),
            )

        # 渲染输出（去掉后缀让 graphviz 自己加）
        out_base = output_path.rsplit(".", 1)[0] if "." in output_path else output_path
        dot.render(out_base, cleanup=True)
        return output_path

    def summary(self) -> str:
        """返回图的统计摘要字符串。"""
        assigned = sum(1 for e in self.nodes.values() if e.assigned_episode >= 0)
        lines = [
            f"CausalPlotGraph: {self.story_id}",
            f"  Plot 数: {len(self.plots)}",
            f"  Event 节点数: {len(self._graph)}  (已分配: {assigned})",
            f"  因果边数:   {self._graph.number_of_edges()}",
            f"  改编操作数: {len(self.adaptation_ops)}",
            f"  是否为 DAG: {self.is_valid_dag()}",
        ]
        for p in self.plots:
            event_cnt = len(self.events_for_plot(p.plot_id))
            lines.append(f"    {p.plot_id} [{p.title}] §{p.start_chapter_num}-§{p.end_chapter_num}  events={event_cnt}")
        return "\n".join(lines)
