import os
from typing import List

import yaml

from .base import BaseCosModel


"""
short_anime:
  chapter_block_cnt: 50
  overlap_chapter_cnt: 10
  arc_range: "5～10章"
  max_workers: 8
  retry_cnt: 3
  model_name: gemini-2.5-pro  # 对应 llms/model_config.yaml 中的 model key
"""


class FictionConfig(BaseCosModel):
    model_name: str = "gemini-2.5-pro"  # 对应 llms/model_config.yaml 中的 model key
    chapter_block_cnt: int = 50
    overlap_chapter_cnt: int = 10
    arc_range: str = "5～10章"
    min_arc_len: int = 5
    max_arc_len: int = 12
    fallback_len: int = 8
    max_workers: int = 8
    retry_cnt: int = 3
    novel_dirs: List[str] = ["ip_gpt/formal/novel_emb/1.0", "ip_gpt/test/novel_emb/1.0"]
    episode_outline_method: str = "cpg"  # 分集大纲生成方法：cpg（因果图拓扑展开）/ sequential（按章节顺序批次生成）
    # CPG 候选对筛选参数
    cpg_candidate_top_k: int = 10           # 每个事件最多关联的邻居数
    cpg_candidate_max_chapter_dist: int = 30  # 候选对允许的最大章节距离
    cpg_max_candidate_pairs: int = 500      # 候选对总量上限（超出截断并警告）
    cpg_edges_batch_size: int = 50          # 每批送给 LLM 的候选对数量
    cpg_emb_top_k: int = 5                  # embedding 第二路：每个事件最多取多少相似邻居
    cpg_emb_sim_threshold: float = 0.75     # embedding 相似度阈值（余弦相似度）
    cpg_emb_model: str = "server:251477"   # venus embedding 模型
    enable_script_fix: bool = True  # 是否启用剧本字数校验与自动修正
    use_memory_graph: bool = False  # 剧本生成侧是否启用记忆图谱（Narrative Memory Graph）
    script_generate_method: str = "parallel"  # 剧本生成方法：parallel（并行）/ sequential（逐集串行，可叠加 use_memory_graph）
    # 记忆图谱参数（use_memory_graph=True 时生效）
    state_max_entities: int = 200          # DNMG 节点+边总数上限，超出时 LRU 淘汰
    state_fallback_recent_n: int = 20      # ChromaDB 检索失败时 fallback 的最近 N 个实体
    state_retrieve_top_k: int = 15         # ChromaDB 语义检索返回 top-K 实体
    state_extract_top_k: int = 20          # LLM 从大纲抽取实体后，按置信度保留 top-K 个
    extract_model_name: str = ""           # DNMG 抽取步骤使用的模型，为空时回退到 model_name

    def __init__(self, yaml_path: str = None):
        super().__init__()
        # 先加载 generation_config.yaml（本地默认配置）
        _generation_config_path = os.path.join(os.path.dirname(__file__), "generation_config.yaml")
        self._update_from_generation_config(_generation_config_path)
        # 再用 trpc_python.yaml 覆盖（线上/部署环境配置，优先级更高）
        self._update_from_yaml(yaml_path)

    def _update_from_generation_config(self, yaml_path: str):
        """从 configs/generation_config.yaml 加载本地默认配置"""
        if not os.path.exists(yaml_path):
            return
        with open(yaml_path, "r", encoding="utf-8") as file:
            config_data = yaml.safe_load(file)
        if not config_data:
            return
        # 嵌套 section → FictionConfig 属性前缀映射
        # cpg.candidate_top_k → cpg_candidate_top_k
        # dnmg.use_memory_graph → use_memory_graph（无前缀）
        nested_sections = {"cpg": "cpg_", "dnmg": ""}
        for key, value in config_data.items():
            if key in nested_sections and isinstance(value, dict):
                prefix = nested_sections[key]
                for sub_key, sub_value in value.items():
                    attr = prefix + sub_key
                    if hasattr(self, attr):
                        setattr(self, attr, sub_value)
            elif hasattr(self, key):
                setattr(self, key, value)

    def _update_from_yaml(self, yaml_path: str):
        """从yaml文件加载配置，使用hasattr动态更新字段"""
        if yaml_path is None:
            yaml_path = os.path.join(
                os.path.dirname(__file__), "../../../trpc_python.yaml"
            )
        if not os.path.exists(yaml_path):
            return

        with open(yaml_path, "r", encoding="utf-8") as file:
            config_data = yaml.safe_load(file)

        if not config_data or "short_anime" not in config_data:
            return

        fiction_config = config_data["short_anime"]

        nested_sections = {"cpg": "cpg_", "dnmg": ""}
        for key, value in fiction_config.items():
            if key in nested_sections and isinstance(value, dict):
                prefix = nested_sections[key]
                for sub_key, sub_value in value.items():
                    attr = prefix + sub_key
                    if hasattr(self, attr):
                        setattr(self, attr, sub_value)
            elif hasattr(self, key):
                setattr(self, key, value)

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """类方法：从yaml文件创建配置实例"""
        instance = cls()
        instance._update_from_yaml(yaml_path)
        return instance
