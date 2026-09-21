import os
from typing import Dict

import yaml
from trpc.log import logger


MODEL_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "model_config.yaml")
TRPC_YAML_PATH = os.path.join(os.path.dirname(__file__), "../../../trpc_python.yaml")


class ModelConfig:
    def __init__(self, config_path: str = None, yaml_path: str = None):
        if config_path is None:
            config_path = MODEL_CONFIG_PATH
        if yaml_path is None:
            yaml_path = TRPC_YAML_PATH

        all_config: Dict = self._load(config_path)

        # 从 trpc_python.yaml 读 model_name
        model_name = self._load_model_name(yaml_path)

        model_data = all_config.get(model_name)
        if model_data is None:
            raise ValueError(f"ModelConfig: model '{model_name}' not found in config")

        self.model_name: str = model_name
        self.model_type: str = model_data.get("model_type", "venus")
        self.content_max_word_cnt: int = model_data.get("content_max_word_cnt", 80000)
        self.turncat_word_cnt: int = model_data.get("turncat_word_cnt", 200000)
        self.episode_batch_size: int = model_data.get("episode_batch_size", 60)
        self.summary_truncate_length: int = model_data.get("summary_truncate_length", 200000)
        self.max_workers: int = all_config.get("max_workers", 8)
        self.min_chapter_idx: int = all_config.get("min_chapter_idx", 1)
        self.max_chapter_idx: int = all_config.get("max_chapter_idx", 450)

        logger.info(f"[ModelConfig] using model: {self.model_name} (type={self.model_type})")

    def _load(self, config_path: str) -> Dict:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"ModelConfig file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _load_model_name(self, yaml_path: str) -> str:
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"trpc yaml not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        model_name = data.get("short_anime", {}).get("model_name")
        if not model_name:
            raise ValueError("ModelConfig: 'short_anime.model_name' not found in trpc_python.yaml")
        return model_name


# 单例
model_cfg = ModelConfig()
