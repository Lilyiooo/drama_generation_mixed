import os
from typing import Any

import yaml
from trpc.log import logger


LLM_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "generation_config.yaml")


class LLMCallConfig:
    def __init__(self, data: dict):
        self.temperature: float = data.get("temperature", 0.6)
        self.max_new_tokens: int = data.get("max_new_tokens", 32768)
        self.max_tokens: int = data.get("max_tokens", 64000)
        self.max_retries: int = data.get("max_retries", 3)
        self.retry_delay: int = data.get("retry_delay", 60)
        self.use_cache: bool = data.get("use_cache", False)
        self.use_io_log: bool = data.get("use_io_log", False)
        self.llm_provider: str = data.get("llm_provider", "venus")


class LLMCacheConfig:
    def __init__(self, data: dict):
        self.max_size: int = data.get("max_size", 50000)
        self.cache_dir: str = data.get("cache_dir", "../.cache/llm_cache")


TRPC_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../trpc_python.yaml")


class LLMConfig:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = LLM_CONFIG_PATH
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # 用 trpc_python.yaml 的 short_anime 节点覆盖
        if os.path.exists(TRPC_CONFIG_PATH):
            with open(TRPC_CONFIG_PATH, "r", encoding="utf-8") as f:
                trpc_data = yaml.safe_load(f) or {}
            for key, value in trpc_data.get("short_anime", {}).items():
                if isinstance(value, dict) and isinstance(data.get(key), dict):
                    data[key].update(value)
                else:
                    data[key] = value
        self.call = LLMCallConfig(data.get("llm_caller", {}))
        self.cache = LLMCacheConfig(data.get("cache", {}))
        logger.info(f"[LLMConfig] use_cache={self.call.use_cache}, use_io_log={self.call.use_io_log}")


# 单例
llm_config = LLMConfig()
