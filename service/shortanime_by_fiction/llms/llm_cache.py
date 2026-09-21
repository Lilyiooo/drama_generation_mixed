import os
import time
import pickle
import tempfile
from pathlib import Path
from collections import OrderedDict
from typing import Tuple, Optional
from trpc.log import logger
import json
import hashlib

try:
    import json_repair as _json_repair
except ImportError:
    _json_repair = None

from service.shortanime_by_fiction.configs.llm_config import llm_config


class LLMCache:
    """LLM查询本地文件缓存系统"""

    def __init__(self, cache_dir: str = None, max_size: int = None):
        """
        初始化缓存系统

        Args:
            cache_dir: 缓存目录路径，默认读取 llm_config.yaml
            max_size: 最大缓存条目数，默认读取 llm_config.yaml
        """
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                llm_config.cache.cache_dir,
            )
        if max_size is None:
            max_size = llm_config.cache.max_size

        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self.cache_file = self.cache_dir / "llm_cache.pkl"
        self.index_file = self.cache_dir / "cache_index.json"
        self.data_dir = self.cache_dir / "data"

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.cache_index = self._load_index()

    def _load_index(self) -> OrderedDict:
        """加载缓存索引"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    raw = f.read()
                try:
                    index_data = json.loads(raw)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to load cache index: {e}")
                    if _json_repair is not None:
                        logger.warning("Attempting to repair cache index with json_repair ...")
                        index_data = _json_repair.loads(raw)
                        if isinstance(index_data, dict) and index_data:
                            logger.warning(f"Repaired cache index: {len(index_data)} entries recovered, rewriting ...")
                            result = OrderedDict(index_data)
                            self._atomic_write_index(result)
                            return result
                    logger.warning("Cache index unrecoverable, starting fresh")
                    return OrderedDict()
                return OrderedDict(index_data)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")

        return OrderedDict()

    def _atomic_write_index(self, index: OrderedDict):
        """原子写入缓存索引（先写临时文件，再 rename，避免写入中途崩溃留下损坏文件）"""
        try:
            dir_path = self.index_file.parent
            with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', dir=dir_path, delete=False, suffix='.tmp'
            ) as tmp:
                json.dump(dict(index), tmp, ensure_ascii=False, indent=2)
                tmp_path = tmp.name
            os.replace(tmp_path, self.index_file)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")

    def _save_index(self):
        """保存缓存索引"""
        self._atomic_write_index(self.cache_index)

    def _generate_cache_key(self, query, model_name, temperature, max_new_tokens, **kwargs) -> str:
        """生成缓存键"""
        key_data = {
            "query": query if isinstance(query, str) else json.dumps(query, ensure_ascii=False),
            "model_name": model_name,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "kwargs": kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(key_str.encode('utf-8')).hexdigest()

    def get(self, cache_key: str) -> Optional[Tuple[str, str]]:
        """从缓存中获取结果"""
        if cache_key not in self.cache_index:
            return None

        self.cache_index.move_to_end(cache_key)
        self._save_index()

        cache_file = self.cache_dir / "data" / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to read cache file {cache_key}: {e}")
                self.delete(cache_key)

        return None

    def set(self, cache_key: str, think: str, answer: str):
        """将结果存入缓存"""
        if len(self.cache_index) >= self.max_size:
            oldest_key = next(iter(self.cache_index))
            self.delete(oldest_key)

        cache_file = self.data_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump((think, answer), f)
        except Exception as e:
            logger.warning(f"Failed to write cache file {cache_key}: {e}")
            return
        logger.info(f"set cache key: {cache_key}")

        self.cache_index[cache_key] = {
            "timestamp": time.time(),
            "file": cache_file.name
        }
        self._save_index()

    def delete(self, cache_key: str):
        """删除缓存条目"""
        if cache_key in self.cache_index:
            cache_file = self.data_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_key}: {e}")

            del self.cache_index[cache_key]
            self._save_index()

    def clear(self):
        """清空所有缓存"""
        for cache_key in list(self.cache_index.keys()):
            self.delete(cache_key)

        self.cache_index = OrderedDict()
        self._save_index()


# 全局缓存实例
llm_cache = LLMCache()
