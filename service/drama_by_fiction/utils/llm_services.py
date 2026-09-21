import os
import time
import uuid
import json
import pickle
import hashlib
import threading
from pathlib import Path
from collections import OrderedDict
from typing import Tuple, Optional

import yaml
import requests


from trpc.log import logger


DEFAULT_SYSTEM_PROMPT = "You are an AI assistant that helps people find information."

# ==================== 请求-响应日志记录 ====================
# 控制是否启用请求-响应对记录，可通过环境变量或直接修改此变量控制
ENABLE_REQUEST_LOG = os.environ.get("LLM_ENABLE_REQUEST_LOG", "0") == "1"


class LLMRequestLogger:
    """
    线程安全的 LLM 请求-响应对日志记录器。
    将每次请求的 payload 和返回的 data 以 JSONL 格式追加写入缓存文件，
    便于后续回溯查看。
    """

    def __init__(self, log_dir: str = None):
        if log_dir is None:
            log_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "../.cache",
                "llm_request_logs",
            )
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _get_log_file(self) -> Path:
        """按日期分文件，便于管理和查找"""
        date_str = time.strftime("%Y-%m-%d")
        return self.log_dir / f"request_log_{date_str}.jsonl"

    def log(self, payload: dict, data: dict, duration: float):
        """
        记录一次请求-响应对。

        Args:
            payload: 请求 payload（含 model, messages, params 等）
            data: 响应原始 JSON data
            duration: 请求耗时(秒)
        """
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": round(duration, 3),
            "request": payload,
            "response": data,
        }

        log_file = self._get_log_file()
        try:
            with self._lock:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write request log: {e}")


# 全局日志记录器实例
_request_logger = LLMRequestLogger()


class LLMCache:
    """LLM查询本地文件缓存系统"""

    def __init__(self, cache_dir: str = None, max_size: int = 100):
        """
        初始化缓存系统

        Args:
            cache_dir: 缓存目录路径，默认为项目根目录下的cache/llm_cache
            max_size: 最大缓存条目数
        """
        if cache_dir is None:
            # 默认缓存目录
            cache_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "../.cache",
                "llm_cache",
            )

        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self.cache_file = self.cache_dir / "llm_cache.pkl"
        self.index_file = self.cache_dir / "cache_index.json"

        # 创建缓存目录
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 加载缓存索引
        self.cache_index = self._load_index()

    def _load_index(self) -> OrderedDict:
        """加载缓存索引"""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    index_data = json.load(f)
                    # 转换为有序字典，保持访问顺序
                    return OrderedDict(index_data)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")

        return OrderedDict()

    def _save_index(self):
        """保存缓存索引"""
        try:
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(dict(self.cache_index), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")

    def _generate_cache_key(
        self, query, model_name, temperature, max_new_tokens, **kwargs
    ) -> str:
        """生成缓存键"""
        key_data = {
            "query": query if isinstance(query, str) else str(query),
            "model_name": model_name,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "kwargs": kwargs,
        }

        # 使用SHA256生成唯一键
        key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> Optional[Tuple[str, str]]:
        """从缓存中获取结果"""
        if cache_key not in self.cache_index:
            return None

        # 更新访问时间
        self.cache_index.move_to_end(cache_key)
        self._save_index()

        # 读取缓存文件
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to read cache file {cache_key}: {e}")
                # 删除损坏的缓存条目
                self.delete(cache_key)

        return None

    def set(self, cache_key: str, think: str, answer: str):
        """将结果存入缓存"""
        # 检查缓存大小，如果超过限制则删除最旧的条目
        if len(self.cache_index) >= self.max_size:
            oldest_key = next(iter(self.cache_index))
            self.delete(oldest_key)

        # 保存缓存内容
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump((think, answer), f)
        except Exception as e:
            logger.warning(f"Failed to write cache file {cache_key}: {e}")
            return

        # 更新索引
        self.cache_index[cache_key] = {
            "timestamp": time.time(),
            "file": cache_file.name,
        }
        self._save_index()

    def delete(self, cache_key: str):
        """删除缓存条目"""
        if cache_key in self.cache_index:
            # 删除缓存文件
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_key}: {e}")

            # 从索引中删除
            del self.cache_index[cache_key]
            self._save_index()

    def clear(self):
        """清空所有缓存"""
        for cache_key in list(self.cache_index.keys()):
            self.delete(cache_key)

        # 重新创建索引文件
        self.cache_index = OrderedDict()
        self._save_index()


# 全局缓存实例
llm_cache = LLMCache()


class VenusClient:
    def __init__(self):
        url, token = self._load_config()
        self.url = url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

    def _load_config(self):
        trpc_path = os.path.join(os.path.dirname(__file__), "../../../trpc_python.yaml")
        with open(trpc_path) as fp:
            config = yaml.safe_load(fp)
        url = config["API_URLS"]["venus"]
        token = config["API_KEYS"]["venus"]
        return url, token

    def chat(
        self,
        model_name,
        messages,
        enable_stream=False,
        **llm_params,
    ):
        if messages[-1]["role"] == "assistant":
            messages = messages[:-1]  # 去掉最后一个assistant

        # gpt-5.4/5.5 等模型不支持自定义 temperature，只允许默认值
        NO_TEMPERATURE_MODELS = ("gpt-5.4", "gpt-5.5")
        if any(model_name.startswith(m) for m in NO_TEMPERATURE_MODELS):
            llm_params.pop("temperature", None)

        payload = {
            "task_id": "test_" + str(uuid.uuid4()),
            "model": model_name,
            "messages": messages,
            "stream": enable_stream,
        }
        payload.update(llm_params)
        t_start = time.time()
        rsp = requests.post(
            self.url,
            headers=self.headers,
            json=payload,
            stream=enable_stream,
            timeout=(5, 600),
        )
        duration = time.time() - t_start
        if rsp.status_code != 200:
            raise RuntimeError(f"Error: {rsp.status_code}, {rsp.text}")
        data = rsp.json()

        # 记录请求 payload 和响应 data
        if ENABLE_REQUEST_LOG:
            _request_logger.log(payload, data, duration)

        think = data["choices"][0]["message"].get("reasoning_content", "")
        answer = data["choices"][0]["message"]["content"]
        return think, answer


def query_llm(
    query,
    model_name,
    temperature=0.6,
    max_new_tokens=32768,
    enable_stream=False,
    request_key="",
    max_retries=3,
    retry_delay=60,
    validator=None,
    **kwargs,
):
    if isinstance(query, str):
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
    elif isinstance(query, list):
        messages = query
    else:
        raise ValueError("query must be str or list.")

    client = VenusClient()

    llm_params = {
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
    }
    if "claude" in model_name.lower():
        llm_params["max_tokens"] = max_new_tokens

    if kwargs:
        llm_params.update(kwargs)

    for attempt in range(1, max_retries + 1):
        try:
            think, answer = client.chat(
                model_name,
                messages,
                enable_stream,
                **llm_params,
            )
            if validator is not None:
                is_valid, error_msg = validator(answer)
                if not is_valid:
                    logger.warning(
                        f"{request_key} response {attempt} invalid: {error_msg}"
                    )
                    if attempt < max_retries:
                        continue

                    return think, ""

            logger.debug(f"{request_key} response {attempt} success: {answer}")
            return think, answer
        except Exception as e:
            logger.error(f"{request_key} response {attempt} failed: {e}")
            time.sleep(retry_delay * attempt)

    return "", ""


if __name__ == "__main__":

    models = [
        # "gemini-2.5-pro",
        # "gemma-4-31b-it",
        # "qwen3-30b-a3b-thinking-2507",
        "claude-opus-4-6",
    ]
    for model_name in models:
        print("model_name:", model_name)
        think, answer = query_llm("介绍下腾讯的滨海大厦", model_name=model_name)
        print("think:", think)
        print("answer:", answer)
