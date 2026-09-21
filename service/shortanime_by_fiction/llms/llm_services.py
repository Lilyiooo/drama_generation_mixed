import json
import os
import time
import uuid

import yaml
import requests
from trpc.log import logger


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

    def _process_payload(self, payload: dict) -> dict:
        """处理payload，针对不同模型做特殊参数处理"""
        model_name = payload.get("model", "")
        if model_name == "claude-opus-4-7":
            payload.pop("temperature", None)
        if model_name == "gpt-5.5":
            payload.pop("max_tokens", None)
            payload.pop("temperature", None)
        return payload

    def _process_response(self, rsp) -> tuple[str, str, dict]:
        """处理响应，返回 (think, answer, usage)"""
        if rsp.status_code != 200:
            raise RuntimeError(f"Error: {rsp.status_code}, {rsp.text}")
        data = rsp.json()
        finish_reason = data["choices"][0]["finish_reason"]
        if finish_reason in ['content_filter']:
            raise RuntimeError(f"Error: {finish_reason}")
        think = data["choices"][0]["message"].get("reasoning_content", "")
        answer = data["choices"][0]["message"]["content"]
        usage = {
            "finish_reason": finish_reason,
            **data.get("usage", {}),
        }
        return think, answer, usage

    def _process_stream_response(self, rsp) -> tuple[str, str, dict]:
        """处理 SSE 流式响应，拼接完整后返回 (think, answer, usage)"""
        if rsp.status_code != 200:
            raise RuntimeError(f"Error: {rsp.status_code}, {rsp.text}")

        t_start = time.time()
        t_first_token = None
        t_last_log = t_start

        think_parts = []
        answer_parts = []
        usage = {}
        finish_reason = None

        for raw_line in rsp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            choices = data.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                finish_reason = choices[0].get("finish_reason") or finish_reason
                content = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or ""

                if (content or reasoning) and t_first_token is None:
                    t_first_token = time.time()
                    logger.info(f"[stream] 首token耗时: {t_first_token - t_start:.3f}s")

                if content:
                    answer_parts.append(content)
                if reasoning:
                    think_parts.append(reasoning)

            if "usage" in data:
                usage = data["usage"]

            now = time.time()
            if now - t_last_log >= 60:
                t_last_log = now
                current_answer = "".join(answer_parts)
                current_think = "".join(think_parts)
                logger.info(
                    f"[stream] 进行中，已用时 {now - t_start:.1f}s，"
                    f"think={len(current_think)} 字符，answer={len(current_answer)} 字符，"
                    f"末尾: {(current_think or current_answer)[-200:]!r}"
                )

        if finish_reason in ["content_filter"]:
            raise RuntimeError(f"Error: {finish_reason}")

        return (
            "".join(think_parts),
            "".join(answer_parts),
            {"finish_reason": finish_reason, **usage},
        )

    def encode_texts(self, text_list: list, model_name: str = "server:251477") -> list:
        """批量 embedding，返回向量列表（与 text_list 等长）。
        使用 venus llmproxy embeddings endpoint。
        """
        emb_url = "http://v2.open.venus.oa.com/llmproxy/embeddings"
        results = []
        for text in text_list:
            body = {"input": text, "model": model_name}
            rsp = requests.post(emb_url, headers=self.headers, json=body, timeout=(5, 60))
            if rsp.status_code != 200:
                raise RuntimeError(f"Venus embedding error: {rsp.status_code}, {rsp.text}")
            data = rsp.json()
            results.append(data["data"][0]["embedding"])
        return results

    def chat(
        self,
        model_name,
        messages,
        enable_stream=False,
        **llm_params,
    ):
        if messages[-1]["role"] == "assistant":
            messages = messages[:-1]  # 去掉最后一个assistant

        payload = {
            "task_id": "test_" + str(uuid.uuid4()),
            "model": model_name,
            "messages": messages,
            "stream": enable_stream,
        }
        payload.update(llm_params)
        payload = self._process_payload(payload)
        rsp = requests.post(
            self.url,
            headers=self.headers,
            json=payload,
            stream=enable_stream,
            timeout=(5, 600),
        )
        if enable_stream:
            return self._process_stream_response(rsp)
        return self._process_response(rsp)


def _to_float_list(v) -> list:
    """将向量转为纯 Python List[float]，兼容 numpy array 和嵌套列表。"""
    try:
        # numpy array / tensor
        return v.tolist()
    except AttributeError:
        pass
    return [float(x) for x in v]


def _normalize_embeddings(raw, n: int) -> list:
    """将 encode_texts 的原始返回值规范化为 List[List[float]]。

    encode_texts 可能返回：
      - List[List[float]]          正常情况
      - numpy array shape (n, dim) 批量
      - numpy array shape (dim,)   单条（n=1）时退化为 1D
    """
    try:
        import numpy as np
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 1:
            # 单条退化为 1D，补充 batch 维度
            arr = arr.reshape(1, -1)
        return arr.tolist()
    except Exception:
        pass
    # fallback：逐条转换
    if n == 1 and not isinstance(raw[0], (list, tuple)) and not hasattr(raw[0], '__iter__'):
        # raw 本身就是单条向量 [f, f, f, ...]
        return [[float(x) for x in raw]]
    return [_to_float_list(v) for v in raw]


class VenusEmbeddingFunction:
    """ChromaDB-compatible embedding function，使用 Venus embedding API。

    符合 chromadb EmbeddingFunction 接口：__call__(input: List[str]) -> List[List[float]]
    可直接注入 EntityVectorStore(embedding_fn=VenusEmbeddingFunction())。
    """

    def __init__(self, model_name: str = "server:251477"):
        self._venus = VenusClient()
        self._model_name = model_name

    def name(self) -> str:
        return f"venus-{self._model_name}"

    def __call__(self, input: list) -> list:  # noqa: A002
        """input: List[str]，返回 List[List[float]]（纯 Python float，兼容 ChromaDB rust 后端）"""
        if not input:
            return []
        try:
            raw = self._venus.encode_texts(input, model_name=self._model_name)
            return _normalize_embeddings(raw, len(input))
        except Exception as e:
            logger.warning(f"VenusEmbeddingFunction failed: {e}")
            raise

    def embed_documents(self, input: list) -> list:  # noqa: A002
        """ChromaDB >= 0.6 新接口，批量 embed。"""
        return self.__call__(input)

    def embed_query(self, input: str) -> list:  # noqa: A002
        """ChromaDB >= 0.6 新接口，单条 query embed。"""
        result = self.__call__([input])
        return result[0] if result else []
