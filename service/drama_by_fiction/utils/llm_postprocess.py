"""LLM后处理相关功能：prompt渲染 → LLM调用 → 后处理 → 缓存"""

import time

import json_repair

import trpc
from trpc.log import logger

from .task import Task
from .llm_services import VenusClient, llm_cache, DEFAULT_SYSTEM_PROMPT
from .prompt_manager import prompt_manager


# 模块级复用，避免每次调用都读取配置文件
_venus_client = None


def _get_venus_client() -> VenusClient:
    """懒加载单例 VenusClient"""
    global _venus_client
    if _venus_client is None:
        _venus_client = VenusClient()
    return _venus_client


def post_extract_json(response):
    """从LLM响应中提取JSON数据"""
    return json_repair.loads(response)


def post_validate_json_schema(response, schema):
    """从LLM响应中提取JSON并验证schema，验证失败抛异常"""
    from jsonschema import validate

    json_data = json_repair.loads(response)
    validate(instance=json_data, schema=schema)
    return json_data


def _build_messages(query):
    """将query构造为LLM消息列表"""
    if isinstance(query, str):
        return [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
    elif isinstance(query, list):
        return query
    else:
        raise ValueError(f"query 必须是 str 或 list，实际类型: {type(query)}")


def _call_llm_with_cache(
    ctx,
    task: Task,
    model_name: str,
    messages: list,
    request_key: str,
    llm_params: dict,
    max_retries: int,
    retry_delay: int,
    use_cache: bool,
):
    """
    LLM调用核心：缓存检查 → LLM调用 → 后处理 → 缓存写入。

    重试策略：
    - 缓存命中但后处理失败：清除缓存，下一轮重新调用LLM
    - LLM调用成功但后处理失败：直接重试（不等待）
    - LLM调用失败：等待后重试（退避递增）
    """
    client = _get_venus_client()
    cache_key = llm_cache._generate_cache_key(
        task.query,
        model_name,
        llm_params.get("temperature", 0.6),
        llm_params.get("max_new_tokens", 32768),
    )

    for attempt in range(1, max_retries + 1):
        is_cache_hit = False

        # 仅首次尝试时检查缓存
        if attempt == 1 and use_cache:
            cached = llm_cache.get(cache_key)
            if cached is not None:
                task.think, task.answer = cached
                is_cache_hit = True
                logger.debug_context(ctx, f"{request_key} attempt {attempt} cache hit")
            else:
                logger.debug_context(ctx, f"{request_key} attempt {attempt} cache miss")

        # 调用LLM
        if not is_cache_hit:
            try:
                task.think, task.answer = client.chat(
                    model_name,
                    messages,
                    enable_stream=False,
                    **llm_params,
                )
            except Exception as e:
                logger.error_context(
                    ctx, f"{request_key} attempt {attempt} LLM调用失败: {e}"
                )
                if attempt < max_retries:
                    time.sleep(retry_delay * attempt)
                    continue
                raise

        task.result = task.answer

        # 后处理
        if task.post_process_func is not None:
            try:
                task.result = task.post_process_func(
                    task.answer, *task.post_process_args, **task.post_process_kwargs
                )
            except Exception as e:
                logger.warning_context(
                    ctx,
                    f"{request_key} attempt {attempt} 后处理失败 "
                    f"({task.post_process_func.__name__}): {e}, "
                    f"原始响应: {task.answer[:200]}...",
                )
                if is_cache_hit:
                    llm_cache.delete(cache_key)
                if attempt < max_retries:
                    continue
                raise

        # 成功：缓存新结果
        if not is_cache_hit:
            llm_cache.set(cache_key, task.think, task.answer)

        logger.debug_context(ctx, f"{request_key} attempt {attempt} 成功")
        return task.result

    raise ValueError(f"LLM查询失败（已重试{max_retries}次）")


def query_llm_with_postprocess(
    ctx,
    task: Task,
    model_name: str,
    max_retries: int = 3,
    retry_delay: int = 60,
    use_cache: bool = False,
    **kwargs,
):
    """
    带prompt渲染、后处理和缓存的LLM查询（对外主接口）。

    流程: prompt渲染 → 构造消息 → LLM调用(含缓存/重试) → 后处理 → 返回结果

    Args:
        ctx: TRPC上下文
        task: Task对象，包含 stage/prompt_name/variables 等信息
        model_name: 模型名称
        max_retries: 最大重试次数
        retry_delay: 重试间隔基数（秒），实际等待 = retry_delay * attempt
        use_cache: 是否启用缓存
        **kwargs: 传递给LLM的额外参数（temperature, max_new_tokens 等）
    """
    debug_info = f"{task.stage}_{task.prompt_name}_{task.debug_info}"
    request_key = f"{task.stage}_{task.prompt_name}"

    try:
        if trpc.config.global_config_obj.global_config.namespace == "Development":
            use_cache = True
    except Exception:
        pass

    try:
        # 渲染prompt
        task.query = prompt_manager.format_prompt(
            stage=task.stage,
            name=task.prompt_name,
            variables=task.variables,
        )
        logger.debug_context(ctx, f"{debug_info} query: {task.query}")
        logger.info_context(ctx, f"{debug_info} query length: {len(task.query)}")

        # 构造消息和LLM参数
        messages = _build_messages(task.query)
        llm_params = {"temperature": 0.6, "max_new_tokens": 32768}
        if "claude" in model_name.lower():
            llm_params["max_tokens"] = llm_params["max_new_tokens"]
        llm_params.update(kwargs)

        # 调用LLM
        result = _call_llm_with_cache(
            ctx,
            task,
            model_name,
            messages,
            request_key,
            llm_params,
            max_retries,
            retry_delay,
            use_cache,
        )

        logger.debug_context(ctx, f"{debug_info} answer: {task.answer}")
        return result
    except Exception as e:
        logger.error_context(ctx, f"{debug_info} error: {e}")
        raise
