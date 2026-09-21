"""LLM调用相关功能"""
import json
import time
import traceback
from typing import Tuple, Dict, Any, Optional, List, Callable

import re
import json_repair
from trpc.log import logger

from .task import Task
from .llm_services import VenusClient
from .taiji_llm_service import TaijiClient
from .openai_compatible_service import OpenAICompatibleClient
from .llm_cache import llm_cache
from .prompt_manager import prompt_manager
from .llm_io_saver import save_llm_io
from service.shortanime_by_fiction.configs.llm_config import llm_config


class ContextLimitExceededError(Exception):
    """上下文限制超出异常，不应该重试"""
    pass


class ProhibitedContentError(Exception):
    """违禁内容异常，不应该重试"""
    pass


class ContentFilterError(Exception):
    """内容过滤异常，不应该重试"""
    pass


def validate_json_schema(json_data, schema):
    import json
    import json_repair
    from jsonschema import validate, ValidationError

    try:
        if isinstance(json_data, str):
            json_data = json_repair.loads(json_data)
        if isinstance(schema, str):
            schema = json_repair.loads(schema)
        validate(instance=json_data, schema=schema)
        json_str = json.dumps(json_data, ensure_ascii=False)
        return True, json_str
    except json.JSONDecodeError as e:
        return False, f"JSON解析错误: {str(e)}"
    except ValidationError as e:
        return False, f"Schema验证失败: {str(e)}"
    except Exception as e:
        return False, f"验证过程中发生错误: {str(e)}"


def post_extract_json(response):
    """从响应中提取JSON数据"""
    json_data = json_repair.loads(response)
    return json_data


def _run_llm_call(
    ctx,
    task,
    model_name,
    temperature=llm_config.call.temperature,
    max_new_tokens=llm_config.call.max_new_tokens,
    enable_stream=False,
    request_key="",
    max_retries=llm_config.call.max_retries,
    retry_delay=llm_config.call.retry_delay,
    use_cache=llm_config.call.use_cache,
    llm_provider=llm_config.call.llm_provider,
    **kwargs,
):
    """执行LLM调用，含缓存与重试"""
    if isinstance(task.query, str):
        messages = [
            {"role": "user", "content": task.query},
        ]
    elif isinstance(task.query, list):
        messages = task.query
    else:
        raise ValueError("query must be str or list.")

    debug_info = f"[{model_name}]{task.stage}:{task.prompt_name}:{task.debug_info}"
    logger.debug_context(ctx, f"{debug_info} query: {task.query}")
    logger.info_context(ctx, f"{debug_info} query length: {len(task.query) if isinstance(task.query, str) else len(str(task.query))}")

    client = TaijiClient() if llm_provider == "taiji" else OpenAICompatibleClient() if llm_provider == "openai_compatible" else VenusClient()

    llm_params = {
        "temperature": temperature,
        # "thinking_enabled": False,
        "max_new_tokens": max_new_tokens,
    }
    if kwargs:
        llm_params.update(kwargs)

    # 生成缓存键
    cache_key = llm_cache._generate_cache_key(task.query, model_name, temperature, max_new_tokens, **kwargs)

    for attempt in range(1, max_retries + 1):

        is_cache_hit = False

        # 只在第一次尝试时检查缓存
        if attempt == 1 and use_cache:
            cached_result = llm_cache.get(cache_key)
            if cached_result is not None:
                task.think, task.answer = cached_result
                is_cache_hit = True
                logger.info_context(ctx, f"{request_key} response {attempt} cache hit...")
                logger.debug_context(ctx, f"{request_key} answer:  {task.answer}")
            else:
                logger.info_context(ctx, f"{request_key} response {attempt} cache miss")

        try:
            # 如果没有缓存命中，调用LLM
            if not is_cache_hit:
                try:
                    task.think, task.answer, usage = client.chat(
                        model_name,
                        messages,
                        enable_stream,
                        **llm_params,
                    )
                    logger.debug_context(ctx, f"{request_key} answer:  {task.answer}")
                    logger.info_context(ctx, f"{request_key} usage: {usage}")
                except Exception as e:
                    # 错误码查询: https://iwiki.woa.com/p/4018506465?from=iWiki_search
                    m = re.search(r'(\{.*\})', str(e), re.DOTALL)
                    err = json.loads(m.group(1)) if m else {}
                    err_code = (err.get('error') or {}).get('code', 0)
                    err_msg = (err.get('error') or {}).get('message', '')
                    if err_code == '4003':
                        raise ContextLimitExceededError(f"上下文限制超出: {e}") from e
                    elif err_msg == 'PROHIBITED_CONTENT':
                        raise ProhibitedContentError(f"违禁内容: {e}") from e
                    elif 'content_filter' in str(e):
                        # raise ContentFilterError(f"内容过滤: {e}") from e
                        logger.warning_context(ctx, f"{request_key} content filter: {e}")
                        task.think, task.answer = '', ''
                    else:
                        logger.error_context(ctx, f"{request_key} attempt {attempt} failed during response generation: {e}")
                        logger.info_context(ctx, f"{request_key} sleeping {retry_delay * attempt}s before retry...")
                        time.sleep(retry_delay * attempt)
                        continue

            logger.info_context(ctx, f"{request_key} length of think:  {len(task.think)}")
            logger.info_context(ctx, f"{request_key} length of answer:  {len(task.answer)}")
            task.result = task.answer

            # 进行validate
            if task.validate_func is not None:
                try:
                    ok, msg = task.validate_func(task.answer, *task.validate_args, **task.validate_kwargs)
                    if not ok:
                        logger.warning_context(ctx,
                            f"{request_key} response attempt {attempt} validate failed: {msg}, answer: {task.answer}"
                        )
                        if is_cache_hit:
                            llm_cache.delete(cache_key)
                        continue
                except Exception as e:
                    logger.warning_context(ctx,
                        f"{request_key} response attempt {attempt} validate error: {e}"
                    )
                    if is_cache_hit:
                        llm_cache.delete(cache_key)
                    continue

            # 进行后处理
            if task.post_process_func is not None:
                try:
                    task.result = task.post_process_func(task.answer, *task.post_process_args, **task.post_process_kwargs)
                except Exception as e:
                    logger.warning_context(ctx,
                        f"{request_key} attempt {attempt} post_process failed [{type(e).__name__}: {e}]\nfunc: {task.post_process_func.__name__}\ntraceback: {traceback.format_exc()}\nresponse: {task.answer}"
                    )
                    if is_cache_hit:
                        llm_cache.delete(cache_key)
                    continue
            
            # 只有新调用LLM且后处理成功时才缓存结果
            if not is_cache_hit and use_cache:
                llm_cache.set(cache_key, task.think, task.answer)

            logger.debug_context(ctx, f"{request_key} response attempt {attempt} success")
            return task.result
        except ContextLimitExceededError:
            logger.error_context(ctx, f"{request_key} context limit exceeded, aborting")
            raise
        except ProhibitedContentError:
            logger.error_context(ctx, f"{request_key} prohibited content, aborting")
            raise
        except ContentFilterError:
            logger.error_context(ctx, f"{request_key} content filter triggered, aborting")
            raise
        except Exception as e:
            logger.error_context(ctx, f"{request_key} response generation overall failed after {attempt} attempts: {e}")
            logger.info_context(ctx, f"{request_key} sleeping {retry_delay * attempt}s before retry...")
            time.sleep(retry_delay * attempt)

    logger.error_context(ctx, f"LLM query failed after {max_retries} attempts")
    raise ValueError(f"大模型调用失败(已重试{attempt}次)")


def query_llm(
    ctx,
    task: Task,
    model_name: str,
    max_retries=llm_config.call.max_retries,
    llm_provider=llm_config.call.llm_provider,
    **kwargs,
):
    """
    通用的LLM查询函数

    Args:
        ctx: TRPC上下文
        model_name: 模型名称
        max_retries: 最大重试次数

    Returns:
        LLM返回的答案字符串，失败返回空字符串
    """
    try:
        debug_info = f"{task.stage}:{task.prompt_name}:{task.debug_info}"
        task.query = prompt_manager.format_prompt(
            domain=task.domain,
            stage=task.stage,
            name=task.prompt_name,
            variables=task.variables,
        )

        result = _run_llm_call(
            ctx,
            task,
            model_name=model_name,
            request_key=f"{task.stage}_{task.prompt_name}",
            max_retries=max_retries,
            max_tokens=llm_config.call.max_tokens,
            llm_provider=llm_provider,
            **kwargs,
        )
        logger.debug_context(ctx, f"{debug_info} result: {task.result}")

        if llm_config.call.use_io_log:
            save_llm_io(
                request_key=debug_info,
                model_name=model_name,
                stage=task.stage,
                prompt_name=task.prompt_name,
                messages=task.query if isinstance(task.query, list) else [{"role": "user", "content": task.query}],
                think=task.think,
                answer=task.answer,
                extra=task.extra,
            )

        return result
    except Exception as e:
        logger.error_context(ctx, f"{debug_info} error: {e}")
        raise
