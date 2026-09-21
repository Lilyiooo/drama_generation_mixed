import functools
import time

from google.protobuf.text_format import MessageToString
from trpc import context
from trpc.log import logger


def log_request(func):
    """
    装饰器函数：在请求时打印请求参数
    """

    @functools.wraps(func)
    async def wrapper(self, ctx: context.Context, request, *args, **kwargs):
        # 获取函数名
        func_name = func.__name__
        try:
            logger.with_context_fields(ctx, {"task_id": ctx.get_client_message().server_meta_data["task_id"].decode("utf-8")})
            logger.with_context_fields(
                ctx,
                {
                    "StoryID": request.story_info.story_id,
                    **({"NovelID": request.generate_input.novel_id} if (hasattr(request, "generate_input") and hasattr(request.generate_input, "novel_id")) else {}),
                },
            )
            request_info = MessageToString(request, as_utf8=True)
            logger.info_context(ctx, f"{func_name} request:\n{request_info}")
        except Exception as e:
            logger.warning(f"{func_name} convert request error: {e}")
            logger.info_context(ctx, f"{func_name} request:\n{request}")

        start = time.time()
        # 调用原始函数
        result = await func(self, ctx, request, *args, **kwargs)
        cost_time = time.time() - start
        try:

            rsp_info = MessageToString(result, as_utf8=True)
            logger.info_context(ctx, f"{func_name} rsp:\n{rsp_info}")
        except Exception as e:
            logger.warning(f"{func_name} convert rsp error: {e}")
            logger.info_context(ctx, f"{func_name} rsp:\n{result}")

        logger.info_context(ctx, f"{func_name} cost time: {cost_time:.2f}秒")
        return result

    return wrapper