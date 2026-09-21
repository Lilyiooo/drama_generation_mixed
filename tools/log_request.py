import functools
import time

from trpc import context
from trpc.log import logger


def log_request(func):
    """
    装饰器函数：只打印执行进度，不打印完整 request/response 内容
    """

    @functools.wraps(func)
    async def wrapper(self, ctx: context.Context, request, *args, **kwargs):
        func_name = func.__name__

        # 尽量提取 story_id 作为日志上下文字段（失败则忽略，不打印完整请求）
        try:
            logger.with_context_fields(ctx, {"StoryID": request.story_info.story_id})
        except Exception:
            pass

        logger.info_context(ctx, f"开始 {func_name} ...")
        start = time.time()
        result = await func(self, ctx, request, *args, **kwargs)
        cost_time = time.time() - start
        logger.info_context(ctx, f"{func_name} 完成，耗时 {cost_time:.2f}秒")
        return result

    return wrapper
