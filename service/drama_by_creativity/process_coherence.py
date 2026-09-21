import asyncio
import re
import random
from drama_local.runtime import logger

from .utils import *
from .script_postprocessing import coherence_adjustment_enabled
from .llm_inference import llm_inference_json
from .prompts import POLISH_PLOT_PROMPT

import jinja2
from .local_llm import LLM


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
SLEEP_TIME = 60

async def polish_plot(ctx, prev_plot, next_plot):
    if not coherence_adjustment_enabled():
        return prev_plot, next_plot
    # 设置模型
    script_config = get_model_config("script_model")
    llm_service = LLM(
        model=script_config["model"],
        system_prompt=script_config["system_prompt"],
        temperature=script_config["temperature"],
        top_p=script_config["top_p"],
        top_k=script_config["top_k"],
        max_length=script_config["max_length"],
        max_new_tokens=script_config["max_new_tokens"],
        template=jinja2.Template(POLISH_PLOT_PROMPT)
    )
    
    params = {
        "PrevShortScript": prev_plot,
        "NextShortScript": next_plot,
    }

    last_error = ""
    # 首次请求 + 最多三次重试；每次返回都实际校验。
    for attempt in range(4):
        ret = (await llm_service.request(ctx, params, f"{random.randint(0, 1000)}")).response
        try:
            return split_polished_scripts(ret)
        except ValueError as error:
            last_error = str(error)
            logger.error_context(ctx, f"一致性润色分隔符校验失败（{attempt + 1}/4）：{last_error}")
            if attempt < 3:
                await asyncio.sleep(SLEEP_TIME)

    logger.error_context(ctx, f"一致性调整失败，返回原始生成剧本。原因：{last_error}")
    return prev_plot, next_plot


def split_polished_scripts(text):
    # 兼容模型将换行输出为字面量反斜杠+n；只识别独立分隔行。
    # 不解码全文，不处理后续分隔符，保留当前集剩余全部内容。
    separator = re.search(r"(?m)^(?:\\n)*####(?:\\n)*\r?$", text)
    if separator is None:
        raise ValueError("未找到上一集与当前集之间的独立 #### 分隔行")
    previous = text[:separator.start()]
    current = text[separator.end():]
    # 仅消耗分隔行两侧各一个换行，保留正文原有的所有空白。
    if previous.endswith("\r\n"):
        previous = previous[:-2]
    elif previous.endswith("\n"):
        previous = previous[:-1]
    if current.startswith("\n"):
        current = current[1:]
    if not previous.strip() or not current.strip():
        raise ValueError("分隔符两侧剧本不能为空")
    return previous, current
