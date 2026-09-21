import time
import random
from drama_local.runtime import logger

from .utils import *
from .llm_inference import llm_inference_json
from .prompts import EXTRACT_ABS_PROMPT

import jinja2
from .local_llm import LLM


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

async def update_plot_abs(ctx, now_episode_abs, now_plot):
    # 设置模型
    auxiliary_config = get_model_config("auxiliary_model")
    llm_service = LLM(
        model=auxiliary_config["model"],
        system_prompt=auxiliary_config["system_prompt"],
        temperature=auxiliary_config["temperature"],
        top_p=auxiliary_config["top_p"],
        top_k=auxiliary_config["top_k"],
        max_length=auxiliary_config["max_length"],
        max_new_tokens=auxiliary_config["max_new_tokens"],
        template=jinja2.Template(EXTRACT_ABS_PROMPT)
    )

    params = {
        "PrevAbs": now_episode_abs,
        "NowPlot": now_plot,
    }
    logger.info_context(ctx, f"抽取摘要的query: {params}")

    ret = (await llm_service.request(ctx, params, f"{random.randint(0, 1000)}")).response
    return ret
