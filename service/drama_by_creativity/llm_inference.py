import json
import time
import random
from drama_local.diagnostics import save_event
from drama_local.runtime import logger
import jinja2
from .local_llm import LLM

from .utils import *
from .prompts import ADJUST_JSON_FORMAT_PROMPT


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

async def llm_inference_json(
    params,
    llm_service,
    task_name,
    ctx,
    retry_times=3,
    check_times=3,
):
    auxiliary_config = get_model_config("auxiliary_model")
    llm_check_service = LLM(
        model=auxiliary_config["model"],
        system_prompt=auxiliary_config["system_prompt"],
        temperature=auxiliary_config["temperature"],
        top_p=auxiliary_config["top_p"],
        top_k=auxiliary_config["top_k"],
        max_length=auxiliary_config["max_length"],
        max_new_tokens=auxiliary_config["max_new_tokens"],
        template=jinja2.Template(ADJUST_JSON_FORMAT_PROMPT)
    )

    def remember(text, source, generation, repair):
        trace = save_event("json_attempts", task_name=task_name,
                           context=dict(getattr(ctx, "fields", {})),
                           generation_attempt=generation, repair_attempt=repair,
                           model_call=getattr(source, "last_trace_dir", None), text=text)
        llm_service.last_output = text
        llm_service.last_trace_dir = trace

    status = False
    for retry_id in range(retry_times):
        ret = (await llm_service.request(ctx, params, f"{random.randint(0, 1000)}")).response
        remember(ret, llm_service, retry_id + 1, 0)

        last_output_checked = False
        for check_id in range(check_times):
            last_output_checked = True
            try:
                processed_ret = post_process(ret)
                ret_json = json.loads(processed_ret)

                if not isinstance(ret_json, list) and not isinstance(ret_json, dict):
                    raise ValueError(f"转换的JSON结果不是list或dict格式：\n{ret_json}")

                save_event("json_validation", task_name=task_name, generation_attempt=retry_id+1,
                           repair_attempt=check_id, status="accepted",
                           source=llm_service.last_trace_dir)
                status = True
                break

            except Exception as e:
                llm_service.last_error = f"{type(e).__name__}: {e}"
                save_event("json_validation", task_name=task_name, generation_attempt=retry_id+1,
                           repair_attempt=check_id, status="rejected", reason=str(e),
                           source=llm_service.last_trace_dir, text=ret)
                logger.error_context(ctx, f"任务：{task_name} 的结果JSON解析失败, 尝试第{check_id+1}/{check_times}次修改结果为JSON格式。错误生成结果：{ret}\n错误信息：{e}")
                
                if "模型服务" in ret:
                    time.sleep(60)
                    break

                ret = (await llm_check_service.request(ctx, {"Ret": ret}, f"{random.randint(0, 1000)}")).response
                remember(ret, llm_check_service, retry_id + 1, check_id + 1)
                last_output_checked = False
                logger.info_context(ctx, f"任务：{task_name} ，第{check_id+1}/{check_times}次修改JSON格式，调整后结果：\n{ret}")

        if status:
            break
        else:
            save_event("json_retry", task_name=task_name, generation_attempt=retry_id+1,
                       next_action="regenerate" if retry_id+1 < retry_times else "return_empty_object",
                       last_output_checked=last_output_checked, source=llm_service.last_trace_dir, text=ret)
            logger.error_context(ctx, f"任务：{task_name} 的结果经过3次格式调整依旧失败，尝试第{retry_id+1}/{retry_times}次重新生成结果。")

    if not status:
        logger.error_context(ctx, f"任务：{task_name} 的结果经过3次重新生成依旧失败，返回空结果。")
        return {}

    return ret_json
