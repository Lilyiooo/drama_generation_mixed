# -*- coding: utf-8 -*-
import os
import traceback

import uuid
import json

import yaml
import requests
import sseclient


class VenusClient(object):
    def __init__(self, model_name):
        self.model_name = model_name if model_name else "deepseek-r1-local-II"
        trpc_path = os.path.join(os.path.dirname(__file__), "../../../trpc_python.yaml")
        with open(trpc_path, "r") as f:
            trpc_config = yaml.safe_load(f)
        self.url = trpc_config["API_URLS"]["venus"]
        token = trpc_config["API_KEYS"]["venus"]
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

    def _process_stream_response(self, resp, return_think):
        client = sseclient.SSEClient(resp)
        think, answer = "", ""
        for event in client.events():
            # print(event.data.strip())
            try:
                if event.data.strip() == "[DONE]":  # venus流式输出结束
                    resp.close()
                    break

                if not event.data.strip():
                    continue

                data = json.loads(event.data.strip())
                # 大模型输出, venus 思考放在reasoning_content，答案放在content中
                if "choices" in data and data["choices"][0]["delta"].get("content", ""):
                    answer += data["choices"][0]["delta"]["content"]
                if "choices" in data and data["choices"][0]["delta"].get(
                    "reasoning_content", ""
                ):
                    think += data["choices"][0]["delta"]["reasoning_content"]
            except Exception as e:
                e_msg = traceback.format_exc()
                print(e_msg, e)
                print(event.data.strip())
        return (think, answer) if return_think else answer

    def _process_non_stream_response(self, resp, return_think):
        data = resp.json()
        think = data["choices"][0]["message"].get("reasoning_content", "")
        answer = data["choices"][0]["message"]["content"]
        return (think, answer) if return_think else answer

    def chat(
        self,
        messages,
        temperature=0.6,
        top_p=0.95,
        top_k=50,
        max_new_tokens=8192,
        enable_stream=True,
        return_think=True,
        **kwargs,
    ):
        if messages[-1]["role"] == "assistant":
            messages = messages[:-1]  # 去掉最后一个assistant

        payload = {
            "task_id": "test_" + str(uuid.uuid4()),
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "do_sample": True,
            "top_p": top_p,
            "top_k": top_k,
            "stream": enable_stream,
        }
        payload.update(kwargs)

        rsp = requests.post(
            self.url,
            headers=self.headers,
            json=payload,
            stream=enable_stream,
            timeout=(5, 600),
        )
        if rsp.status_code != 200:
            raise RuntimeError(f"Error: {rsp.status_code}, {rsp.text}")

        if enable_stream:
            return self._process_stream_response(rsp, return_think)
        else:
            return self._process_non_stream_response(rsp, return_think)


def query_llm(
    query,
    model_name=None,
    temperature=0.6,
    top_p=0.95,
    top_k=50,
    max_new_tokens=32768,
    enable_stream=False,
    return_think=True,
    model_type="taiji",
    wsid="11343",
    **kwargs,
):
    if isinstance(query, str):
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": query},
        ]
    elif isinstance(query, list):
        messages = query
    else:
        raise ValueError("query must be str or list.")
    client = VenusClient(model_name)
    return client.chat(
        messages,
        temperature,
        top_p,
        top_k,
        max_new_tokens,
        enable_stream,
        return_think,
        **kwargs,
    )


def validate_json_schema(json_data, schema):
    from jsonschema import validate, ValidationError
    import json_repair

    try:
        # 如果输入是字符串，尝试解析为JSON
        if isinstance(json_data, str):
            json_data = json_repair.loads(json_data)
        if isinstance(schema, str):
            schema = json_repair.loads(schema)
        # 执行验证
        validate(instance=json_data, schema=schema)
        json_str = json.dumps(json_data, ensure_ascii=False)
        return True, json_str

    except json.JSONDecodeError as e:
        return False, f"JSON解析错误: {str(e)}"
    except ValidationError as e:
        return False, f"Schema验证失败: {str(e)}"
    except Exception as e:
        return False, f"验证过程中发生错误: {str(e)}"


if __name__ == "__main__":
    think, answer = query_llm("你好", model_name="gemini-2.5-pro", model_type="venus")
    print(think, answer)
