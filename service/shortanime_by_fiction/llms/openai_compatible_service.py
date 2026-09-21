import os

import yaml
import requests


class OpenAICompatibleClient:
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
        url = config["API_URLS"]["openai_compatible"]
        token = config["API_KEYS"]["openai_compatible"]
        return url, token

    def chat(
        self,
        model_name,
        messages,
        enable_stream=False,
        **llm_params,
    ):
        if messages[-1]["role"] == "assistant":
            messages = messages[:-1]

        payload = {
            "model": model_name,
            "messages": messages,
            "stream": enable_stream,
        }
        payload.update(llm_params)

        rsp = requests.post(
            self.url,
            headers=self.headers,
            json=payload,
            stream=enable_stream,
            timeout=(5, 600),
        )
        if rsp.status_code != 200:
            raise RuntimeError(f"Error: {rsp.status_code}, {rsp.text}")

        data = rsp.json()
        think = data["choices"][0]["message"].get("reasoning_content", "")
        answer = data["choices"][0]["message"]["content"]
        usage = {
            "finish_reason": data["choices"][0].get("finish_reason", ""),
            **data.get("usage", {}),
        }
        return think, answer, usage
