"""Exercise the adapter over HTTP without loading the RPC application."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import unittest
from unittest.mock import patch

import jinja2

spec = importlib.util.spec_from_file_location(
    "local_llm_under_test", Path(__file__).resolve().parents[1] / "local_llm.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LocalLLMTest(unittest.TestCase):
    def setUp(self):
        test = self
        self.status = 200
        self.body = {"choices": [{"message": {"content": '{"ok":true}',
                                             "reasoning_content": "private reasoning"},
                                  "finish_reason": "stop"}]}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                test.path = self.path
                test.payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                test.authorization = self.headers.get("Authorization")
                self.send_response(test.status)
                self.end_headers()
                self.wfile.write(json.dumps(test.body).encode())

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.env = patch.dict(os.environ, {
            "DRAMA_LLM_BASE_URL": f"http://127.0.0.1:{self.server.server_port}/v1/",
            "DRAMA_LLM_MODEL": "test-local",
            "DRAMA_LLM_TIMEOUT": "2",
            "HTTP_PROXY": "http://127.0.0.1:1",
        }, clear=True)
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self):
        client = module.LLM(model="configured-model", system_prompt="system",
                            temperature=0.4, top_p=1, top_k=50,
                            max_length=64000, max_new_tokens=123,
                            template=jinja2.Template("Write {{ topic }}"))
        return asyncio.run(client.request(None, {"topic": "故事"}, "test-id"))

    def test_template_payload_and_response(self):
        self.assertEqual(self.request().response, '{"ok":true}')
        self.assertEqual(self.path, "/v1/chat/completions")
        self.assertEqual(self.payload["messages"], [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Write 故事"}])
        self.assertEqual(self.payload["model"], "test-local")
        self.assertEqual(self.payload["max_tokens"], 123)
        self.assertEqual(self.payload["top_k"], 50)
        self.assertFalse(self.payload["stream"])
        self.assertFalse(self.payload["chat_template_kwargs"]["enable_thinking"])
        self.assertNotIn("max_length", self.payload)
        self.assertIsNone(self.authorization)

    def test_environment_overrides(self):
        with patch.dict(os.environ, {"DRAMA_LLM_MAX_TOKENS": "0",
                                     "DRAMA_LLM_THINKING": "1",
                                     "DRAMA_LLM_API_KEY": "test-key"}):
            self.request()
        self.assertNotIn("max_tokens", self.payload)
        self.assertTrue(self.payload["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(self.authorization, "Bearer test-key")
        with patch.dict(os.environ, {"DRAMA_LLM_MAX_TOKENS": "456"}):
            self.request()
        self.assertEqual(self.payload["max_tokens"], 456)

    def test_http_failure(self):
        self.status = 400
        self.body = {"error": "context length exceeded"}
        with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
            self.request()

    def test_invalid_empty_and_truncated_responses(self):
        for body in ({"choices": []}, {"choices": [{"message": {"content": None}}]},
                     {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]}):
            with self.subTest(body=body):
                self.body = body
                with self.assertRaises(ValueError):
                    self.request()


if __name__ == "__main__":
    unittest.main()
