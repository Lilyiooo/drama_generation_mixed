"""
测试辅助工具：protobuf 消息的 JSON 序列化/反序列化、打印等。
"""

import os
import json

from google.protobuf.json_format import MessageToJson, Parse
from google.protobuf.text_format import MessageToString


def format_print(data):
    """格式化打印 protobuf 消息（中文友好）"""
    print(MessageToString(data, as_utf8=True))


def save_pb_json(test_data_dir: str, filename: str, msg):
    """将 protobuf 消息保存为 JSON 文件（中文友好）"""
    json_str = MessageToJson(
        msg,
        preserving_proto_field_name=True,
        including_default_value_fields=True,
    )
    filepath = os.path.join(test_data_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        # 重新 dump 一次以保证缩进美观
        f.write(json.dumps(json.loads(json_str), ensure_ascii=False, indent=2))
    print(f"  [SAVED] {filepath}", flush=True)


def load_pb_json(test_data_dir: str, filename: str, msg_class):
    """从 JSON 文件加载 protobuf 消息"""
    filepath = os.path.join(test_data_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        json_str = f.read()
    msg = Parse(json_str, msg_class())
    print(f"  [LOADED] {filepath}", flush=True)
    return msg


def save_and_print(test_data_dir: str, step_name: str, req_filename: str, rsp_filename: str, request, response):
    """保存请求和响应，并打印摘要"""
    print(f"\n{'='*60}")
    print(f"  {step_name}")
    print(f"{'='*60}")
    save_pb_json(test_data_dir, req_filename, request)
    save_pb_json(test_data_dir, rsp_filename, response)
    format_print(response)
