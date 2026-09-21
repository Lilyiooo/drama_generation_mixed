"""桥段库开关的兼容读取与 Prompt 参考内容选择。"""

from __future__ import annotations

import os
from typing import Any


# 对外 protobuf 契约中建议加入：
#   bool disable_plot_retrieval = 7;
# 当前线上 stub 尚未包含该字段，所以也兼容读取 protobuf unknown field。
DISABLE_PLOT_RETRIEVAL_FIELD_NUMBER = 7
DISABLE_PLOT_RETRIEVAL_ENV = "DRAMA_DISABLE_PLOT_RETRIEVAL"


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data) and shift < 70:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
    raise ValueError("invalid protobuf varint")


def _unknown_bool_field(message: Any, field_number: int) -> bool | None:
    """从旧版 protobuf message 的 unknown fields 中读取新增 bool 字段。"""
    serialize = getattr(message, "SerializeToString", None)
    if not callable(serialize):
        return None

    try:
        data = serialize()
    except Exception:
        return None

    offset = 0
    found: bool | None = None
    try:
        while offset < len(data):
            tag, offset = _read_varint(data, offset)
            number = tag >> 3
            wire_type = tag & 0x07
            if wire_type == 0:
                value, offset = _read_varint(data, offset)
                if number == field_number:
                    found = bool(value)
            elif wire_type == 1:
                offset += 8
            elif wire_type == 2:
                length, offset = _read_varint(data, offset)
                offset += length
            elif wire_type == 5:
                offset += 4
            else:
                return found
            if offset > len(data):
                return found
    except (TypeError, ValueError):
        return found
    return found


def is_plot_retrieval_disabled(request_or_generate_input: Any) -> bool:
    """返回当前请求是否明确要求禁用桥段库检索。"""
    generate_input = getattr(request_or_generate_input, "generate_input", request_or_generate_input)

    # 新版 stub 发布后直接走正式字段。
    if hasattr(generate_input, "disable_plot_retrieval"):
        return bool(generate_input.disable_plot_retrieval)

    # 允许新版二进制客户端调用仍使用旧 stub 的服务端。
    unknown_value = _unknown_bool_field(generate_input, DISABLE_PLOT_RETRIEVAL_FIELD_NUMBER)
    if unknown_value is not None:
        return unknown_value

    # 本地脚本兼容入口；未设置时保持原有检索行为。
    return os.environ.get(DISABLE_PLOT_RETRIEVAL_ENV, "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def select_prompt_reference(generate_input: Any, retrieved_reference: str) -> str:
    """用户参考优先；禁用检索且无用户参考时返回空串以删除 Prompt 段落。"""
    user_reference = str(getattr(generate_input, "reference", "") or "").strip()
    if user_reference:
        return user_reference
    if is_plot_retrieval_disabled(generate_input):
        return ""
    return retrieved_reference or "暂无"

