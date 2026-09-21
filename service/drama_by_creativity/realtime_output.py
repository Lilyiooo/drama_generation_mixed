"""
实时落盘工具：当环境变量 DRAMA_OUTPUT_DIR 存在时，自动保存中间生成结果到本地。
不影响原有逻辑，不落盘时零开销。
"""
import os
import json
import datetime
from drama_local.runtime import atomic_json

def save_realtime(path_parts: list, data):
    """
    落盘中间结果到 DRAMA_OUTPUT_DIR（如果已设置）。
    path_parts: 从输出根目录起的目录+文件名，如 ["04_episode_outline", "chunk_01.json"]
    """
    base = os.environ.get("DRAMA_OUTPUT_DIR")
    if not base:
        return
    *dirs, filename = path_parts
    full_dir = os.path.join(base, *dirs)
    os.makedirs(full_dir, exist_ok=True)
    filepath = os.path.join(full_dir, filename)
    atomic_json(filepath, json.loads(json.dumps(data, ensure_ascii=False, default=str)))
    print(f"  💾 [实时落盘] {filepath}  ({datetime.datetime.now().strftime('%H:%M:%S')})")
