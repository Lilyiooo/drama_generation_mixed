"""Task数据类定义"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, Callable


@dataclass
class Task:
    """LLM任务数据类"""

    stage: str
    prompt_name: str
    variables: dict
    debug_info: str = ""
    query: str = None
    post_process_func: Optional[Callable] = None
    post_process_args: Tuple = field(default_factory=tuple)
    post_process_kwargs: Dict[str, Any] = field(default_factory=dict)

    think: str = None  # 模型输出
    answer: str = None  # 模型输出
    result: Optional[Any] = None  # 后处理结果
