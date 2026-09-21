"""Task数据类定义"""
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, List, Callable


@dataclass
class Task:
    """LLM任务数据类"""
    domain: str
    stage: str
    prompt_name: str
    variables: dict
    debug_info: str = ""
    query: str = None
    validate_func: Optional[Callable] = None
    validate_args: Tuple = field(default_factory=tuple)
    validate_kwargs: Dict[str, Any] = field(default_factory=dict)
    post_process_func: Optional[Callable] = None
    post_process_args: Tuple = field(default_factory=tuple)
    post_process_kwargs: Dict[str, Any] = field(default_factory=dict)

    extra: Dict[str, Any] = field(default_factory=dict)  # 附加信息（如 story_info/generate_input），仅用于日志，不影响执行

    think: str = None            # 模型输出
    answer: str = None           # 模型输出
    result: Optional[Any] = None # 后处理结果