import os
from functools import lru_cache
import re

import yaml

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


class PromptManager:
    def __init__(self):
        self.prompt_data = {}

    @lru_cache(maxsize=128)
    def _load_file_raw(self, path: str):
        """加载原始文件内容"""
        full_path = os.path.join(PROMPTS_DIR, path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Prompt file not found at: {full_path}")

        with open(full_path, "r", encoding="utf-8") as f:
            if path.endswith(".yaml") or path.endswith(".yml"):
                return yaml.safe_load(f)
            else:
                return f.read()

    def _resolve_includes(self, content: str, depth: int = 0) -> str:
        """递归替换 @@INCLUDE: ... 占位符"""
        if depth > 10:
            raise ValueError("Include depth exceeded 10 levels")

        pattern = r'@@INCLUDE:\s*([^\n]+)'

        def replace_include(match):
            module_ref = match.group(1).strip()
            if not module_ref.endswith('.txt'):
                module_ref += '.txt'

            try:
                module_content = self._load_file_raw(module_ref)
                if isinstance(module_content, str):
                    return self._resolve_includes(module_content, depth + 1)
                return str(module_content)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Failed to include {module_ref}: {e}")

        resolved = re.sub(pattern, replace_include, content)

        # 检查是否还有未替换的占位符
        if re.search(pattern, resolved):
            raise ValueError("Unresolved includes in content")

        return resolved

    def _load_prompt_file(self, path: str):
        """加载提示词文件，自动替换占位符"""
        content = self._load_file_raw(path)

        if isinstance(content, dict):
            return content

        try:
            resolved = self._resolve_includes(content)
            return resolved
        except Exception as e:
            raise ValueError(f"Error resolving includes in {path}: {e}")

    def get_prompt(self, domain: str, stage: str, name: str, file_type: str = "txt") -> str:
        """获取原始提示词模板"""
        cache_key = f"{domain}_{stage}_{name}"
        if cache_key in self.prompt_data:
            return self.prompt_data[cache_key].strip()

        path = os.path.join(domain, stage, f"{name}.{file_type}")
        content = self._load_prompt_file(path)

        if isinstance(content, dict):
            content = content.get("template", "")

        self.prompt_data[cache_key] = content
        return content

    def format_prompt(
        self, domain: str, stage: str, name: str, variables: dict, file_type: str = "txt"
    ) -> str:
        """获取并格式化提示词"""
        template = self.get_prompt(domain, stage, name, file_type)
        try:
            for key, value in variables.items():
                template = template.replace(f"{{{key}}}", str(value))
            return template
        except KeyError as e:
            raise ValueError(f"Missing variable in prompt formatting: {e}")


# 创建一个单例，方便在项目中各处调用
prompt_manager = PromptManager()