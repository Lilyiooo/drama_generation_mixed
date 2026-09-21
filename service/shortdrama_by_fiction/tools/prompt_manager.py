import os
from functools import lru_cache
import yaml

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")


class PromptManager:
    @lru_cache(maxsize=128)  # 使用缓存避免重复读取文件
    def _load_prompt_file(self, path: str):
        full_path = os.path.join(PROMPTS_DIR, path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Prompt file not found at: {full_path}")

        with open(full_path, "r", encoding="utf-8") as f:
            if path.endswith(".yaml") or path.endswith(".yml"):
                return yaml.safe_load(f)
            else:
                return f.read()

    def get_prompt(self, stage: str, name: str, file_type: str = "yaml") -> str:
        """获取原始提示词模板"""
        path = os.path.join(stage, f"{name}.{file_type}")
        print("loading prompt from", name)
        content = self._load_prompt_file(path)
        optim = None
        if isinstance(content, dict):  # 如果是 YAML 加载的字典
            for ii in content.keys():
                if ii.startswith("template"):
                    template_key = ii
                else:
                    optim_key = ii
            return content[template_key], content[optim_key]
        return content, optim

    def format_prompt(
        self, stage: str, name: str, variables: dict, file_type: str = "yaml", prmtype: str = "raw"
    ) -> str:
        """获取并格式化提示词"""
        template, optim_template = self.get_prompt(stage, name, file_type)
        #if prmtype=="raw":
        #    template = raw_template
        #else:
        #    template = optim_template
        try:
            print("keys to replace", variables.keys())
            for key, value in variables.items():
                template = template.replace(f"{{{key}}}", str(value))
            return template, optim_template 
        except KeyError as e:
            raise ValueError(f"Missing prompt: {e} for {stage} {name} {file_type}")

    def get_optim_prompt(self, template: str, variables: dict):
        try:
            for key, value in variables.items():
                template = template.replace(f"{{{key}}}", str(value))
            return template 
        except KeyError as e:
            raise ValueError(f"Missing prompt: {e} for optim prompt")            




# 创建一个单例，方便在项目中各处调用
prompt_manager = PromptManager()


if __name__ == "__main__":
    variables = {"Plot": "这是情节内容", "RoleList": "这是角色列表"}
    prompt, optim = prompt_manager.format_prompt("diagnosis", "padding_analyze_v1", variables)
    print(prompt)
