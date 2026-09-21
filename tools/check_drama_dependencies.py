"""Report missing startup dependencies without importing the application."""
import importlib.util
import sys

MODULES = {
    "yaml": "PyYAML",
    "requests": "requests",
    "jinja2": "Jinja2",
    "json_repair": "json-repair",
}


def main():
    missing = [name for name in MODULES if importlib.util.find_spec(name) is None]
    if not missing:
        return 0
    print(f"当前 Python：{sys.executable}", file=sys.stderr)
    print("缺少以下启动依赖：", file=sys.stderr)
    for name in missing:
        print(f"  {name}: {MODULES[name]}", file=sys.stderr)
    print("请使用当前 Python 安装：python -m pip install -r requirements-local.txt", file=sys.stderr)
    print("本地创作链路不需要腾讯 tRPC、COS 或内部协议包。", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
