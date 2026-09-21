# -*- coding: utf-8 -*-
"""This is server module"""
import dataclasses, dataclasses_json
import os
import sys
from typing import List

import gflags

import trpc
import trpc_cos
import yaml
# from sqlalchemy import create_engine
from trpc.log import logger
import cos_orm
from service.drama_by_creativity import DramaByCreativity
from trpc_script_drama_operator import rpc, pb

from service.drama_by_fiction import DramaByFiction
from service.shortanime_by_fiction import ShortDramaByFiction

# add the stub to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "stub")))  # noqa
if __name__ != '__main__':  # noqa
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # noqa

try:
    # import sdk and plugins
    # pylint: disable=unused-import
    # import tconf
    #import tjg_opentracing
    #import trpc_log_atta
    #import trpc_metrics_m007
    import trpc_naming_polaris
    import trpc_validation
    #import trpc_opentracing_tjg
    import trpc_opentelemetry_galileo
    # DO NOT remove the plugin below!
    #import trpc_metrics_runtime
    # pylint: enable=unused-import
except ImportError as e:
    logger.info(f"fail to import plugins {e}")
    # raise e

# gflags cmd
FLAGS = gflags.FLAGS

# define config file
gflags.DEFINE_string('conf', os.path.abspath(os.path.join(os.path.dirname(__file__), "trpc_python.yaml")),
                     'trpc python framework config file')

def usage():
    """使用方式"""

    help_str = f"\nFlags from {__file__} :\n"
    for name, val in FLAGS.FlagDict().items():
        help_str += f"  --{name} ({val.help}) val: {val.value} type: {type(val.value).__name__} \n"
        help_str += f"    default : {val.default} \n"
    help_str += "\nstart server with conf, for example: `python3 trpc_main.py  --conf=trpc_python.yaml` \n"
    print(help_str)
    sys.exit(1)

def parse_args(argv: List[str]):
    """解析命令行参数
    :param argv: 命令行参数
    """

    if "-c" in argv:
        index = argv.index("-c")
        argv[index] = "--conf"
    try:
        res = FLAGS(argv)
        if len(res) > 1:
            print(f"args: `{res[1:]}` is invalid !!! \n")
    except Exception as err:  # pylint: disable=broad-exception-caught
        if isinstance(err, gflags.UnrecognizedFlagError):
            if err.flagname in ["help", "h"]:
                usage()
        print(f"{err}\n")
        usage()

def serve(conf_path: str):
    """启动服务
    :param conf_path: 配置文件路径
    """
    svr = trpc.new(conf_path)

    cos = trpc_cos.Client(f"trpc.{trpc.config.global_config_obj.server.app}.{trpc.config.global_config_obj.server.server}.CommonCos")

    rpc.register_DramaByCreativityServicer_server(svr.get_service(f"trpc.{trpc.config.global_config_obj.server.app}.{trpc.config.global_config_obj.server.server}.DramaByCreativity"), DramaByCreativity(cos))
    rpc.register_DramaByFictionServicer_server(svr.get_service(f"trpc.{trpc.config.global_config_obj.server.app}.{trpc.config.global_config_obj.server.server}.DramaByFiction"), DramaByFiction(cos))
    rpc.register_DramaByFictionServicer_server(svr.get_service(f"trpc.{trpc.config.global_config_obj.server.app}.{trpc.config.global_config_obj.server.server}.ShortDramaByFiction"), ShortDramaByFiction(cos))
    rpc.register_DramaByFictionServicer_server(svr.get_service(f"trpc.{trpc.config.global_config_obj.server.app}.{trpc.config.global_config_obj.server.server}.ShortAnimeByFiction"), ShortDramaByFiction(cos))

    svr.serve()

def main():
    """主函数"""

    # 解析命令行参数
    parse_args(sys.argv)
    logger.info("load config %s", FLAGS.conf)

    # 启动服务
    serve(FLAGS.conf)

if __name__ == '__main__':
    sys.exit(main())
