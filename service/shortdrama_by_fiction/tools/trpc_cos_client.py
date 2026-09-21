# -*- coding: utf-8 -*-
"""
@Author: jaydenfu
@Date: 2020-06-17 16:21:08
LastEditTime: 2020-12-08 11:23:04
LastEditors: jaydenfu
@Description: cos client based on trpc-python
@FilePath: /trpc_cos/trpc_cos/client.py
"""

import asyncio
from collections import ChainMap
import hmac
import hashlib
import re
import time
from typing import Mapping, List, Callable
import urllib
import urllib.parse
from urllib.parse import quote

from trpc import client as tclient
from trpc import codec, context
from trpc.client import Options
from trpc.client.options import (
    with_req_head,
    with_protocol,
    with_serialization_type,
    with_target,
)
from trpc.codec.serialization import SerializationType
from trpc.http.client import new_rsp_object
from trpc.http.codec import ClientReqHeader


_TARGET_DSL_REG = (
    r"^innercos:\/\/(?P<secretid>\S*?):(?P<secretkey>\S*?)"
    r"@(?P<schema>\S*?):\/\/(?P<addr>\S*?)/(?P<bucket>\S*?)-(?P<appid>\S*?)\.(?P<domain>\S*?)$"
)


class Client:
    """
    cos client for operation in cos platform
    """

    __target_reg = re.compile(_TARGET_DSL_REG)  # regrex object for parsing target

    __service_name: str  # callee service name
    __host: str  # host for cos http request
    __secretid: str  # cos secretid
    __secretkey: str  # cos secretkey
    __real_addr: str  # cos service address
    __gheaders: Mapping  # http请求时公共headers
    __expired: int = 600  # 认证过期时间，10分钟
    __options: List[Callable] = None  # additional options for client invoke

    def __init__(self, service_name: str, options: List[Callable] = None):
        """
        :param config: config for cos request
        :param options: options for trpc client invoke
        """
        self.__service_name = service_name
        self.__options = list()
        # parse target dsl
        opts = Options()
        if options is not None:
            self.__options = options
            for each in options:
                each(opts)
        if opts.target == "":
            opts.load_client_config(service_name)
        self.__parse_target_dsl(opts.target)
        self.__gheaders = {}

    def __parse_target_dsl(self, target: str):
        """parse target dsk

        get secretid, secretkey, bucket, appid, domain, addr from target
        """
        res = self.__target_reg.fullmatch(target)
        if res is None:
            raise Exception("invalid target format")
        self.__secretid = res.group("secretid")
        self.__secretkey = res.group("secretkey")
        self.__host = "{}-{}.{}".format(
            res.group("bucket"), res.group("appid"), res.group("domain")
        )
        self.__real_addr = "{}://{}".format(res.group("schema"), res.group("addr"))

    async def head_bucket(self, ctx: context.Context):
        """
        head bucket asynchronously

        :param ctx: trpc context
        """
        uri = "/"
        method = "Head"
        params = dict()
        headers = {"Host": self.__host}
        sign, _ = self.__gen_sign(method, uri, headers, params)
        headers["Authorization"] = sign
        await self.__send_request(ctx, method, uri, headers, None)
        return

    def head_bucket_sync(self, ctx: context.Context):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.head_bucket(ctx))

    async def get_object(self, ctx: context.Context, uri: str) -> bytes:
        """
        get object from cos asynchronously

        :param ctx: trpc context
        :param uri: object uri

        :return raw data of object from cos
        """
        method = "Get"
        params = dict()
        headers = {"Host": self.__host}
        sign, _ = self.__gen_sign(method, uri, headers, params)
        headers["Authorization"] = sign
        rsp = await self.__send_request(ctx, method, uri, headers, None)
        return rsp

    def get_object_sync(self, ctx: context.Context, uri: str) -> bytes:
        """get object sync"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.get_object(ctx, uri))

    async def put_object(
        self,
        ctx: context.Context,
        uri: str,
        data: bytes,
        req_header: Mapping[str, str] = None,
    ) -> str:
        """
        put object to cos asynchronously

        :param ctx: trpc context
        :param uri: object uri
        :param data: raw data of object to cos
        :param header: request header to cos

        :return link url of object
        """
        uri = "/" + uri.lstrip("/")
        # uri = uri.lower()
        method = "Put"
        params = dict()
        headers = {"Host": self.__host, "Content-Length": str(len(data))}
        sign, _ = self.__gen_sign(method, uri, headers, params)
        headers["Authorization"] = sign
        if req_header is not None:
            headers.update(req_header)
        await self.__send_request(ctx, method, uri, headers, data)
        url = self.__host + uri
        return url

    def put_object_sync(
        self,
        ctx: context.Context,
        uri: str,
        data: bytes,
        req_header: Mapping[str, str] = None,
    ) -> str:
        """put object sync"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.put_object(ctx, uri, data, req_header))

    async def del_object(self, ctx: context.Context, uri: str):
        """
        delete object from cos asynchronously

        :param ctx: trpc context
        :param uri: object uri
        """
        method = "Delete"
        params = dict()
        headers = {"Host": self.__host}
        sign, _ = self.__gen_sign(method, uri, headers, params)
        headers["Authorization"] = sign
        await self.__send_request(ctx, method, uri, headers, None)
        return

    def del_object_sync(self, ctx: context.Context, uri: str):
        """del object sync"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.del_object(ctx, uri))

    @staticmethod
    def __sp_lower(data: str) -> str:
        ret = ""
        length = len(data)
        i = 0
        while i < length:
            if data[i] == "%" and i + 2 < length:
                ret += data[i]
                ret += data[i + 1].lower()
                ret += data[i + 2].lower()
                i += 3
            else:
                ret += data[i]
                i += 1
        return ret

    def __gen_sign(
        self, method: str, uri: str, headers: Mapping, params: Mapping
    ) -> List[str]:
        """
        generate sign for cos platform auth
        """
        uri = "/" + uri.lstrip("/")

        now = time.time()
        key_time = str(int(now) - self.__expired) + ";" + str(int(now) + self.__expired)

        sha1_hmac = hmac.new(
            self.__secretkey.encode("utf-8"), key_time.encode("utf-8"), hashlib.sha1
        )
        sign_key = sha1_hmac.hexdigest()

        sorted_values = sorted(params.items(), key=lambda val: val[0])
        format_param = urllib.parse.urlencode(sorted_values, quote_via=quote)
        format_param = Client.__sp_lower(format_param)

        headers = dict((k.lower(), v) for k, v in headers.items())
        sorted_values = sorted(headers.items(), key=lambda val: val[0])
        format_header = urllib.parse.urlencode(sorted_values, quote_via=quote)
        format_header = Client.__sp_lower(format_header)

        format_string = (
            method.lower()
            + "\n"
            + uri
            + "\n"
            + format_param
            + "\n"
            + format_header
            + "\n"
        )

        sha1 = hashlib.sha1(format_string.encode("utf-8"))
        string_to_sign = "sha1\n" + key_time + "\n" + sha1.hexdigest() + "\n"

        sha1_hmac = hmac.new(
            sign_key.encode("utf-8"), string_to_sign.encode("utf-8"), hashlib.sha1
        )
        q_header = ";".join(sorted(headers.keys())).lower()
        q_param = ";".join(sorted(params.keys())).lower()
        sign = (
            "q-sign-algorithm=sha1&q-ak=%s&q-sign-time=%s&q-key-time=%s&q-header-list=%s&q-url-param-list=%s&"
            "q-signature=%s"
        ) % (
            self.__secretid,
            key_time,
            key_time,
            q_header,
            q_param,
            sha1_hmac.hexdigest(),
        )
        return sign, format_param

    async def __send_request(
        self, ctx: context.Context, method: str, uri: str, headers: Mapping, data: bytes
    ) -> bytes:
        """
        common request method for all cos operation
        based on trpc http client

        :param ctx: trpc context
        :param method: http method, Get/Put/Delete
        :param uri: object uri
        :param headers: http request headers
        :param data: http request body

        :return http response body

        :raise TRPC Exception when invoke failed
        """
        uri = "/" + uri.lstrip("/")
        # set msg
        ctx, msg = codec.clone_client_message(ctx)
        msg.client_rpc_name = uri
        msg.callee_service = self.__service_name
        msg.callee_method = method
        if msg.namespace == "":
            msg.namespace = "Production"
        if msg.caller_service_name == "":
            msg.caller_service_name = "trpc_cos"
        if msg.env_name == "":
            msg.env_name = "prd"

        # set req header
        req_header = ClientReqHeader(method, self.__host)
        req_header.headers = dict()
        for key, val in ChainMap(self.__gheaders, headers).items():
            req_header.headers[key] = val
        # set options
        callopts = [
            with_req_head(req_header),
            with_protocol("http"),
            with_serialization_type(SerializationType.Noop),
            with_target(self.__real_addr),
        ]
        callopts = self.__options + callopts
        # invoke
        proxy = tclient.get_client()
        await proxy.invoke(ctx, data, new_rsp_object, callopts)
        rsp_head = ctx.get_client_message().client_rsp_head
        return rsp_head.body


def new_client(service_name: str, options: List[Callable] = None) -> Client:
    """
    create a cos client

    :param config: config for cos request
    :param options: options for trpc client invoke

    :return cos client
    """
    cli = Client(service_name, options)
    return cli
