# -*- coding=utf-8
import os
import time
import json
import hashlib
import logging
import yaml

from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
from qcloud_cos.cos_exception import CosException
from trpc.log import logger

cos_logger = logging.getLogger("qcloud_cos.cos_client")
cos_logger.setLevel(logging.ERROR)


def hash_to_32bit_str(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    hash_obj = hashlib.sha256()
    hash_obj.update(data)
    hash_val = hash_obj.hexdigest()[:32]
    return hash_val


COS_CLIENT = None


class CosClient(object):
    def __init__(self, sid, skey, region, bucket, endpoint, token=None, scheme="https"):
        self.sid = sid
        self.skey = skey
        self.region = region
        self.bucket = bucket
        self.token = token
        self.scheme = scheme
        self.endpoint = endpoint
        self.cfg = CosConfig(
            Region=region,
            SecretId=sid,
            SecretKey=skey,
            Token=token,
            Scheme=scheme,
            Endpoint=endpoint,
        )
        self.client = CosS3Client(self.cfg)
        self.retry = 3
        self.default_cache_dir = ".cache/cos_cache"
        os.makedirs(self.default_cache_dir, exist_ok=True)

    def upload(self, cos_path, local_path):
        for retry in range(self.retry):
            try:
                self.client.upload_file(
                    Bucket=self.bucket, Key=cos_path, LocalFilePath=local_path
                )
                return True
            except CosException as err:
                logger.warning(
                    f"COS Error: Failed to upload {local_path}, {err}, retry {retry} times."
                )
                time.sleep(0.1)
        logger.error(f"COS Error: Max retries for uploading {local_path}")
        return False

    def download(self, cos_path, local_path):
        if not self.object_exists(cos_path):
            return False

        for retry in range(self.retry):
            try:
                # headers = {"Accept-Encoding": "identity"}  # 明确表示不接受任何压缩
                self.client.download_file(
                    Bucket=self.bucket, Key=cos_path, DestFilePath=local_path
                )
                return True
            except (CosException, KeyError, TypeError, AttributeError) as err:
                logger.warning(
                    f"COS Error: Failed to download {cos_path}, {err}, retry {retry} times."
                )
                time.sleep(0.1)
        logger.error(f"COS Error: Max retries for downloading {cos_path}")
        return False

    def object_exists(self, cos_path):
        return self.client.object_exists(self.bucket, cos_path)

    def list_objects(self, cos_dir, limit=1000):
        # 初始化分页参数
        marker = ""
        is_truncated = True
        files = []
        # 分页列出所有文件
        while is_truncated:
            response = self.client.list_objects(
                Bucket=self.bucket, Prefix=cos_dir, Marker=marker, MaxKeys=1000
            )
            for content in response.get("Contents", []):
                files.append(content["Key"])

            # 更新分页参数
            is_truncated = response["IsTruncated"].lower() == "true"

            if limit > 0 and len(files) >= limit:
                is_truncated = False

            if is_truncated:
                marker = response["NextMarker"]
        if limit > 0:
            files = files[:limit]
        return files

    def get_llm_cache(self, query, model_dir="default"):
        cache_name = hash_to_32bit_str(query)
        cos_path = f"ip_gpt/llm_kv_cache/{model_dir}/{cache_name}.json"
        local_path = f"{self.default_cache_dir}/{cache_name}.json"

        # 对象存在且下载成功
        if self.download(cos_path, local_path):
            with open(local_path) as fp:
                data = json.load(fp)
                return data[query]
        return ""

    def save_llm_cache(self, query, answer, model_dir="default"):
        cache_name = hash_to_32bit_str(query)

        cos_path = f"ip_gpt/llm_kv_cache/{model_dir}/{cache_name}.json"
        local_path = f"{self.default_cache_dir}/{cache_name}.json"

        data = {query: answer}
        with open(local_path, "w") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)

        self.upload(cos_path, local_path)

        return cache_name


def get_cos_client():
    global COS_CLIENT
    if COS_CLIENT is None:
        work_dir = os.path.join(os.path.dirname(__file__), "../../..")
        config_path = os.path.join(work_dir, "trpc_python.yaml")
        with open(config_path) as f:
            cos_params = yaml.safe_load(f)["cos"]
        COS_CLIENT = CosClient(**cos_params)
    return COS_CLIENT


if __name__ == "__main__":
    client = get_cos_client()
    LOCAL_PATH = "a.json"
    REMOTE_PATH = "ip_gpt/test/cos_cleint_test.json"
    client.upload(REMOTE_PATH, LOCAL_PATH)
    client.download(REMOTE_PATH, "down_" + LOCAL_PATH)
