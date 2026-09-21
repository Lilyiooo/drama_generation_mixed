# 关闭桥段库检索

创意生成链路支持一个向后兼容的布尔开关：

```proto
message GenerateInputByCreativity {
  // 原字段 1-6 保持不变。
  bool disable_plot_retrieval = 7;
}
```

- 未传或传 `false`：保持原有行为，查询桥段库并将召回结果注入相关 Prompt。
- 传 `true`：前三集大纲、全量分集大纲和逐集剧本均跳过桥段库查询；没有用户主动传入 `reference` 时，Prompt 中整个“创作参考套路/桥段”段落也会被删除。
- 用户主动传入的 `reference` 不属于桥段库召回内容，因此开关开启后仍会保留。

服务端代码兼容尚未声明字段 7 的旧版 Python stub：新版二进制 protobuf 客户端传来的 unknown field 仍可被识别。正式发布时，应在 `trpc_script_drama_operator` 协议仓库中加入上述字段并重新发布 stub。

本地全流程测试可以直接使用：

```bash
python service/drama_by_creativity/tests/run_test.py \
  --all \
  --plot_type SHORT_CARTOON \
  --episode_nums 60 \
  --disable_plot_retrieval
```

也可以通过环境变量兼容旧 stub：

```bash
DRAMA_DISABLE_PLOT_RETRIEVAL=1 ./run_60ep.sh
```
