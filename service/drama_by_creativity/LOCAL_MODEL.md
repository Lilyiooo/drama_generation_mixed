# 本地 60 集创作流程

`run_60ep_qwen36_27b.sh` 对应的创作流程已独立于腾讯服务：

- 本地数据类取代内部协议包，Python 方法取代 RPC 服务接口。
- Python logging 取代 tRPC 日志。
- 本地 JSON 取代 COS，文件按输出目录和故事 ID 隔离，原子写入。
- 桥段检索关闭，不导入桥段库客户端。
- 所有模型调用（包括 v9 事件线细化）连接本地 vLLM。

原有提示词文件、生成顺序、字数调整、Future Map、叙事记忆和事件线算法保留。
这不保证 Qwen 和 Gemini 的文本质量相同。其他小说改编服务和历史评测入口
仍在仓库中，但本入口不导入它们的腾讯 SDK，也不需要安装原 requirements.txt。

## 安装和运行

Python 3.10 或更高，在项目根目录：

```bash
python3 -m pip install -r requirements-local.txt
bash run_60ep_qwen36_27b.sh
```

默认输出：
`output/60ep_0906_1302_v9_future5_state2_complete_noqd_qwen36_27b`

默认接口 `http://127.0.0.1:8000/v1/chat/completions`，服务名称 `qwen-local`。
该名称是 vLLM 的 served-model-name，请确保服务实际加载了目标 Qwen 模型。
跨节点运行时：

```bash
DRAMA_LLM_BASE_URL=http://GPU节点IP:8000/v1 bash run_60ep_qwen36_27b.sh
```

可选环境变量：

- `PYTHON_BIN`：安装了 requirements-local.txt 的 Python，默认 python3。
- `DRAMA_LLM_MODEL`：模型服务名称，所有生成阶段统一使用。
- `DRAMA_LLM_THINKING`：默认 0，关闭思考模式；设为 1 开启。
- `DRAMA_LLM_TIMEOUT`：读取超时秒数，0 表示不限时；连接超时 5 秒。
- `DRAMA_LLM_MAX_TOKENS`：Qwen 启动脚本默认 0，省略输出上限，与推理示例一致。
  直接调用 Python 且不设置时，保留各阶段 model_configs.yaml 的 max_new_tokens。
- `DRAMA_LLM_API_KEY`：仅服务启用认证时设置。

## 本地数据

- `01_script_proposal`、`02_demo_drama`、`03_story_outline`、`04_episode_outline`、
  `04_future_map`、`05_drama`：原有可查看的中间结果和剧本。
- `local_store/<story_id>/script_content/episode_0.json`：内部读写使用的第 1 集，
  索引从 0 开始，内容保留场次分隔符。用于前文连续性及从前三集生成总纲。
- `narrative_memory/<story_id>`：原有叙事记忆、世界状态。
- `pipeline_result.json`：全流程成功后的结构化结果。

不同输出目录不会共用内部剧本文件。重新运行相同输出目录仍会更新该目录；
如要另开一次实验，用 `DRAMA_OUTPUT_DIR=新目录 bash run_60ep.sh`。
单独续跑需要相应的前置文件；不会再去 COS 获取历史内容。

## 验证和备份

```bash
python3 tools/check_drama_dependencies.py
python3 service/drama_by_creativity/tests/run_test.py --help
python3 -m pytest service/drama_by_creativity/tests tests_local -q
```

测试依赖 pytest，可单独安装。tests_local 包含数据复制、JSON 往返、存储隔离、
提示词哈希校验和完整 60 集输出回放测试。回放使用历史模型输出，不评估 Qwen
的生成质量；事件线通过单独的测试验证。真实推理需要可访问的 vLLM 服务。

本地化前源码备份在 `localization_backup/before_local_runtime.tar.gz`。
历史生成目录未修改。

## 中间输出与失败诊断

只要设置了 `DRAMA_OUTPUT_DIR`，所有实际模型请求都会记录到：

```text
diagnostics/<本次进程时间与唯一ID>/
  model_calls/<唯一调用ID>/
    record.json          # 接口、模型参数、输入、请求上下文
    prompt.txt           # 实际发送的提示词
    http_response.txt    # 原始 HTTP 返回体；错误状态或非法 JSON 也保留
    output.txt           # 模型正文；空正文、截断正文也保留
    status.json          # 调用耗时与错误
  json_attempts/...      # 每轮原始生成及每次 JSON 修复的完整文本
  json_validation/...    # 解析结果、失败原因、对应输出的路径
  json_retry/...         # 重新生成/最终失败记录
  eventline_json_retry/... # 事件线 JSON 失败与重新生成原因
  validation_events/...  # 大纲校验、润色、字数检查等警告和错误全文
  skipped_episodes/...   # 集号、跳过原因、最后原始输出及关联路径
```

每个事件目录都有 `record.json`；有输出的事件还包含 `output.txt`。
请求没有收到返回时，不会伪造模型文本：通过 `status.json` 记录连接或读取错误。
`skipped_episodes` 中还会保留已生成但被弃用的部分正文（如逐场生成中途失败）。
目录按进程和调用隔离，多次运行、并行调用不会覆盖先前诊断。

诊断保存每次调用的原文和失败原因。之前已经丢失的失败输出不能追溯恢复。
场次大纲现使用下述重试策略，其他 JSON 阶段仍使用原有策略。


## 场次大纲的重试与纠错顺序

1. 原始生成最多 3 次（包含首次），每次使用相同原始提示词和任务参数。
2. 每次检查 JSON 语法、非空列表、全部必需字段及类型、本集场次编号。
   成功立即返回；不合格则进行下一次原始生成。
3. 3 次都不合格时，才调用专用 JSON 纠错模型，最多 3 次。
   每次都以最后一份收到的原始输出为基础，附上最新校验错误和完整合法 JSON 模板。
4. 纠错结果逐次校验，包括最后一次；还检查原文可识别的场次编号、数量和顺序是否保留。
   这些检查能发现丢场次，但不等于自动证明全文语义完全未改。
5. 最终仍不合格时记录失败和跳集原因，保留全部生成及纠错文本。

原始场次生成提示词未改；专用纠错提示词位于
`service/drama_by_creativity/prompts/scene_repair.py`。
世界观、角色、大纲、Future Map、世界状态等其他阶段不采用这个新顺序。

## 梗概记忆 baseline（默认关闭）

接口字段：`GenerateInputByCreativity.summary_memory_baseline: bool = False`。
不设置时使用原来的 Future Map + Future Driving + World State 方法。
设为 true 时，只对剧本阶段的记忆方法进行替换：

- 跳过完整分集大纲后的 Future Map 和分集贡献映射生成。
- 不创建或读取 NarrativeMemory，不生成 Future Driving，不更新世界状态。
- 每集完成润色、字数调整、正文保存后，额外调用模型生成本集梗概。
- 后续每集按集号顺序传入全部前面集的梗概，不检索、不只截取前五集、不混入未来集梗概。
- 梗概作为原有记忆变量的内容，baseline 的提示词仅调整该段标题、移除前五集摘要段。
- 前一集正文、既有大纲、人物世界观、场次重试、润色、字数调整等其他流程保留。
- 梗概存放于 `summary_memory/<story_id>/memory_state.json` 及逐集 JSON，支持从相同目录恢复。
  重写较早一集会使汇总文件中更晚的旧梗概失效；历史逐集文件不自动删除，但不再被读取。
  梗概缺失或生成失败会中止 baseline 后续生成，避免在缺少记忆的情况下继续。

CLI：`run_test.py` 和 `run_from_episode_outline.py` 均支持 `--summary-memory-baseline`。
也可设 `DRAMA_SUMMARY_MEMORY_BASELINE=true`；该环境开关为整次进程启用 baseline。

从此前 Qwen 大纲续跑，并在结束后执行 continuous_load.sh：

```bash
DRAMA_SUMMARY_MEMORY_BASELINE=true bash run_from_outline_then_occupy.sh
```

不设置该环境变量（或设为 false）且接口字段为 false 时，仍运行原模式。
两种模式请使用不同输出目录进行对比。已有启动脚本默认创建新的时间戳目录。
本选项不会关闭总纲阶段的 v9 事件线细化，也不会修改原始提示词文件。


### 选择故事大纲的细化阶段

默认只细化“发展、高潮”。全流程入口支持按传入顺序选择阶段：

```bash
DRAMA_OUTPUT_DIR="$PWD/output/qwen36_27b_all_stages_$(date +%Y%m%d_%H%M%S)" \
bash run_60ep_qwen36_27b.sh --eventline-stages 开端 发展 高潮 结局
```

也可传 `--eventline-stages "开端,发展,高潮,结局"`，或设置环境变量
`DRAMA_EVENTLINE_STAGES="开端,发展,高潮,结局"`。显式命令行参数优先，并开启细化。
只选一部分时如 `--eventline-stages 发展 结局`。阶段名需与故事大纲的 framework 一致；
重复、空列表或不存在的阶段会报错，不会静默跳过。

各阶段依次接收更新后的整份大纲和累计 trope bank。产物在
`03_story_outline/eventline_refinement_v9/`，包括每阶段的 before/after 大纲、
`v9_result_*`、`trope_bank_after_*` 和最终 manifest。
从已有 episode outline 或已细化大纲续跑会跳过本步骤；要应用阶段选择，需运行故事大纲生成步骤。


### 分集大纲：先分配阶段集数，再分阶段展开

新生成分集大纲默认先调用一次模型，输入详细故事大纲、总集数、题材和前三集参考，
按内容复杂度与整体叙事节奏分配各阶段的连续集数范围，再按大纲阶段顺序逐阶段生成。
此规划独立于事件线细化阶段选择：即使只细化发展、高潮，分集大纲仍覆盖大纲中的全部阶段。

每次阶段展开保留原有题材、世界观、核心故事和前三集参考（默认分支还含人物设定），
详细事件仅传当前阶段 event_list；另传全剧阶段名称/范围、当前阶段分配理由和此前所有已生成的集大纲。
前三集正文只是参考，本次第1—3集大纲也重新生成，避免样稿被误认为已包含在结果里。

产物：
- `04_stage_plan/stage_plan.json`：总集数、每阶段起止集号、节奏理由。
- `04_stage_plan/attempts/`：规划及阶段结构校验结果，包含失败原因与当次解析结果；原始请求/输出继续由 diagnostics 保存。
- `04_episode_outline/chunk_01_epXX-YY.json` 等：每阶段的分集大纲，沿用原有数组结构，原断点续跑入口可读取。
- `04_stage_plan/completion.json`：全剧范围最终校验通过后写入。

范围不连续、重叠、缺阶段/缺集等结构问题会带原因重试，连续3次失败即停止，
不会以最后一次错误结果继续生成正文。结构校验不能代替叙事内容审读。
重新生成时使用新的输出目录，避免混入旧批次文件。从已有详细大纲的入口启动也会自动执行阶段规划；
从已有 episode outline 续写正文则不重新规划。

### v9 局部链独立评审（2026-09-14）

局部树生成候选链后，每条链独立调用一次评委，所有链通过线程池并行评分（默认并发上限32，由 `parallel_workers` 或业务入口环境变量 `DRAMA_EVENTLINE_PARALLEL_WORKERS` 控制），不再每16条合成一个请求。请求包含完整已确定前序事件、该候选链、故事基本信息、当前目标、当前阶段后续锚点和 trope bank。

提示词仅给出扣除当前局部链后到达目标锚点的剩余事件预算；内部计算和落盘仍保留扣减前后数据，跨轮次不会重置。评委要独立判断是否真正达成目标，不可仅相信生成者自报进度，也不可将强制补锚点当成预算内的合理推进。

三个维度均0—100分，满分100分：`coherence`（与历史及链内叙事一致性）40%、`novelty`（因果机制新颖性）30%、`budget_fit`（预算内抵达目标的可行性）30%。总分没有额外到达奖励；同分依次比较一致性、预算可行性、新颖性，再保留原候选顺序。每次评分要求提供 `consistency_review`、`novelty_review`、`budget_review` 与 `reason`，明确引用事件并核对人物生死、身份、记忆、道具、地点时间和前序已完成事件。缺失评分或依据会触发已有 schema 重试，不能用默认高分替代。

v9结果文件每轮的 `terminal_chain_score_records` 保存逐链分数、审查说明、预算和原始回复；`terminal_chain_scores` 保存用于比较的各链总分，`selected_local_chain` / `round_committed` 保存实际选中及提交内容。原始提示词和模型输出继续由 diagnostics 记录。

此次仅调整局部链评审及排序。单步候选生成/评分、树扩展、trope提取、已有强制补锚点分支保持原逻辑；新的评分不等于另加全剧终审。旧运行产物不变，下次详细大纲细化自动使用新评审方式。

百分制评审强调：一致性检查不限于列举事项；novelty重点对比bank中具体因果机制的相似性；预算分依据实际目标完成度、缺失步骤与剩余额度连续给分，零预算未达成不机械给最低分。

### 全剧分集大纲检查与优化

各阶段分集大纲生成并通过结构校验后，默认调用一次全剧审校模型（沿用分集大纲模型配置），再进入未来需求规划与正文生成。原模式和概要记忆模式都会执行此步骤。

输入包括题材、世界观、核心故事、人物设定、创作参考、总集数和全部逐集大纲。模型检查情节矛盾、人物与世界状态连续性、跨集衔接及铺垫与结果，返回检查结论和需要替换的完整集条目；每项包含集号、关联集号与修改理由。无需修改时返回空替换列表。

程序先校验全部替换项的集号、字段与结构，再统一更新各阶段 chunk 文件；未指定的集保持原内容。结构不合格会带反馈重试，连续3轮失败则停止，不进入后续生成。此校验保证结构和替换范围有效，叙事修正质量仍取决于模型判断。

`04_episode_outline_review/before.json` 保存审校前全剧大纲，`review.json` 保存结论、理由与替换项，`after.json` 保存审校后全剧大纲；原始模型请求和回复继续保存在 diagnostics 中，结构校验记录位于 `04_stage_plan/attempts/episode_outline_review_*.json`。完成标记在修订后的阶段文件全部写回后生成。

从详细故事大纲接续生成会执行审校；从已有逐集大纲直接续写正文则直接读取已有条目，不补跑审校。已有运行产物不会自动修改。


### Simple 版逐集大纲三轮审校

`_simple` 在各阶段集大纲全部生成后，连续执行三轮“全剧评审 → 全部集大纲重写”。
每轮输入上一轮的完整结果（首轮为阶段生成结果），以及题材、世界观、核心故事、人物设定等少量背景；不传详细故事大纲、阶段规划或前三集样稿。
沿用独立 review/rewrite demo 的详细评审和完整修订提示词，允许必要的大范围情节修改，并处理未列在评审意见中的问题；修订允许调整原阶段边界。
即使评审问题列表为空，也执行该轮完整重写，固定完成三轮。每次评审同时按统一标准给当前版本打 0—100 总分；第三轮重写后使用同一评审提示词补充最终评分。正常七次逻辑调用（四次评审评分、三次完整重写），JSON 解析和结构失败重试另计。
每轮在 `04_episode_outline_review/round_01` 至 `round_03` 保存 `before.json`、`review_model_result.json`、`review.json`、`after.json`。
根目录 `before.json` 为最初版本；`after.json` 为四个候选中分数最高的版本，`review.json` 为该版本对应的评审。`selection.json` 保存四版分数、路径和最终选择，分数相同优先较早版本；`final_review.json` 保存第三轮重写后的评审评分，`status.json` 记录进度或失败原因。
模型原始请求和响应沿用 diagnostics 落盘；结构重试记录使用独立轮次名称保存在 `04_stage_plan/attempts/`。
任何一轮失败立即停止，保留已完成轮次，不用缺集结果继续生成正文。三轮重写和最后评分全部成功后才返回最高分大纲并更新正式分集文件。评审只校验 JSON 对象与有效的 0—100 数值 total_score，不再限制问题条目、证据、关联集号或建议结构，评审内容完整传给重写模型。完整大纲仍校验全部集数、连续集号和必需字段。
