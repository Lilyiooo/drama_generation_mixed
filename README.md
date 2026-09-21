# Drama Generation Mixed

面向 60 集长篇短剧生成的混合框架：在 ScriptPipeline 原生“创意 → 世界观/人设 → 总纲 → 分集大纲 → 场次大纲 → 剧本”流程中，加入**单状态生命周期记忆、Hybrid 状态检索和保守的因子化场次状态门控**。

本仓库对应实验中的当前方法：

> **ScriptPipeline + Native Narrative Memory + State Lifecycle + Hybrid Retrieval + Factorized Scene Gate**

生成、状态抽取、检查和必要的一次修订使用 `Qwen3.6-27B`；最终逻辑/质量评测使用独立的 `Qwen3.8-27B`。评测模型不参与剧本生成。

## 方法概览

```text
创意
  └─ 世界观 / 人设 / 故事总纲 / 60 集分集大纲
       └─ 第 N 集场次大纲
            ├─ ScriptPipeline 原生叙事记忆
            ├─ 九字段状态生命周期
            │    └─ Hybrid 检索（相关性 + 近期性 + 字段优先级）
            └─ 因子化场次状态门控
                 ├─ 无硬冲突：保留原场次
                 ├─ 证据不足：记录不确定，保留原场次
                 └─ 已证实硬冲突：最多修订一次 → 复查 → 失败则回退原场次
                      └─ 完整剧本
                           └─ 原生记忆更新 + 状态生命周期更新
```

状态生命周期维护以下九类信息：

- `character_state`
- `relationship_state`
- `known_information`
- `unknown_information`
- `confirmed_facts`
- `unresolved_threads`
- `resources_and_evidence`
- `current_goals`
- `timeline`

当前方法**不使用**义务记忆、Strategy 经验卡或经验卡选择器。

## 实验分组

| 分组 | ScriptPipeline 原生流程 | 原生叙事记忆 | 状态生命周期 | Hybrid 检索 | 因子化场次门控 |
|---|---:|---:|---:|---:|---:|
| `candidate_gate` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `control_hybrid` | ✓ | ✓ | ✓ | ✓ | — |

配套对比中的 direct baseline 是“本集大纲 + 此前全部剧集原文 → 单次直接写作”，不走本仓库的场次 Agent、状态记忆、检索和门控。

## 目录结构

```text
service/drama_by_creativity/
  generate_episode_script.py       # 分集剧本主生成链
  state_lifecycle_memory.py        # 九字段状态、哈希链、Hybrid 检索接入
  scene_state_gate.py              # 场次/状态一致性检查与缓存
  scene_gate_decision.py           # 因子化证据分类与确定性汇总
  conservative_scene_gate.py       # 一次修订、复查、语义回退策略
tools/
  state_hybrid_study.py             # 状态生命周期实验和恢复
  full_scene_gate_study.py          # Candidate/Control 完整 60 集 A/B 入口
  evaluate_with_drama_evaluator.py  # v2.4 评测适配器
vendor/tencent_drama/
  local_pipeline/gated_engine.py    # NMF 状态合并与校验逻辑
  local_pipeline/state_retrieval.py # Full/Recent/BM25/Hybrid 检索
  nmf_v2_7_package/                 # 状态引擎依赖
evaluation/drama_evaluator_logic_quality/
                                      # Qwen3.8 多评审/审核/按需仲裁评测器
scripts/
  serve_generation.sh               # Qwen3.6 vLLM 服务
  serve_evaluator.sh                # Qwen3.8 vLLM 服务
results/
  baseline_comparison_summary_v24.md
```

## 环境

推荐：

- Linux
- Python 3.12
- 8 × NVIDIA H100 80GB（复现实验配置；其他 GPU 需相应调整 TP、上下文和并发）
- CUDA 12.8
- vLLM 0.19.0
- PyTorch 2.10.0 + CUDA 12.8
- OpenAI-compatible Chat Completions API

安装生成和评测所需的公开依赖：

```bash
conda create -n drama-mixed python=3.12 pip -y
conda activate drama-mixed
python -m pip install -r requirements-local.txt
python -m pip install -r vendor/tencent_drama/nmf_v2_7_package/requirements.txt
python -m pip install -r evaluation/drama_evaluator_logic_quality/requirements.txt
```

另行安装与你的 CUDA/驱动匹配的 PyTorch 和 vLLM。本仓库不包含模型权重。

## 配置故事输入

默认的完整创意入口位于 `service/drama_by_creativity/tests/run_test.py`。修改其中：

- `TEST_CORE_STORY`：核心创意
- `TEST_TOPIC`：题材
- `TEST_ROLE_SETTING`：初始人物设定
- `TEST_REFERENCE`：创作参考，可为空

默认实验使用 1 季、60 集、每集目标 1000–1400 中文字符。

## 启动生成模型

设置本地 Qwen3.6 模型目录并使用 8 卡、131072 上下文启动：

```bash
export GENERATION_MODEL_PATH=/path/to/Qwen3.6-27B
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TP_SIZE=8 MAX_MODEL_LEN=131072 MAX_NUM_SEQS=4 \
GPU_MEMORY_UTILIZATION=0.90 \
bash scripts/serve_generation.sh
```

服务默认位于 `http://127.0.0.1:8000/v1`，served model name 为 `Qwen3.6-27B`。检查：

```bash
curl --noproxy '*' --fail http://127.0.0.1:8000/v1/models
```

## 从零生成冻结源资产

完整 Candidate/Control A/B 使用同一份已经生成并校验的故事资产作为共同起点。先生成世界观、人设、总纲、60 集大纲、Future Map、原生记忆和状态初始点：

```bash
export PYTHON_BIN="$(command -v python)"
export DRAMA_LLM_BASE_URL=http://127.0.0.1:8000/v1
export DRAMA_LLM_MODEL=Qwen3.6-27B
export DRAMA_LLM_THINKING=0
export DRAMA_LLM_TIMEOUT=1200
export DRAMA_STATE_LIFECYCLE_HYBRID=1
export DRAMA_OUTPUT_DIR="$PWD/output/full_pipeline_state_hybrid_v2"

bash run_60ep.sh --story_id APID-test-001
```

该目录随后只作为冻结资产来源。A/B 的每条轨迹都会从第 1 集重新写剧本，并独立演化原生叙事记忆和生命周期状态；不会复制源剧本或源动态记忆。

## 运行当前方法 A/B

离线准备实验目录，不调用模型：

```bash
bash full_scene_gate_study.sh prepare \
  --source "$PWD/output/full_pipeline_state_hybrid_v2" \
  --source-story-id APID-test-001
```

先进行 R01 每组两集的冒烟测试，再续跑完整 R01：

```bash
bash full_scene_gate_study.sh generate \
  --run R01 --limit 2 --workers 2 --execute-api

bash full_scene_gate_study.sh generate \
  --run R01 --workers 2 --execute-api
```

扩展为 R01、R02、R03 三次运行：

```bash
bash full_scene_gate_study.sh generate \
  --all-runs --workers 3 --execute-api
```

查看状态（只读，不调用 API）：

```bash
bash full_scene_gate_study.sh status --all-runs
```

同一条轨迹内部严格按集顺序生成；不同 arm/run 可以并发。失败后重跑相同命令即可恢复，已经绑定正文、原生记忆、状态哈希链和门控结果的集不会重做。

## 启动评测模型

生成完成后停止 Qwen3.6 并释放 GPU，再启动 Qwen3.8：

```bash
export EVALUATION_MODEL_PATH=/path/to/Qwen3.8-27B
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TP_SIZE=8 MAX_MODEL_LEN=131072 MAX_NUM_SEQS=4 \
GPU_MEMORY_UTILIZATION=0.90 \
bash scripts/serve_evaluator.sh
```

服务默认位于 `http://127.0.0.1:8001/v1`，served model name 为 `Qwen3.8-27B`。

## 运行 v2.4 逻辑/质量评测

```bash
export DRAMA_EVAL_BASE_URL=http://127.0.0.1:8001/v1
export DRAMA_EVAL_API_KEY=EMPTY

bash full_scene_gate_study.sh evaluate \
  --all-runs --workers 3 --execute-api

bash full_scene_gate_study.sh report --all-runs
```

评测器采用剧本逻辑、剧本质量各 3 个独立评审、各维度审核与按需仲裁；台账分和独立整体分 50/50 融合。评测温度为 0，thinking 关闭。完整实现位于 `evaluation/drama_evaluator_logic_quality/`。

## 关键环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DRAMA_LLM_BASE_URL` | `http://127.0.0.1:8000/v1` | 生成模型 API |
| `DRAMA_LLM_MODEL` | 由入口设置 | 生成模型名 |
| `DRAMA_STATE_LIFECYCLE_HYBRID` | `0` | 是否启用状态生命周期与 Hybrid 检索 |
| `DRAMA_TENCENT_ROOT` | `vendor/tencent_drama` | 可选的外部状态引擎根目录覆盖 |
| `DRAMA_OUTPUT_DIR` | 入口决定 | 生成结果目录 |
| `DRAMA_EVAL_BASE_URL` | `http://127.0.0.1:8001/v1` | 评测模型 API |
| `DRAMA_EVAL_API_KEY` | `EMPTY` | 本地兼容 API key |
| `PYTHON_BIN` | `python3` | Shell 入口使用的 Python |

## 输出与审计

每个 arm/run 主要包含：

```text
05_drama/                                  # 逐集正文
narrative_memory/                          # ScriptPipeline 原生叙事记忆
state_lifecycle/<story-id>/initial_state.json
state_lifecycle/<story-id>/updates/E*.json # 九字段状态增量和哈希链
state_lifecycle/<story-id>/retrieval/E*.json
scene_state_gate/<story-id>/E*/            # 检查、修订、复查、回退和请求缓存
drama_evaluations_logic_quality_qwen38_v24/
```

实验会冻结输入、代码哈希、arm/run 绑定和脚本摘要。若资产或实现变化，程序拒绝把新结果混入已有实验目录。

## 已完成结果

同一故事、每种方法 3 次、每次 60 集：

| 方法 | 剧本逻辑（均值 ± 样本标准差） | 剧本质量（均值 ± 样本标准差） |
|---|---:|---:|
| Direct outline + all previous scripts | 79.57 ± 3.04 | 80.87 ± 3.37 |
| ScriptPipeline + Lifecycle-Hybrid | 83.02 ± 1.83 | **86.53 ± 1.89** |
| ScriptPipeline + Lifecycle-Hybrid + Factorized Scene Gate | **85.22 ± 0.38** | 85.47 ± 1.12 |

Candidate 相对 direct baseline：逻辑平均 `+5.64`、质量平均 `+4.60`，两个指标均为 3/3 配对取胜；角色状态提升 `+10.17`，重复控制提升 `+8.67`。详细表格和边界说明见 `results/baseline_comparison_summary_v24.md`。

## 测试

快速确定性测试与语法检查：

```bash
python -m unittest \
  service.drama_by_creativity.tests.test_scene_gate_decision \
  service.drama_by_creativity.tests.test_full_scene_gate_study.ConservativeGateTests

python -m compileall -q service drama_local tools evaluation vendor
bash -n full_scene_gate_study.sh scripts/serve_generation.sh scripts/serve_evaluator.sh
```

`service/drama_by_creativity/tests/` 还包含需要构造完整流水线、缓存和恢复状态的集成测试；它们的执行时间明显长于上述快速测试。

## 上游与归属说明

本仓库是在以下工程基础上的研究集成：

- ScriptPipeline：https://github.com/rimory6/ScriptPipeline
- drama_evaluator_logic_quality：https://github.com/rimory6/drama_evaluator_logic_quality

仓库保留上游目录结构，并将本实验所需的状态引擎、检索器和评测器一并放入仓库，以避免运行时依赖本机绝对路径。模型权重、运行输出、缓存、日志、私有数据和 API 密钥均不在版本库中。
