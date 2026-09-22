
# 剧本评测器

本目录收录 `Tencent-drama/drama_evaluator` 的评测代码、评分提示词和方法文档。完整方法实验使用本评测体系，以本地 `Qwen3.8-27B` 作为固定评测模型；剧本生成与状态抽取使用另一个模型，二者不在这里混用。

## 命令行使用

在仓库根目录运行；先安装 `drama_evaluator/requirements.txt`，并启动兼容 OpenAI Chat Completions 的评测服务。

```bash
export DRAMA_EVAL_BASE_URL=http://127.0.0.1:8001/v1
export DRAMA_EVAL_API_KEY=EMPTY
python drama_evaluator/evaluate_multi_agent.py /path/to/script.txt \
  --output-dir /path/to/evaluation_output \
  --model Qwen3.8-27B \
  --context-mode direct \
  --direct-char-limit 60000 \
  --chunk-chars 24000 \
  --max-output-tokens 16000
```

`evaluate_multi_agent.py` 是多评审、审核、按需仲裁的入口；评分细则保存在 `prompts/`，详细设计见 `PIPELINE.md`。真实实验还会由 `local_pipeline/drama_recover.py` 管理导出、模型配置和异常恢复。

当前剧本质量标准采用论文研究口径：人物质量、戏剧吸引力、情节规划质量、分集剧本质量和设定质量。五个模块各自以 90 分为台账基线，模块内确认问题和亮点直接按照 S/A/B 锚点加扣；主角、人物关系、台词、视觉化等底层单元只承担证据分类功能，不再设置内部占比。情节规划提示词使用 1–5 行为量表辅助定级，程序继续采用 S/A/B 台账分与独立整体评分 50/50 融合。流程产出评分、证据台账与行为诊断。

## Full-scene-gate 配对评测复现

本目录保留论文实验中比较 `control_hybrid` 与 `candidate_gate` 的实际评测入口、冻结协议、异常恢复和汇总脚本。脚本从 `$SCRIPT_ROOT/output/full_scene_gate_ab_v1` 导出每组 60 集完整剧本，冻结评测器与 Prompt 哈希，运行多 Agent 评测并计算 candidate-control 配对差值。

在仓库根目录设置运行环境：

```bash
export SCRIPT_ROOT=/path/to/ScriptPipeline
export PYTHON_BIN=/path/to/python
export DRAMA_EVAL_BASE_URL=http://127.0.0.1:8001/v1
export DRAMA_EVAL_API_KEY=EMPTY
```

当前 v2.4 入口：

```bash
EVAL_DIR=evaluation/drama_evaluator_logic_quality

# R01：准备输入、冻结协议、运行两组评测并生成配对报告
"$EVAL_DIR/run_full_scene_gate_r01_v24.sh" run

# R02/R03：逐个 run 并行运行两组评测，最后汇总三次运行
"$EVAL_DIR/run_full_scene_gate_r02_r03_v24.sh" run

# 只查询状态或重新汇总，不调用模型
"$EVAL_DIR/run_full_scene_gate_r01_v24.sh" status
"$EVAL_DIR/run_full_scene_gate_r02_r03_v24.sh" report
```

`run_full_scene_gate_r01_v24.sh` 保留实际 R01 协议中的 `16000` 输出上限。该实验的 candidate 逻辑 R03 曾发生输出截断；`recover_full_scene_gate_r01_v24.sh`、`recover_full_scene_gate_r01_v24.py` 和 `recover_logic_r03_v24.py` 保存了只恢复缺失评审、保护既有评审哈希并重建下游结果的流程。R02/R03 使用 `24576` 输出上限和 3 次结构解析重试。

`summarize_full_scene_gate_r01_v24.py` 冻结并验证 R01 v2.4 协议；`summarize_full_scene_gate_all_runs_v24.py` 冻结并验证 R02/R03 协议，并汇总 R01–R03 的配对均值与样本标准差。`run_full_scene_gate_r01.sh` 与 `summarize_full_scene_gate_r01.py` 是 v2.4 之前的历史 R01 入口，仅用于核对旧结果。

运行产物保存在 `$SCRIPT_ROOT/output/full_scene_gate_ab_v1`，日志保存在 `$SCRIPT_ROOT/logs`，均不随代码仓库上传。

`badcase_scripts/` 中的外部完整剧本没有上传；它们并非运行评测器所必需。运行结果、Web 上传内容、`.env` 和任何 API 密钥也未上传。`run_badcase_*.sh` 是历史测试脚本，若要复现对应测试，需自行提供有使用权限的测试剧本。
