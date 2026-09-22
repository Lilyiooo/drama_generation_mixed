#!/usr/bin/env bash
set -euo pipefail

EVAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/inspire/qb-ilm/project/exploration-topic/wangqiqi-CZXS25210124/anaconda3/envs/qwen36-vllm/bin/python}"
SCRIPT_ROOT="${SCRIPT_ROOT:-/inspire/hdd/global_user/wangqiqi-CZXS25210124/ScriptPipeline}"
EXPERIMENT_ROOT="$SCRIPT_ROOT/output/full_scene_gate_ab_v1"
LOG_ROOT="$SCRIPT_ROOT/logs"
ACTION="${1:-run}"

export DRAMA_EVAL_BASE_URL="${DRAMA_EVAL_BASE_URL:-http://127.0.0.1:8001/v1}"
export DRAMA_EVAL_API_KEY="${DRAMA_EVAL_API_KEY:-EMPTY}"
export DRAMA_EVAL_MODEL=Qwen3.8-27B
export DRAMA_EVAL_TEMPERATURE=0
export DRAMA_EVAL_ENABLE_THINKING=false
export PYTHONUNBUFFERED=1

arms=(control_hybrid candidate_gate)
runs=(R02 R03)

prepare_inputs() {
  mkdir -p "$LOG_ROOT"
  for run in "${runs[@]}"; do
    for arm in "${arms[@]}"; do
      "$PYTHON_BIN" "$EVAL_DIR/tools/export_pipeline_script.py" \
        "$EXPERIMENT_ROOT/$arm/$run" \
        "$EXPERIMENT_ROOT/$arm/$run/drama_evaluator_logic_quality_input/full_60_episodes.txt"
    done
  done
  "$PYTHON_BIN" "$EVAL_DIR/summarize_full_scene_gate_all_runs_v24.py" --prepare-protocol
}

run_one() {
  local arm="$1" run="$2"
  "$PYTHON_BIN" "$EVAL_DIR/evaluate_multi_agent.py" \
    "$EXPERIMENT_ROOT/$arm/$run/drama_evaluator_logic_quality_input/full_60_episodes.txt" \
    --output-dir "$EXPERIMENT_ROOT/$arm/$run/drama_evaluations_logic_quality_qwen38_v24/full_60_episodes" \
    --model Qwen3.8-27B --context-mode direct --direct-char-limit 300000 --chunk-chars 100000 \
    --review-workers 2 --audit-workers 2 --arbitration-workers 2 \
    --max-output-tokens 24576 --timeout 1200 --retries 3 --parse-retries 3
}

case "$ACTION" in
  prepare)
    prepare_inputs
    ;;
  status)
    "$PYTHON_BIN" "$EVAL_DIR/summarize_full_scene_gate_all_runs_v24.py"
    ;;
  report)
    "$PYTHON_BIN" "$EVAL_DIR/summarize_full_scene_gate_all_runs_v24.py" --write
    ;;
  run)
    prepare_inputs
    models="$(curl --noproxy '*' --fail --silent --show-error "$DRAMA_EVAL_BASE_URL/models")"
    [[ "$models" == *'Qwen3.8-27B'* ]] || { echo '8001 服务未返回 Qwen3.8-27B' >&2; exit 1; }
    exec 9>"$EXPERIMENT_ROOT/.logic_quality_r02_r03_v24.lock"
    flock -n 9 || { echo 'R02/R03 v2.4 评测已经在运行' >&2; exit 2; }
    failed=0
    for run in "${runs[@]}"; do
      pids=()
      for arm in "${arms[@]}"; do
        (set -o pipefail; run_one "$arm" "$run" 2>&1 | tee -a "$LOG_ROOT/full_scene_gate_logic_quality_v24_${arm}_${run}.log") &
        pids+=("$!")
        echo "$arm/$run PID=$!"
      done
      for index in "${!pids[@]}"; do
        if wait "${pids[index]}"; then code=0; else code=$?; failed=1; fi
        echo "${arms[index]}/$run exit_code=$code"
      done
    done
    "$PYTHON_BIN" "$EVAL_DIR/summarize_full_scene_gate_all_runs_v24.py"
    [[ "$failed" -eq 0 ]] || { echo '部分评测未完成；成功结果已保留，重跑同一命令续跑。' >&2; exit 1; }
    "$PYTHON_BIN" "$EVAL_DIR/summarize_full_scene_gate_all_runs_v24.py" --write
    ;;
  *)
    echo '用法：run_full_scene_gate_r02_r03_v24.sh {prepare|run|status|report}' >&2
    exit 2
    ;;
esac
