#!/usr/bin/env bash
set -euo pipefail

EVAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/inspire/qb-ilm/project/exploration-topic/wangqiqi-CZXS25210124/anaconda3/envs/qwen36-vllm/bin/python}"
SCRIPT_ROOT="${SCRIPT_ROOT:-/inspire/hdd/global_user/wangqiqi-CZXS25210124/ScriptPipeline}"
EXPERIMENT_ROOT="$SCRIPT_ROOT/output/full_scene_gate_ab_v1"
INPUT="$EXPERIMENT_ROOT/candidate_gate/R01/drama_evaluator_logic_quality_input/full_60_episodes.txt"
OUTPUT="$EXPERIMENT_ROOT/candidate_gate/R01/drama_evaluations_logic_quality_qwen38_v24/full_60_episodes"
LOG_ROOT="$SCRIPT_ROOT/logs"
ACTION="${1:-run}"

export DRAMA_EVAL_BASE_URL="${DRAMA_EVAL_BASE_URL:-http://127.0.0.1:8001/v1}"
export DRAMA_EVAL_API_KEY="${DRAMA_EVAL_API_KEY:-EMPTY}"
export DRAMA_EVAL_MODEL=Qwen3.8-27B
export DRAMA_EVAL_TEMPERATURE=0
export DRAMA_EVAL_ENABLE_THINKING=false
export PYTHONUNBUFFERED=1

run_candidate() {
  "$PYTHON_BIN" "$EVAL_DIR/evaluate_multi_agent.py" "$INPUT" \
    --output-dir "$OUTPUT" \
    --model Qwen3.8-27B --context-mode direct --direct-char-limit 300000 --chunk-chars 100000 \
    --review-workers 2 --audit-workers 2 --arbitration-workers 2 \
    --max-output-tokens 24576 --timeout 1200 --retries 3 --parse-retries 2
}

case "$ACTION" in
  prepare)
    "$PYTHON_BIN" "$EVAL_DIR/recover_full_scene_gate_r01_v24.py" prepare
    ;;
  status)
    "$PYTHON_BIN" "$EVAL_DIR/recover_full_scene_gate_r01_v24.py" status
    ;;
  run)
    mkdir -p "$LOG_ROOT"
    "$PYTHON_BIN" "$EVAL_DIR/recover_full_scene_gate_r01_v24.py" prepare
    models="$(curl --noproxy '*' --fail --silent --show-error "$DRAMA_EVAL_BASE_URL/models")"
    [[ "$models" == *'Qwen3.8-27B'* ]] || { echo '8001 服务未返回 Qwen3.8-27B' >&2; exit 1; }
    exec 9>"$EXPERIMENT_ROOT/.logic_quality_r01_v24.lock"
    flock -n 9 || { echo 'R01 v2.4 评测或恢复任务已经在运行' >&2; exit 2; }
    set -o pipefail
    "$PYTHON_BIN" "$EVAL_DIR/recover_logic_r03_v24.py" run 2>&1 | tee -a "$LOG_ROOT/full_scene_gate_logic_quality_v24_recovery_candidate.log"
    run_candidate 2>&1 | tee -a "$LOG_ROOT/full_scene_gate_logic_quality_v24_recovery_candidate.log"
    "$PYTHON_BIN" "$EVAL_DIR/recover_full_scene_gate_r01_v24.py" verify
    bash "$EVAL_DIR/run_full_scene_gate_r01_v24.sh" report
    ;;
  *)
    echo '用法：recover_full_scene_gate_r01_v24.sh {prepare|run|status}' >&2
    exit 2
    ;;
esac
