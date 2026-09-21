#!/usr/bin/env bash
# 从随仓库提供的88分大纲生成60集，关闭两个生成后调整环节。
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
GEN_PYTHON="${PYTHON_BIN:-python3}"
export PYTHONUNBUFFERED=1
export DRAMA_LLM_BASE_URL="${DRAMA_LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
export DRAMA_LLM_MODEL="${DRAMA_LLM_MODEL:-qwen-local}"
export DRAMA_LLM_THINKING="${DRAMA_LLM_THINKING:-0}"
export DRAMA_LLM_TIMEOUT="${DRAMA_LLM_TIMEOUT:-0}"
export DRAMA_LLM_MAX_TOKENS="${DRAMA_LLM_MAX_TOKENS:-0}"
export DRAMA_DISABLE_PLOT_RETRIEVAL=1
export DRAMA_SUMMARY_MEMORY_BASELINE=false
export DRAMA_OUTLINE_ONLY=0
export DRAMA_ADJUST_WORD_COUNT=false
export DRAMA_ADJUST_COHERENCE=false
SOURCE_DIR="$SCRIPT_DIR/output/qwen36_27b_eight_fields_demo_20260916_060508_407704"
RUN_ID="outline88_no_adjustments_$(date +%Y%m%d_%H%M%S)"
export DRAMA_OUTPUT_DIR="${DRAMA_OUTPUT_DIR:-$SCRIPT_DIR/output/$RUN_ID}"
if [[ $# -ne 0 ]]; then
  echo '本脚本固定使用88分大纲；通过 DRAMA_OUTPUT_DIR 和 PYTHON_BIN 等环境变量配置。' >&2
  exit 2
fi
"$GEN_PYTHON" tools/check_drama_dependencies.py
mkdir -p "$DRAMA_OUTPUT_DIR"
printf '输入：%s\n输出：%s\n字数调整：false；衔接调整：false\n' "$SOURCE_DIR" "$DRAMA_OUTPUT_DIR"
"$GEN_PYTHON" -u service/drama_by_creativity/tests/run_from_episode_outline.py \
  --source-dir "$SOURCE_DIR" \
  --output-dir "$DRAMA_OUTPUT_DIR" \
  --story-id "$RUN_ID" \
  --plot-type SHORT_CARTOON \
  --start-episode 1 --end-episode 60 \
  2>&1 | tee -a "$DRAMA_OUTPUT_DIR/generation.log"
