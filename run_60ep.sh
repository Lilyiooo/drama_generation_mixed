#!/usr/bin/env bash
# 生成完整60集剧本（全流程 + 实时落盘）
set -euo pipefail

cd "$(dirname "$0")"

"${PYTHON_BIN:-python3}" tools/check_drama_dependencies.py

# 本地 vLLM；只需 requirements-local.txt 中的公开依赖。
export DRAMA_DISABLE_PLOT_RETRIEVAL=1
export DRAMA_ADJUST_WORD_COUNT=false
export DRAMA_ADJUST_COHERENCE=false
export DRAMA_LLM_BASE_URL="${DRAMA_LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
export DRAMA_LLM_MODEL="${DRAMA_LLM_MODEL:-qwen-local}"
export DRAMA_LLM_THINKING="${DRAMA_LLM_THINKING:-0}"
export DRAMA_LLM_TIMEOUT="${DRAMA_LLM_TIMEOUT:-0}"
export DRAMA_LLM_MAX_TOKENS="${DRAMA_LLM_MAX_TOKENS:-0}"
export PYTHONUNBUFFERED=1

if [[ "${DRAMA_DISABLE_PLOT_RETRIEVAL:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    export DRAMA_OUTPUT_DIR="${DRAMA_OUTPUT_DIR:-$(pwd)/output/60ep_$(date +%m%d_%H%M)_v9_future5_state2_complete_noqd}"
else
    export DRAMA_OUTPUT_DIR="${DRAMA_OUTPUT_DIR:-$(pwd)/output/60ep_$(date +%m%d_%H%M)_v9_future5_state2_complete}"
fi
# 故事总纲生成后细化全部四阶段，单锚点最多4事件、最多4轮局部链。
export DRAMA_EVENTLINE_REFINEMENT_ENABLED=1
export DRAMA_EVENTLINE_STAGES="开端,发展,高潮,结局"
export DRAMA_EVENTLINE_MAX_EVENTS_TO_ANCHOR="${DRAMA_EVENTLINE_MAX_EVENTS_TO_ANCHOR:-4}"
export DRAMA_EVENTLINE_MAX_LOCAL_CHAINS_PER_ANCHOR="${DRAMA_EVENTLINE_MAX_LOCAL_CHAINS_PER_ANCHOR:-4}"
mkdir -p "$DRAMA_OUTPUT_DIR"

echo "=========================================="
echo "  创意到完整剧本生成"
echo "  vLLM 接口: $DRAMA_LLM_BASE_URL"
echo "  模型: $DRAMA_LLM_MODEL"
echo "  输出目录: $DRAMA_OUTPUT_DIR"
echo "=========================================="

"${PYTHON_BIN:-python3}" service/drama_by_creativity/tests/run_test.py \
    --all \
    --plot_type SHORT_CARTOON \
    --episode_nums 60 \
    "$@"

echo ""
echo "=========================================="
echo "  完成！输出目录: $DRAMA_OUTPUT_DIR"
echo "=========================================="
ls -R "$DRAMA_OUTPUT_DIR"
