#!/usr/bin/env bash
set -euo pipefail

: "${GENERATION_MODEL_PATH:?Set GENERATION_MODEL_PATH to the local Qwen3.6-27B directory}"
VLLM_BIN="${VLLM_BIN:-vllm}"
CACHE_ROOT="${DRAMA_CACHE_ROOT:-$PWD/cache/qwen36}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XDG_CACHE_HOME="$CACHE_ROOT"
export HF_HOME="$CACHE_ROOT/huggingface"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export CUDA_CACHE_PATH="$CACHE_ROOT/cuda"
export FLASHINFER_WORKSPACE_BASE="$CACHE_ROOT/flashinfer"
export TMPDIR="${DRAMA_TMPDIR:-$PWD/tmp/qwen36}"
export VLLM_RPC_BASE_PATH="${VLLM_RPC_BASE_PATH:-/tmp/drama-mixed-vllm-ipc}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
mkdir -p "$CACHE_ROOT" "$TMPDIR" "$VLLM_RPC_BASE_PATH"

exec "$VLLM_BIN" serve "$GENERATION_MODEL_PATH" \
  --served-model-name Qwen3.6-27B \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" \
  --tensor-parallel-size "${TP_SIZE:-8}" \
  --dtype bfloat16 --max-model-len "${MAX_MODEL_LEN:-131072}" \
  --max-num-seqs "${MAX_NUM_SEQS:-4}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --reasoning-parser qwen3 "$@"
