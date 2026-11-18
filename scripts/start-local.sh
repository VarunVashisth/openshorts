#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
mkdir -p .local/logs output uploads .cache/huggingface
export PATH="$ROOT/.venv/bin:$ROOT/bin:$PATH"
export HF_HOME="$ROOT/.cache/huggingface"
export XDG_CONFIG_HOME="$ROOT/.config"
export MPLCONFIGDIR="$ROOT/.cache/matplotlib"
export FONTCONFIG_FILE="$ROOT/.local/fonts.conf"
# CTranslate2 needs CUDA 12 cuBLAS and cuDNN 9, installed inside this venv.
SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export FFMPEG_THREADS=2
export VITE_PROXY_TARGET=http://127.0.0.1:8000
export VITE_RENDER_TARGET=http://127.0.0.1:3100
export REMOTION_BUNDLE_PATH="$ROOT/remotion"
export OUTPUT_DIR="$ROOT/output"
pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
uvicorn app:app --host 127.0.0.1 --port 8000 > .local/logs/backend.log 2>&1 &
pids+=("$!")
(cd render-service && exec node dist/server.js) > .local/logs/renderer.log 2>&1 &
pids+=("$!")
(cd dashboard && exec node node_modules/vite/bin/vite.js --host 127.0.0.1 --port 5175 --strictPort) > .local/logs/dashboard.log 2>&1 &
pids+=("$!")
printf 'OpenShorts: http://localhost:5175\nLogs: %s/.local/logs\nPress Ctrl+C to stop.\n' "$ROOT"
wait -n
