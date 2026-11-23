#!/bin/bash
# InstaBox 上启动 minimind 推理服务（PID 文件托管；supervisord 可用时优先用 supervisord.conf）
set -u
BASE=/data/home/minimind-system
cd "$BASE" || exit 1
if [ -f "$BASE/service.pid" ] && kill -0 "$(cat "$BASE/service.pid")" 2>/dev/null; then
  echo "already running: $(cat "$BASE/service.pid")"
  exit 0
fi
set -a
. "$BASE/secrets.env"
set +a
export MODEL_DIR="$BASE/current"
export MINIMIND_REPO="${MINIMIND_REPO:-/data/home/minimind}"
export TORCH_THREADS=2
export TORCH_INTEROP_THREADS=1
mkdir -p "$BASE/logs"
nohup "$BASE/venv/bin/uvicorn" serving.app:app --host 0.0.0.0 --port 8080 \
  >> "$BASE/logs/service.log" 2>&1 &
echo $! > "$BASE/service.pid"
echo "started pid $(cat "$BASE/service.pid")"
