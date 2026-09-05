#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRIPT_DIR="$ROOT_DIR/dashboard/scripts"
# shellcheck source=load-config.sh
source "$SCRIPT_DIR/load-config.sh"

SSH_ARGS=(
  -o BatchMode=yes
  -o ConnectTimeout=12
  -o StrictHostKeyChecking=accept-new
  -i "$RUNPOD_SSH_KEY"
  -p "$RUNPOD_PORT"
)
SCP_ARGS=(
  -o BatchMode=yes
  -o ConnectTimeout=12
  -o StrictHostKeyChecking=accept-new
  -i "$RUNPOD_SSH_KEY"
  -P "$RUNPOD_PORT"
)
REMOTE="$RUNPOD_USER@$RUNPOD_HOST"

printf -v preflight_command 'test -x %q && test -x %q' "$REMOTE_PYTHON" "$REMOTE_VLLM_BIN"
if ! ssh "${SSH_ARGS[@]}" "$REMOTE" "$preflight_command"; then
  echo "The Runpod vLLM environment is missing. Run $SCRIPT_DIR/bootstrap-new-runpod.sh first." >&2
  exit 1
fi

printf -v make_dir_command 'mkdir -p %q' "$REMOTE_APP_DIR"
ssh "${SSH_ARGS[@]}" "$REMOTE" "$make_dir_command"
scp "${SCP_ARGS[@]}" \
  "$ROOT_DIR/dashboard/server/benchmark_server.py" \
  "$REMOTE:$REMOTE_APP_DIR/benchmark_server.py"
scp "${SCP_ARGS[@]}" \
  "$ROOT_DIR/colab/a100_40GB_concurrency_client_stage1/a100_concurrency_sweep/concurrency_sweep.py" \
  "$REMOTE:$REMOTE_SWEEP_SCRIPT"
scp "${SCP_ARGS[@]}" "$CONFIG_FILE" "$REMOTE:$REMOTE_APP_DIR/server.env"

printf -v remote_command 'bash -s -- %q' "$REMOTE_APP_DIR"
ssh "${SSH_ARGS[@]}" "$REMOTE" "$remote_command" <<'REMOTE_SCRIPT'
set -euo pipefail
app_dir=$1
set -a
# shellcheck disable=SC1090
source "$app_dir/server.env"
set +a

mkdir -p "$REMOTE_RUN_ROOT" "$REMOTE_HF_HOME"
pid_file="$REMOTE_WORKSPACE/benchmark_server.pid"
server_log="$REMOTE_WORKSPACE/benchmark_server.log"
if [ -f "$pid_file" ]; then
  pid=$(cat "$pid_file")
  if kill -0 "$pid" 2>/dev/null && ps -p "$pid" -o args= | grep -q benchmark_server; then
    kill "$pid"
    for _ in 1 2 3 4 5; do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  fi
fi

cd "$REMOTE_APP_DIR"
nohup env \
  SWEEP_SCRIPT="$REMOTE_SWEEP_SCRIPT" SWEEP_PYTHON="$REMOTE_PYTHON" \
  VLLM_BIN="$REMOTE_VLLM_BIN" HF_HOME="$REMOTE_HF_HOME" \
  SWEEP_RUN_ROOT="$REMOTE_RUN_ROOT" SWEEP_OUTPUT="$REMOTE_OUTPUT" \
  BENCHMARK_MODEL="$BENCHMARK_MODEL" BENCHMARK_CONCURRENCY="$BENCHMARK_CONCURRENCY" \
  BENCHMARK_MAX_NUM_SEQS="$BENCHMARK_MAX_NUM_SEQS" \
  BENCHMARK_MAX_NUM_BATCHED_TOKENS="$BENCHMARK_MAX_NUM_BATCHED_TOKENS" \
  BENCHMARK_MAX_MODEL_LEN="$BENCHMARK_MAX_MODEL_LEN" \
  BENCHMARK_GPU_MEMORY_UTILIZATION="$BENCHMARK_GPU_MEMORY_UTILIZATION" \
  BENCHMARK_INPUT_TOKENS="$BENCHMARK_INPUT_TOKENS" \
  BENCHMARK_OUTPUT_TOKENS="$BENCHMARK_OUTPUT_TOKENS" \
  "$REMOTE_PYTHON" -m uvicorn benchmark_server:app \
  --host "$REMOTE_API_HOST" --port "$REMOTE_API_PORT" \
  > "$server_log" 2>&1 < /dev/null &
echo $! > "$pid_file"

for _ in $(seq 1 20); do
  if curl -fsS "http://$REMOTE_API_HOST:$REMOTE_API_PORT/api/health"; then echo; exit 0; fi
  sleep 1
done
echo "Runpod benchmark API failed to become healthy" >&2
exit 1
REMOTE_SCRIPT
