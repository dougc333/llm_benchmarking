#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=load-config.sh
source "$SCRIPT_DIR/load-config.sh"

SSH_ARGS=(
  -o BatchMode=yes
  -o ConnectTimeout=12
  -o StrictHostKeyChecking=accept-new
  -i "$RUNPOD_SSH_KEY"
  -p "$RUNPOD_PORT"
)
REMOTE="$RUNPOD_USER@$RUNPOD_HOST"

printf -v remote_command 'bash -s -- %q %q %q' \
  "$REMOTE_PYTHON" "$REMOTE_PYTHON_VERSION" "$VLLM_VERSION"

ssh "${SSH_ARGS[@]}" "$REMOTE" "$remote_command" <<'REMOTE_SCRIPT'
set -euo pipefail
python_path=$1
python_version=$2
vllm_version=$3
venv_dir=$(dirname "$(dirname "$python_path")")

command -v nvidia-smi >/dev/null || {
  echo "nvidia-smi is unavailable. Start a Runpod GPU template with NVIDIA drivers." >&2
  exit 1
}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if command -v uv >/dev/null 2>&1; then
  uv_command=(uv)
else
  python3 -m pip install --user --upgrade uv
  uv_command=(python3 -m uv)
fi

if [ ! -x "$python_path" ]; then
  "${uv_command[@]}" venv "$venv_dir" \
    --python "$python_version" --seed --managed-python
fi

"${uv_command[@]}" pip install \
  --python "$python_path" \
  "vllm==$vllm_version" fastapi uvicorn ninja \
  --torch-backend=auto

"$python_path" - <<'PY'
import fastapi
import torch
import uvicorn
import vllm

print("vLLM:", vllm.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("FastAPI:", fastapi.__version__)
print("Uvicorn:", uvicorn.__version__)
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the Runpod GPU")
PY
REMOTE_SCRIPT

"$SCRIPT_DIR/deploy-runpod-server.sh"

cat <<EOF
Fresh-pod installation complete.
Next, open the tunnel on macOS:
  $SCRIPT_DIR/open-runpod-tunnel.sh
EOF
