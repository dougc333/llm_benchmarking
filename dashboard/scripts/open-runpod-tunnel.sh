#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=load-config.sh
source "$SCRIPT_DIR/load-config.sh"

ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o StrictHostKeyChecking=accept-new \
  -i "$RUNPOD_SSH_KEY" \
  -p "$RUNPOD_PORT" \
  -L "$LOCAL_API_PORT:$REMOTE_API_HOST:$REMOTE_API_PORT" \
  "$RUNPOD_USER@$RUNPOD_HOST"
