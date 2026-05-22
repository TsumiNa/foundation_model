#!/usr/bin/env bash
# Convenience wrapper for the continual multi-task rehearsal demo.
# Usage:
#   ./run_continual_rehearsal_demo.sh [CONFIG_PATH] [-- additional CLI args...]
#
# If CONFIG_PATH is omitted, the default sample config in samples/ is used.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

DEFAULT_CONFIG="${REPO_ROOT}/samples/continual_rehearsal_demo_config.toml"

CONFIG_FILE="${1:-${DEFAULT_CONFIG}}"
shift || true
EXTRA_ARGS=()
if [[ $# -gt 0 ]]; then
  EXTRA_ARGS=("$@")
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config file not found: ${CONFIG_FILE}" >&2
  exit 1
fi

python3 -m foundation_model.scripts.continual_rehearsal_demo --config-file "${CONFIG_FILE}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
