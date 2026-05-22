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

DATE_SUFFIX="$(date +"%y%m%d")"

function has_flag() {
  local flag="$1"
  for arg in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do
    if [[ "${arg}" == "${flag}" || "${arg}" == ${flag}=* ]]; then
      return 0
    fi
  done
  return 1
}

# Append a date suffix to the config's output_dir so repeated runs don't clobber prior artifacts.
OUTPUT_OVERRIDE=()
if ! has_flag "--output-dir"; then
  OUTPUT_BASE="$(python3 - "$CONFIG_FILE" <<'PY'
import sys
from pathlib import Path

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:
        print("", end="")
        sys.exit(0)

loaded = tomllib.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = loaded.get("output_dir") if isinstance(loaded, dict) else None
if isinstance(value, str) and value.strip():
    print(value.strip(), end="")
PY
)"
  if [[ -z "${OUTPUT_BASE}" ]]; then
    OUTPUT_BASE="${REPO_ROOT}/artifacts/continual_rehearsal"
  fi
  OUTPUT_BASE="${OUTPUT_BASE%/}"
  OUTPUT_OVERRIDE=(--output-dir "${OUTPUT_BASE}_${DATE_SUFFIX}")
fi

python3 -m foundation_model.scripts.continual_rehearsal_demo --config-file "${CONFIG_FILE}" "${OUTPUT_OVERRIDE[@]+"${OUTPUT_OVERRIDE[@]}"}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
