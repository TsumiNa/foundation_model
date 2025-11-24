#!/usr/bin/env bash
# Convenience wrapper for running the dynamic task suite with a config file.
# Usage:
#   ./run_dynamic_task_suite.sh [CONFIG_PATH] [-- additional CLI args...]
#
# If CONFIG_PATH is omitted, the default sample config in samples/ is used.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

DEFAULT_CONFIG="${REPO_ROOT}/samples/dynamic_task_suite_config.toml"

CONFIG_FILE="${1:-${DEFAULT_CONFIG}}"
shift || true
EXTRA_ARGS=()
if [[ $# -gt 0 ]]; then
  EXTRA_ARGS=("$@")
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config file not found: ${CONFIG_FILE}" >&2
  echo "Provide a valid path or update samples/dynamic_task_suite_config.toml." >&2
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

OUTPUT_OVERRIDE=()
if ! has_flag "--output-dir"; then
  OUTPUT_BASE="$(python3 - "$CONFIG_FILE" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
text = config_path.read_text(encoding="utf-8")
suffix = config_path.suffix.lower()
data = {}
if suffix in {".yaml", ".yml"}:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        print("", end="")
        sys.exit(0)
    loaded = yaml.safe_load(text)
elif suffix == ".toml":
    try:
        import tomllib  # type: ignore[attr-defined]
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore
        except ModuleNotFoundError:
            print("", end="")
            sys.exit(0)
    loaded = tomllib.loads(text)
else:
    loaded = {}
if isinstance(loaded, dict):
    value = loaded.get("output_dir")
    if isinstance(value, str) and value.strip():
        print(value.strip(), end="")
PY
)"
  if [[ -z "${OUTPUT_BASE}" ]]; then
    OUTPUT_BASE="${REPO_ROOT}/outputs"
  fi
  OUTPUT_BASE="${OUTPUT_BASE%/}"
  OUTPUT_OVERRIDE=(--output-dir "${OUTPUT_BASE}_${DATE_SUFFIX}")
fi

python3 -m foundation_model.scripts.dynamic_task_suite --config-file "${CONFIG_FILE}" "${OUTPUT_OVERRIDE[@]+"${OUTPUT_OVERRIDE[@]}"}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
