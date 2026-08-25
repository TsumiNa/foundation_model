#!/bin/bash
# Pull the public GHCR image on RIKYU while bypassing the incomplete site
# mirror. The temporary user-level containers/image configuration is removed
# on exit.

set -euo pipefail

PROJ=${PROJ:-$HOME/projects/foundation_model}
VERSION=$(python3 -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["project"]["version"])' \
    "$PROJ/pyproject.toml")
IMAGE=${IMAGE:-$HOME/containers/foundation-model_rikyu-$VERSION.sif}
REGISTRY_CONFIG=$HOME/.config/containers/registries.conf

if [[ -e "$IMAGE" ]]; then
    echo "SIF already exists: $IMAGE"
    exit 0
fi

if [[ -e "$REGISTRY_CONFIG" ]]; then
    echo "Refusing to overwrite existing registry configuration: $REGISTRY_CONFIG" >&2
    echo "Temporarily remove or adapt that configuration before pulling GHCR." >&2
    exit 2
fi

cleanup() {
    rm -f "$REGISTRY_CONFIG"
}
trap cleanup EXIT INT TERM

mkdir -p "$(dirname "$IMAGE")" "$(dirname "$REGISTRY_CONFIG")"
printf '%s\n' \
    'unqualified-search-registries = ["docker.io"]' \
    '' \
    '[[registry]]' \
    'location = "ghcr.io"' >"$REGISTRY_CONFIG"

apptainer pull "$IMAGE" "docker://ghcr.io/tsumina/foundation_model:rikyu-$VERSION"
echo "Pulled $IMAGE"
