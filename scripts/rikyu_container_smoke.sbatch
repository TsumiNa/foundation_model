#!/bin/bash
#SBATCH --job-name=fm-container-smoke
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

module purge

PROJ=${PROJ:-$HOME/projects/foundation_model}
cd "$PROJ"

VERSION=$(python3 -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')
IMAGE=${IMAGE:-$HOME/containers/foundation-model_rikyu-$VERSION.sif}

test -r "$IMAGE"

echo "host=$(hostname) image=$IMAGE"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
apptainer exec --nv \
    --bind "$PROJ:/workspace:ro" \
    --pwd /workspace \
    "$IMAGE" \
    python scripts/rikyu_container_smoke.py --require-cuda --expected-arch aarch64
