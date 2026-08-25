#!/bin/bash
#SBATCH --job-name=fm-h200-container-smoke
#SBATCH --partition=ai-h200-brc
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /etc/profile
set -euo pipefail
module load system/ai-h200-brc nvhpc

PROJ=${PROJ:-$HOME/projects/foundation_model-x86-cuda}
cd "$PROJ"

VERSION=$(sed -n 's/^version = "\([^"]*\)"$/\1/p' pyproject.toml | head -n 1)
test -n "$VERSION"
IMAGE=${IMAGE:-$HOME/containers/foundation-model_cuda13-$VERSION.sif}

test -r "$IMAGE"

echo "host=$(hostname) image=$IMAGE"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
singularity exec --nv \
    --bind "$PROJ:/workspace:ro" \
    --pwd /workspace \
    "$IMAGE" \
    python scripts/rikyu_container_smoke.py --require-cuda --expected-arch x86_64
