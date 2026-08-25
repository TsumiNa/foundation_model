# Container images

The project publishes public CUDA 13 OCI images to GitHub Container Registry. Image versions are
read from `pyproject.toml`.

| Target | Platform | Tags |
|---|---|---|
| Default NVIDIA CUDA 13 | `linux/amd64` | `<project.version>`, plus `latest` on publishing runs |
| RIKEN R-CCS RIKYU | `linux/arm64` | `rikyu-<project.version>` |

For example, version `0.2.1` publishes:

```text
ghcr.io/tsumina/foundation_model:0.2.1
ghcr.io/tsumina/foundation_model:latest
ghcr.io/tsumina/foundation_model:rikyu-0.2.1
```

Both images use Python 3.14, PyTorch's CUDA 13.0 wheel, and the same locked application
dependencies. The default image is suitable for x86-64 NVIDIA data-center GPUs whose host driver
supports CUDA 13, including the R-CCS Cloud `ai-h200-brc` H200 partition. The RIKYU image is the
AArch64 counterpart for Grace/GB200 nodes.

## R-CCS Cloud H200

On `riken-login`, convert the versioned default image to SIF once:

```bash
cd "$HOME/projects/foundation_model-x86-cuda"
VERSION=$(sed -n 's/^version = "\([^"]*\)"$/\1/p' pyproject.toml | head -n 1)
mkdir -p "$HOME/containers"
singularity pull "$HOME/containers/foundation-model_cuda13-$VERSION.sif" \
  "docker://ghcr.io/tsumina/foundation_model:$VERSION"
```

Run the GPU smoke test from a job directory so Slurm logs remain outside the repository:

```bash
mkdir -p "$HOME/jobs/h200-container-smoke"
cd "$HOME/jobs/h200-container-smoke"
PROJ="$HOME/projects/foundation_model-x86-cuda" \
  sbatch "$PROJ/scripts/riken_h200_container_smoke.sbatch"
```

For normal tasks, bind datasets, configs, checkpoints, and outputs into the read-only SIF:

```bash
singularity exec --nv \
  --bind "$RCCS_GROUP_DIR:/work" \
  "$HOME/containers/foundation-model_cuda13-$VERSION.sif" \
  fm pretrain --config /work/configs/pretrain.toml --output-dir /work/runs/pretrain
```

Do not use the default AMD64 image on RIKYU, and do not use the RIKYU AArch64 image on the H200
partition.
