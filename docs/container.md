# Container images

The project publishes one public CUDA 13 OCI image to GitHub Container Registry. It is
**multi-arch**: each tag holds a `linux/amd64` and a `linux/arm64` variant. Docker, Apptainer and
SingularityCE pull the variant that matches the host, so the same name works on x86-64 nodes and on
Grace nodes (RIKYU's GB200, R-CCS's GH200), and a machine downloads only its own variant. The version
is read from `pyproject.toml`.

| Tag | Published by |
|---|---|
| `sha-<git-sha>` | every build that is not a pull request (immutable) |
| `<project.version>`, `latest` | builds of `master` (push or manual run) |

For example, version `2.0.0` publishes:

```text
ghcr.io/tsumina/yammi:2.0.0
ghcr.io/tsumina/yammi:latest
ghcr.io/tsumina/yammi:sha-<git-sha>
```

A manual run from a feature branch publishes only its SHA tag, so it cannot replace a released image.
Images before 2.0.0 were published as `ghcr.io/tsumina/foundation_model`, with separate `rikyu-*`
AArch64 tags.

Both variants are built from `Dockerfile.cuda13` on native runners, with Python 3.14, PyTorch's CUDA
13.0 wheel and the same locked application dependencies. They need a host driver that supports CUDA
13, for example the R-CCS Cloud `ai-h200-brc` H200 partition (x86-64) or RIKYU (AArch64).

## R-CCS Cloud H200

On `riken-login`, convert the versioned default image to SIF once:

```bash
cd "$HOME/projects/foundation_model-x86-cuda"
VERSION=$(sed -n 's/^version = "\([^"]*\)"$/\1/p' pyproject.toml | head -n 1)
mkdir -p "$HOME/containers"
singularity pull "$HOME/containers/yammi_cuda13-$VERSION.sif" \
  "docker://ghcr.io/tsumina/yammi:$VERSION"
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
  "$HOME/containers/yammi_cuda13-$VERSION.sif" \
  yammi pretrain --config /work/configs/pretrain.toml --output-dir /work/runs/pretrain
```

Do not use the default AMD64 image on RIKYU, and do not use the RIKYU AArch64 image on the H200
partition.
