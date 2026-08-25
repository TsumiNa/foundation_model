# RIKYU container image

The repository publishes an OCI image for RIKEN R-CCS RIKYU to GitHub Container Registry (GHCR):

```text
ghcr.io/tsumina/foundation_model:rikyu-<project.version>
```

The version is read from `pyproject.toml`; for example, project version `0.2.1` produces
`rikyu-0.2.1`. Pull requests build the image without publishing it. Pushes to `master` and manual
workflow runs publish it.

## Compatibility choices

RIKYU compute and login nodes are AArch64, so the workflow builds only `linux/arm64` on GitHub's
native `ubuntu-24.04-arm` runner. The locked Linux environment installs PyTorch's CUDA 13 wheel.
At runtime, Apptainer's `--nv` option supplies the host NVIDIA driver; the image supplies PyTorch
and its CUDA user-space libraries.

The image intentionally contains code and runtime dependencies, but not datasets, configs,
checkpoints, or outputs. Bind those from RIKYU storage when the container runs.

## Publish and pull

The GHCR package must be public. GitHub creates a new package as private by default, so after its
first publication open the package settings once and change its visibility to **Public**. A public
package can be pulled from RIKYU without storing GitHub credentials there.

On the RIKYU login node, update the clone and convert the versioned OCI image to SIF once:

```bash
cd "$HOME/projects/foundation_model"
git pull --ff-only

VERSION=$(python3 -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')
mkdir -p "$HOME/containers"
apptainer pull --force "$HOME/containers/foundation-model_rikyu-$VERSION.sif" \
  "docker://ghcr.io/tsumina/foundation_model:rikyu-$VERSION"
```

RIKYU's documented registry mirror explicitly lists Docker Hub, NVIDIA NGC, and Quay rather than
GHCR. Direct HTTPS access to GHCR was verified from the RIKYU login node on 2026-08-25. If site
network policy changes, transfer the already-built SIF to RIKYU instead of rebuilding dependencies
on the cluster.

## GPU smoke test

The smoke test performs a real model forward and backward pass on an allocated B200 GPU:

```bash
mkdir -p "$HOME/jobs"
cd "$HOME/jobs"
sbatch --account="$RIKYU_ACCOUNT" "$HOME/projects/foundation_model/scripts/rikyu_container_smoke.sbatch"
```

The successful job prints `RIKYU_CONTAINER_SMOKE_PASS` together with the architecture, package
version, PyTorch version, CUDA runtime, GPU model, and a finite loss.

For a normal run, explicitly bind the group storage because Apptainer does not bind `/data1`
automatically:

```bash
apptainer exec --nv \
  --bind "$RIKYU_GROUP_DIR:/work" \
  "$HOME/containers/foundation-model_rikyu-$VERSION.sif" \
  fm pretrain --config /work/configs/pretrain.toml --output-dir /work/runs/pretrain
```

Do not write results into the SIF: it is read-only. Keep configs, data, checkpoints, and run
outputs in `$HOME` or `$RIKYU_GROUP_DIR`.
