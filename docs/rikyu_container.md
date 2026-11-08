# RIKYU container image

RIKYU uses the project's multi-arch image from GitHub Container Registry (GHCR); on RIKYU's AArch64
nodes Apptainer pulls its `linux/arm64` variant automatically:

```text
ghcr.io/tsumina/yammi:<project.version>
```

The version is read from `pyproject.toml`. Every published build also has an immutable
`sha-<commit>` tag. Pull requests build the image without publishing it; builds of `master` publish
the version tag and `latest`. See `docs/container.md`.

## Compatibility choices

RIKYU compute and login nodes are AArch64. The workflow builds the `linux/arm64` variant on GitHub's
native `ubuntu-24.04-arm` runner, next to the `linux/amd64` one. The locked Linux environment installs PyTorch's CUDA 13 wheel.
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
bash scripts/rikyu_pull_container.sh
```

RIKYU configures an internal `ghcr.io` mirror. In testing on 2026-08-26, that mirror returned
`NAME_UNKNOWN` for this repository even though direct anonymous HTTPS access to the public GHCR
image worked. The helper temporarily installs a user-level `containers/image` configuration that
uses GHCR directly, removes it on exit, and refuses to overwrite an existing user configuration.
If site network policy changes, transfer an already-built SIF to RIKYU instead.

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
  "$HOME/containers/yammi-$VERSION.sif" \
  yammi pretrain --config /work/configs/pretrain.toml --output-dir /work/runs/pretrain
```

Do not write results into the SIF: it is read-only. Keep configs, data, checkpoints, and run
outputs in `$HOME` or `$RIKYU_GROUP_DIR`.
