#!/bin/bash
# Export every consolidated member into ONE bundle, then load it back the way a user would.
#
# fm export reloads each member and requires its outputs to equal the source checkpoint's exactly, and
# removes the bundle otherwise. After that this job checks the user-facing paths on the bundle alone:
# load_suite / predict / predict_curve in Python, and fm predict --pretrained on the CLI.
#
# Required env: PROJ IMAGE EXPECT_VERSION SET OUTROOT; optional MEMBER_LIMIT, BUNDLE (default "bundle").
# The bundle goes to $OUTROOT/$BUNDLE, which must not exist.
#
#SBATCH --job-name=fm-release-export
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
module purge
: "${PROJ:?}" "${IMAGE:?}" "${EXPECT_VERSION:?}" "${SET:?}" "${OUTROOT:?}"
EXP=experiments/release_suite_v1_20261108
MEMBERS="$PROJ/$EXP/configs/members_$SET.txt"
LICENSE=cc-by-nc-sa-4.0
BUNDLE=${BUNDLE:-bundle}

GOT=$(apptainer exec "$IMAGE" python -c 'import importlib.metadata as m; print(m.version("foundation-model"))')
[ "$GOT" = "$EXPECT_VERSION" ] || { echo "FAIL image reports $GOT, expected $EXPECT_VERSION"; exit 3; }
[ ! -e "$OUTROOT/$BUNDLE" ] || { echo "FAIL $OUTROOT/$BUNDLE exists; export into a new directory"; exit 2; }

args=()
while IFS=$'\t' read -r id _seed; do
    [ -f "$OUTROOT/$id/consolidate/DONE" ] || { echo "FAIL $id has no finished consolidation"; exit 4; }
    args+=(--run "/out/$id/consolidate" --member-id "$id")
done < <(head -n "${MEMBER_LIMIT:-1000}" "$MEMBERS")  # MEMBER_LIMIT: smoke exports only the members it trained

run() {
    apptainer exec --nv --bind "$PROJ:/workspace:ro" --bind "$OUTROOT:/out" \
        --env MPLCONFIGDIR=/tmp/mplconfig_export --pwd /workspace "$IMAGE" "$@"
}

run fm export --config "$EXP/configs/export_$SET.toml" "${args[@]}" --out "/out/$BUNDLE" --license "$LICENSE"

# The user-facing paths, on the bundle alone (no catalog, no data).
run env BUNDLE="$BUNDLE" python - <<'PY'
import warnings
from foundation_model.hub import load_pretrained, load_suite

import os
bundle = "/out/" + os.environ["BUNDLE"]
suite = load_suite(bundle)
model = load_pretrained(bundle)
print("members:", suite.member_ids, "default:", model.member.id, "licence:", model.license)
orders = {m.member.id: m.member.task_order for m in suite}
assert len(set(orders.values())) == len(orders), "two members share one task order"
print("first task per member:", {k: v[0] for k, v in orders.items()})
df = model.predict(["Fe2O3", "NaCl", "Bi2Te3", "SiO2"])
# From 1.0.1 every bounded transform is guarded, so a NaN here means the export missed one.
nan_cols = [c for c in df.columns if df[c].isna().any()]
assert not nan_cols, f"NaN predictions in {nan_cols}"
guarded = [h.name for h in model.config.heads if h.transform is not None and h.transform.steps[-1].kind == "clip"
           and getattr(h.transform.steps[-1], "guard", False)]
print(f"schema {model.config.schema_version}; {len(guarded)} guarded heads: {guarded}")
# Far past the training range the guarded heads saturate instead of returning NaN.
import numpy as np
for name in guarded:
    head = model.config.head(name)
    far = head.transform.inverse(np.array([1e6, -1e6]))
    assert np.isfinite(far).all(), f"{name}: guard does not saturate"
print(df.T.to_string())
with warnings.catch_warnings(record=True):
    curve = model.predict_curve(["Bi2Te3"], task="seebeck")
print("seebeck curve:", curve.shape)
PY
printf 'Fe2O3\nNaCl\nBi2Te3\n' >"$OUTROOT/_export_check_compositions.txt"
# 1.0.1: a local bundle takes a "#member" suffix, as Hub sources do.
run fm predict --pretrained "/out/$BUNDLE#m001" --compositions-file /out/_export_check_compositions.txt \
    --output-dir "/out/_export_check_predict_$BUNDLE"
ls -la "$OUTROOT/_export_check_predict_$BUNDLE"
du -sh "$OUTROOT/$BUNDLE"
echo "PASS export $SET"
