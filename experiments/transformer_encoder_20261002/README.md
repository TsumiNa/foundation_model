# Transformer encoder investigation — 2026-10-02

## Full-budget concat confirmation (2026-10-03)

All 27 short stability trajectories completed successfully. Lowering concat's encoder learning
rate from 3e-4 to 1e-5 removed catastrophic loss growth in all three seeds over the three-task,
12-epoch-per-task diagnostic. At task 3, the mean fraction of latent dimensions with variance
below 1e-6 on the fixed stratified validation probe was 1.04%, versus 20.23% at 3e-5 and 100%
at 3e-4. Mean final validation loss was 2.35 at 1e-5; the matched grouped-CLS short-run control
was 2.42. These source losses do not establish a downstream transfer improvement. Legacy CLS
remained poorly optimized across the tested rates, with very small between-composition variance.

`configs/concat_confirmation.toml` registers a separate full-budget confirmation of concat at
1e-5, selected using source validation loss and representation diagnostics only. It preserves
the main screen's data, three seeds, architecture, heads, seven source tasks, stopping rules,
1/3/7 checkpoints, target budgets and scratch/frozen/full comparisons: three trajectories and
84 target fits. All task-end checkpoints remain available. Continuation and transfer still use
task-end weights, matching the main protocol; the short diagnostic alone saves best/last files.

Full fine-tuning retains the existing 0.1 encoder-LR multiplier, giving 1e-6. Its scheduler floor
must therefore fall below the original 1e-6 floor. The experiment template uses
`min(1e-6, effective_encoder_lr / 10)`: 1e-6 for source/scratch and 1e-7 for frozen/full in this
confirmation. This is a global floor for all optimizer groups because the CLI exposes a common
floor; disclose it alongside the learning rate. Every original screen configuration keeps its
previous floor and learning rates. No production-package or image change is required.

After review and merge, launch `scripts/array.sbatch` with `FM_PROTOCOL` pointing to the merged
confirmation TOML and a separate scripts/data workspace and output root. Generate a worklist
from that protocol; the existing calibrated PACK=3 uses one GPU for its three seeds. First run
an image-only two-epoch functional check through all seven source stages, all target modes,
prediction and both inverse paths. Never mix that smoke or the 12-epoch diagnostic into formal
results. Collect the 84 fits with the confirmation protocol, then compare explicitly against
the matched main-screen controls; preserve campaign identities rather than pooling them.

## Optimization-stability diagnostic (2026-10-03)

The first screen exposed severe grouped-concat instability across all three seeds, including
validation losses above 1e9; legacy CLS also has large validation spikes. A read-only inspection
of the concat seed-1 checkpoint after its first task found all post-tanh latent coordinates
saturated on 128 validation compositions, negligible between-composition variance, and nearly
zero BatchNorm running variances in the classification head. These observations are evidence
of optimization failure, not evidence that concatenation or Transformers intrinsically harm
transfer. Most MLP and modern CLS/mean runs have descending, then stabilizing, stage losses.

The registered diagnostic in `configs/stability.toml` varies only the encoder learning rate
(3e-4, 1e-4, 3e-5, 1e-5) for legacy CLS and grouped concat, with three paired seeds. Three
grouped-CLS runs at the original learning rate provide a control: 27 trajectories total.
Each introduces only the first three source tasks, with a maximum of 12 epochs per task.
Data, heads, head/decoder learning rates, replay and initialization remain matched. Both
best-validation and last-epoch Lightning checkpoints supplement the ordinary end-of-task
checkpoints. Diagnose per-task losses, stored optimizer learning rates, pre-tanh amplitudes,
saturation, between-composition latent variance and head BatchNorm statistics. Use a fixed
validation subset stratified by material-type label for representation diagnostics; it is
not a generalization-performance estimate. Target test labels are not used to choose settings.

This is an optimization pilot, not a replacement for the 1/3/7-task transfer screen. Its
shorter training budget and separate output identity prohibit pooling its results with that
screen. Retain the existing bounded campaign as a diagnostic baseline; stabilize affected
settings before drawing architectural or transfer-scaling conclusions. The user requested
about ten or fewer source tasks: the main round remains at seven, with checkpoints after
every task. Package 0.5.0 and its pinned ARM image suffice for this experiment-only change.
Review and merge this diagnostic PR before GPU execution.

The image-only runner is `scripts/stability_probe.py`; `scripts/stability.sbatch` verifies
the SIF and launches it on one allocated GPU. Supply `FM_WORKSPACE`, `FM_IMAGE`, `FM_DATA_DIR`,
`FM_OUTPUT_ROOT` and `FM_BENCHMARK_REVISION` externally, with a scripts/data-only workspace
from the merged commit and a separate output root. Submit cases 0–11 with `FIRST_CASE=0`,
`PACK=1`, `--array=0-11%3`; submit cases 12–26 with `FIRST_CASE=12`, `PACK=3`,
`--cpus-per-task=12`, `--array=0-4%1`. These limits add at most four GPUs. Grouped packing
uses the already measured three-process calibration; scalar-token legacy remains unpacked.
Each `probe_done.json` stores diagnostics and hashes for every best, last and end checkpoint.
Per-stage CSV logs retain training/validation losses; the Lightning files retain optimizer
learning rates and callback scores. Continuation still uses the end-of-stage weights to match
the baseline: saving a best checkpoint does not itself change the continuation algorithm.

Status: core encoder PR [#69](https://github.com/TsumiNa/foundation_model/pull/69) squash-merged
as `950012a581d838601530eb200edb9572862457aa`; package 0.5.0 ARM image published and verified
on RIKYU. The benchmark is the second implementation PR. No scientific GPU job has started.
This document does not report a Transformer improvement.

Execution update (2026-10-03 JST): benchmark PR [#70](https://github.com/TsumiNa/foundation_model/pull/70)
squash-merged as `c15ca0ed5a956d75d7b81cc9665fe4e8afd1867b` after all review findings were
addressed. The final 34 focused tests passed, along with Ruff, package-wide mypy and the complete
local smoke (seven source stages and 28 target fits, two actual epochs each, both inverse paths).
A completed-source CLI resume probe also passed without retraining. RIKYU GPU smoke array
`161531` was submitted from an isolated scripts/data workspace on group storage, using only the
verified 0.5.0 image. Packing calibration and scientific results remain pending GPU smoke.
GPU smoke `161531` completed all seven source additions and the first target fit, then failed
configuration validation because `fm predict` accepts `auto`/`cpu`, unlike training's `gpu`.
Fix the experiment's prediction/inverse templates to use `auto` on allocated GPUs and validate
both inference modes before retrying. The runtime package/image is unchanged.

Reviewed repository commit: `d0234e451f0ab747f2ebe474a62f7b499f83bae3` (package 0.4.1). Research branch: `codex/transformer-research-20261002`.

## Approved execution amendments

The user authorized implementation and one completed experimental round, then specified:

- Transfer performance and its scaling with source task count are the central outcomes; source
  training accuracy is diagnostic only.
- Add concat pooling: concatenate the final feature tokens and project to a fixed latent width.
  Compare CLS, mean and concat at the same output width and report their parameter counts.
- The fixed element-table alternative has 94 positions (not 92); its input contract remains a
  subsequent architectural experiment.
- Review and merge implementation PRs, update the version, and publish the new RIKYU ARM image
  before training. Run the image-installed package pinned to its immutable SHA; do not substitute
  checkout source through `PYTHONPATH`. The sequential gates are recorded in
  `docs/refactor/transformer-encoders/`.

First round: eight encoder settings × three paired seeds = 24 continual source trajectories.
Retain source checkpoints after 1, 3 and 7 tasks. Compare independent target training, frozen
transfer and full fine-tuning to held-out dielectric-total regression and power-factor functions,
using nested 10% / 100% target-training label budgets. No held-out target labels are source tasks.
The same global composition split and the same target subsets/transforms are shared by every arm.
Material-type classification is the initial source task because its labels cover the entire QC
composition universe, anchoring descriptor exposure across source counts. Subsequent source tasks
are volume, formation energy, Efermi, band gap, Seebeck and ZT. Source task count is also a change
in accumulated continual-training compute; report that cost rather than attributing its effect
solely to task diversity. Source/target physical relationships are intentional transfer signals;
power factor is related to source thermoelectric properties, whereas dielectric total probes a
different property family.

## First-round execution

The canonical protocol is `configs/protocol.toml`. Its 24 source trajectories yield 576 transfer
fits (8 encoders × 3 seeds × 3 source counts × 2 targets × 2 budgets × 2 fine-tuning modes)
and 96 independent target-training fits: **672 target fits**. Each source trajectory introduces
seven tasks; it is not a single-task training run. The larger MLP approximately matches concat's
encoder parameter count. Feature and grouped tokenizers use the same modern attention backbone;
the grouped token-wise feed-forward control removes attention. Latent width, target heads and
the reconstruction decoder are identical across arms.

This is a fixed-hyperparameter first screen, not the larger validation search proposed below.
MLP encoder LR is 0.002; Transformer encoder LR is 0.0003; full fine-tuning uses one tenth of
the corresponding source encoder LR. Target-head initialization is paired by target and seed
across source counts and training modes. Architecture-specific optimizer settings and the
three-seed budget limit the conclusions. Source accuracy cannot decide the winner.

By the user's shared-data convention, preprocessing lives in
`data/scripts/process_transformer_transfer_data_20261002.py`; its date-suffixed Parquets,
scalers and manifest live under `data/transformer_transfer_20261002/`. Other experiment-owned
code stays in this folder. Atomic-fraction aliases inherit the strictest split (test > val >
train); first records are retained without merging phases. Invalid compositions outside the
94-element vocabulary are excluded. Source data excludes global test compositions and both
held-out target columns. Affine target transforms are fitted on each selected training subset
only and inverted for physical-unit scoring. Existing interpolated temperature grids are
retained; no additional smoothing is applied. The manifest records source/data/scaler hashes,
descriptor order, subset membership and the split audit.

Local preparation and validation:

```bash
uv run python data/scripts/process_transformer_transfer_data_20261002.py
uv run pytest data/scripts/process_transformer_transfer_data_20261002_test.py \
  experiments/transformer_encoder_20261002/scripts/benchmark_test.py \
  experiments/transformer_encoder_20261002/scripts/make_worklist_test.py \
  experiments/transformer_encoder_20261002/scripts/make_smoke_data_test.py \
  experiments/transformer_encoder_20261002/scripts/run_lane_test.py \
  experiments/transformer_encoder_20261002/analysis/collect_test.py
uv run python experiments/transformer_encoder_20261002/scripts/make_smoke_data.py \
  --data-dir data/transformer_transfer_20261002 --output data/transformer_transfer_smoke_20261002
uv run python experiments/transformer_encoder_20261002/scripts/make_worklist.py \
  --protocol experiments/transformer_encoder_20261002/configs/protocol.toml \
  --output experiments/transformer_encoder_20261002/artifacts/worklists/main.tsv
```

After this PR is reviewed and merged, stage only this experiment's scripts/configs and prepared
data into an isolated RIKYU workspace. Do not bind `src/` or set `PYTHONPATH`. Export runtime
values `FM_WORKSPACE`, `FM_IMAGE`, `FM_WORKLIST`, `FM_DATA_DIR`, `FM_OUTPUT_ROOT`,
`FM_BENCHMARK_REVISION` (merged benchmark commit), and supply the account outside Git.
The image URI/revision and verified SIF hash are registered in the protocol. `array.sbatch`
verifies that exact SIF; workers enforce ARM, the installed package version/path, one allocated
CUDA GPU, dataset/scaler checksums and an unchanged lane identity on resume.
The protocol also pins the scientific data-manifest checksum. Collection checks this hash,
the complete protocol checksum and one common benchmark revision across all lanes.

Submit `scripts/array.sbatch` from a campaign log directory using `sbatch --account=...`.
Pass `PACK` explicitly and size the array to `ceil(number_of_lanes / PACK)`; adjacent lanes
run concurrently on one GPU. The script defaults to 8 CPUs/GPU, below the enforced 32-CPU cap.
Completed target fits are skipped; unfinished source pretraining resumes from its workflow
checkpoints. Source checkpoints and representative target checkpoints are retained; all
validation/test predictions survive. Do not use the same output root for different modes,
protocols, epoch caps or dataset manifests.

Before the scientific fleet, run one allocated-GPU 400-record functional fixture with
`MAX_EPOCHS=2`, `SOURCE_COUNT=7`, `INVERSE_SMOKE=1`, `PACK=1`, one grouped-concat/seed-0 lane
and a separate smoke output root. This exercises all seven source additions, replay, frozen/full
transfer, prediction, and latent/composition inverse.
The fixture uses batch size 16 so its 32-row low-budget training sets produce real optimizer
updates despite `drop_last`; zero-epoch source stages and target fits are rejected.
Then calibrate representative MLP,
legacy scalar-token, feature-token and grouped workloads: source-only (`SOURCE_ONLY=1`,
`SOURCE_COUNT=1`) on full data for paired seeds 0/1, unpacked first, packed reruns into separate
roots. Use `sacct --format=JobID,Elapsed,AllocTRES,TRESUsageInAve,TRESUsageInMax,ExitCode -P`
to measure utilization, memory and billed allocation time. Choose packing from measured
throughput and contention; previous MLP packing factors are not transferable evidence.

Mirror finished outputs continuously with rsync (routine mirrors exclude `*.pt`, lightning
logs and smoke roots), then collect partial or complete results locally:

```bash
uv run python experiments/transformer_encoder_20261002/analysis/collect.py \
  --root experiments/transformer_encoder_20261002/artifacts/scientific \
  --protocol experiments/transformer_encoder_20261002/configs/protocol.toml \
  --output experiments/transformer_encoder_20261002/results
```

The collector refuses functional/CPU/epoch-capped runs. RMSE averages squared error within each
composition, averages across compositions, then takes the square root. Relative change is
`100 × (transfer RMSE / paired same-encoder scratch RMSE − 1)`; negative values favor transfer.
Also compare against the tuned MLP scratch baseline. Source-count plots show mean ± sample SD
across paired seeds, with counts identifying partial groups. These bars are not predictive
uncertainty. The source-count comparison also changes accumulated compute; it does not establish
a universal scaling law. Test results are confirmation of the fixed protocol; later tuning must
use validation only, followed by a fresh-seed confirmation. Generated artifacts/results remain
untracked and travel through rsync, not Git.

Execution log: 2026-10-02/03 — PR #69 reviewed and merged; ARM 0.5.0 image pulled and checked
against its OCI revision/version labels and SIF SHA256. Local concat functional checks passed
pretraining, scalar/function scratch/frozen/full fits, prediction, and both inverse paths.
Allocated-GPU smoke, packing calibration and scientific results remain pending the benchmark PR.
Smoke log audit caught a zero-training low-budget fixture at batch 128; reduce only functional
fixtures to batch 16 and rerun. Scientific low-budget subsets exceed batch 128.

The original sections below describe the reviewed base commit and broader research proposal;
they are not a claim that the new implementation or campaign has already completed. Current
execution status is recorded as work proceeds.

## Recommendation

Prioritize a **Transformer over KMD feature groups**, evaluated against a feature-token Transformer and strong MLP controls. Preserve the composition descriptor, shared latent interface, task heads, and differentiable inverse-design path. First determine whether attention improves supervised multi-task representations. Evaluate masked pretraining or JEPA only after this comparison.

This is a proposed adaptation of established tabular Transformer ideas, not a claim of a new validated architecture. The main hypothesis is that a token should represent a meaningful descriptor group instead of one arbitrary scalar column.

## 1. What the current model actually does

The relevant flow is:

```text
composition → KMD descriptor X → shared encoder → tanh(h)
                                            ├→ regression head
                                            ├→ classification head
                                            ├→ kernel-regression head, also receiving t
                                            └→ descriptor reconstruction head
```

The encoder sees neither the temperature grid nor other samples in a batch. It is a feature encoder, not an in-context learner and not a temporal model. Replacing it can improve the composition dependence of a curve; it does not directly add attention along temperature. Kernel regression remains responsible for the dependence on the continuous coordinate.

Current Python API usage (the existing scalar-token implementation, not the proposed grouped model):

```python
from foundation_model.models.flexible_multi_task_model import FlexibleMultiTaskModel
from foundation_model.models.model_config import RegressionTaskConfig, TransformerEncoderConfig

model = FlexibleMultiTaskModel(
    task_configs=[RegressionTaskConfig(name="property", dims=[256, 64, 1])],
    encoder_config=TransformerEncoderConfig(
        input_dim=464, d_model=256, num_layers=4, nhead=8, dropout=0.1,
    ),
    enable_autoencoder=False,
)
```

For this API, head input width must match `d_model`. A kernel-regression task additionally needs its `t_sequences` in the forward call. A frozen Python model also needs explicit encoder evaluation mode during training; the CLI callback described below is not automatically installed by constructing this object.

| Component | Observed behavior | Research implication |
|---|---|---|
| Tokenization | Every scalar uses the same `Linear(1, d_model)` | Different columns have no independently learned value projection |
| Feature identity | Fixed sinusoidal encoding of column indices | Columns are distinguishable, but their numerical order introduces a prior without a general tabular interpretation |
| Attention | Full self-attention between descriptor columns | Valid, but potentially expensive for redundant scalar features |
| Transformer block | PyTorch defaults: Post-LayerNorm, ReLU, feed-forward width `4d` | This is a basic Transformer, not a tuned tabular encoder |
| Pooling | Learned CLS or mean pooling | Both propagate gradients to feature tokens; CLS is not inherently a defect |
| Shared output | `tanh(encoder(X))` for all heads | Preserve initially; inverse latent optimization depends on this interface |
| Training | Supervised multi-task losses, optional ordinary reconstruction | This is neither masked pretraining nor JEPA |
| Frozen transfer | CLI callback puts frozen encoder into evaluation mode | Current workflow already disables its dropout; do not attribute old results to a current frozen-dropout bug |
| CLI | TOML workflows construct `MLPEncoderConfig` unconditionally | Transformer is currently usable through the Python model API, not through an encoder switch in `fm pretrain` |
| Checkpoint inference | CLI rebuilds the encoder through the same MLP-only factory | Training support alone would be insufficient; prediction and inverse loading also need coverage |

Code map (paths relative to repository root):

- `src/foundation_model/models/components/foundation_encoder.py`: tokenizer, attention, positions, pooling.
- `src/foundation_model/models/model_config.py`: encoder configurations and optimization schema.
- `src/foundation_model/models/flexible_multi_task_model.py`: shared latent activation, task heads, initialization, reconstruction dimensions.
- `src/foundation_model/models/task_head/kernel_regression.py`: composition/coordinate-dependent function head.
- `src/foundation_model/workflows/_sections.py`, `_engine.py`, `recording.py`: CLI config, model construction, checkpoint restoration.
- `src/foundation_model/workflows/finetune.py`: frozen encoder evaluation callback.
- `src/foundation_model/workflows/task_catalog.py`, `data/composition_sources.py`: descriptors, composition identities, missing targets and splits.
- `src/foundation_model/utils/kmd_plus.py`: differentiable KMD mapping and feature ordering.
- `src/foundation_model/models/inverse_design/`: composition and latent optimization constraints.

### Implementation details that matter for a new comparison

Local instantiation confirmed that adjacent Transformer layers start with identical packed QKV weights, because PyTorch clones the initial layer. The model subsequently reinitializes `nn.Linear` modules, so their feed-forward and output-projection weights differ; packed `in_proj_weight` is not an `nn.Linear`. This is a partial initialization inconsistency, not proof of identical layers throughout training or proof of poor accuracy. Give new architectures explicit, independent initialization and record it. Preserve the existing behavior in a legacy reference arm.

The automatic reconstruction decoder also changes with the encoder: MLP `[464, 256, 384]` creates decoder dimensions `[384, 256, 464]`, whereas a Transformer with latent width 384 creates `[384, 464]`. Therefore, swapping the encoder currently also changes the reconstruction task's capacity. Use the same decoder, weight and target scaling in the controlled comparison. Do not infer a purely linear decoder from its dimensions: the shared `LinearLayer` currently replaces `activation=None` with LeakyReLU.

A valid-looking odd width (`d_model=9`, `nhead=3`) fails in sinusoidal position construction. This is a reproducible boundary defect, but it does not explain historical runs at width 256. No production fix is included in this investigation.

At equal latent width 384, local parameter counts are **219,008** for encoder `[464,256,384]` and **7,099,776** for the current four-layer Transformer (`d=384`, eight heads). Heads and decoder are excluded. Thus a comparison at equal output dimension is not a comparison at equal capacity.

## 2. What the historical experiment establishes

Explicit external inputs to this investigation:

- `artifacts/polymers_dynamic_tasks/`
- `artifacts/polymers_dynamic_tasks_transformer/`
- `experiments/rikyu_hparam_tuning_v2/` for the later MLP recipe, replay experience and corrected Materials Project dataset.

The early Transformer experiment used **190 polymer descriptors**, not today's 464-dimensional materials KMD. Both fine-tuning runs froze the encoder. The saved prediction tables contain 436 rows: four properties × 109 test samples. Joining on `(task, sample_index)` gives identical actual targets in all 436 pairs. These tables lack chemical identifiers, so this verifies the stored row/target alignment, not complete split provenance.

Define the relative RMSE change as

\[
\Delta_q = 100\left(\frac{\operatorname{RMSE}_{q,\mathrm{Transformer}}}
{\operatorname{RMSE}_{q,\mathrm{MLP}}}-1\right).
\]

Negative values favor Transformer. Errors below were recomputed from the saved CSV predictions; units follow the original property tables and have not been independently audited here.

| Polymer property | MLP RMSE | Transformer RMSE | Relative change |
|---|---:|---:|---:|
| Cp | 111.1563 | 124.9418 | +12.40% |
| Rg | 7.8577 | 7.6994 | −2.01% |
| Density | 0.0252845 | 0.0262221 | +3.71% |
| Linear expansion | 3.13605e−5 | 2.97196e−5 | −5.23% |

The result is mixed, not uniformly negative. It supports “no consistent advantage in this experiment,” not “Transformer cannot help.”

Important confounders in the saved hyperparameters:

- MLP latent width 128 versus Transformer width 256, hence different downstream head sizes.
- Encoder learning rates 0.05 versus 0.0005, a factor of 100.
- Transformer: four layers, eight heads, dropout 0.1. No comparable architecture-specific search is established by these artifacts.
- Different training durations/stopping points; MLP also has multiple pretraining log versions. Exact final-checkpoint ancestry needs reconstruction before making convergence comparisons.
- This is one saved comparison, not a paired multi-seed study. TensorBoard contains no learning-rate tags. Later repository scheduler fixes do not by themselves prove which scheduler behavior generated these old runs.

## 3. Why the previous approach may have shown little gain

These are hypotheses to test, ordered by practical relevance; they are not retrospectively proven causes.

1. **Tokenization was weakly adapted to tabular data.** Current tokens are `z_j = x_j w + b + p_j`, with shared `w,b` and a fixed index code `p_j`. A feature tokenizer instead learns `z_j = x_j w_j + b_j`. Both retain feature identity, but only the latter learns a feature-specific direction and offset directly. Adjacent columns are not generally adjacent objects in a physical sequence.
2. **Extra capacity may have been unnecessary.** These are composition/descriptor-to-property tasks, and MLPs already model feature interactions. Attention offers a different inductive bias, not information unavailable to an MLP. More GPU time does not add independent compositions or crystal structure, processing history, defects, or other unobserved variables.
3. **The comparison mixed architecture and optimization.** Width, learning rate, stopping and head dimensions differed. A larger network may underfit through optimization or overfit through variance; training and validation curves are needed to distinguish them.
4. **The task head or data may dominate the error.** Encoder improvement cannot be assumed to cure temperature-grid artifacts, limited kernel capacity, inconsistent target definitions, or composition-only ambiguity. Keep heads fixed first, then inspect head-limited cases separately.
5. **The shared representation can have competing objectives.** Multi-task/replay gradients and reconstruction may favor different features. Monitor per-task validation, forgetting and gradient/activation diagnostics before interpreting an aggregate score.

The prior v2 campaign's scheduler and data-quality findings show why a tuned current MLP must be rerun as the baseline. They do not establish the cause of the earlier polymer result.

## 4. What recent methods suggest

| Evidence | Relevant lesson | Limit of transfer to this project |
|---|---|---|
| [Laya, official repository](https://github.com/NandhaKishorM/laya) | ModernBERT/mmBERT representations feed specialized decision heads | Text pretraining supplies knowledge; its weights are not directly meaningful for KMD columns |
| [ModernBERT, 2024](https://arxiv.org/abs/2412.13663) | Modern encoder implementations, Pre-LN and optimized attention are useful engineering references | Text order, RoPE and long-context/local attention are not automatically appropriate for a short set of material features |
| [FT-Transformer, NeurIPS 2021](https://arxiv.org/abs/2106.11959) | Learn feature-specific numerical tokens and compare under a common tuning protocol | It is an established baseline, not evidence that every tabular problem benefits from attention |
| [Numerical embeddings, NeurIPS 2022](https://github.com/yandex-research/rtdl-num-embeddings) | Numerical representation itself can improve both MLPs and Transformers | KMD already performs a radial-basis expansion; additional embeddings need a control |
| [TabM, ICLR 2025](https://arxiv.org/abs/2410.24210) | Strong MLP-based models remain competitive with attention on tabular benchmarks | Published benchmark rankings are not rankings on our materials tasks |
| [TabPFN-2.5, 2025/2026](https://arxiv.org/abs/2511.08667) | In-context prediction over a labeled support set is another promising use of Transformers | It changes the learning problem and architecture more substantially than replacing our feature encoder |
| [CrabNet, 2021](https://www.nature.com/articles/s41524-021-00545-1) | Element identities and fractions provide physically meaningful tokens | Direct element tokens require a separate input/inverse design contract and do not automatically cover the polymer descriptor domain |
| [T-JEPA, ICLR 2025](https://arxiv.org/abs/2410.05016) | Predicting representations of masked features is a viable research direction | Collapse control, suitable masks and genuinely informative data are necessary; it is a training objective, not a replacement for attention |

The user's Jev link motivates the investigation. Public Jev marketing is not an architecture specification. Our model already emits numerical properties through specialized heads, so avoiding autoregressive text generation is not a new advantage available here.

A very recent [tabular JEPA preprint (2026-09-22)](https://arxiv.org/abs/2609.25541) reports worse aggregate performance and higher cost than its value-only control, with only one run per arm. This is preliminary evidence, not a decisive negative result; it reinforces the need for an objective ablation instead of assuming JEPA is beneficial.

### A KMD-specific caution about self-supervision

For the current element table, KMD can be written

\[
X=WK,\qquad K\in\mathbb R^{94\times464},\quad
W\mathbf 1=\mathbf 1,\quad W\geq0.
\]

Local NumPy diagnostics with the actual default basis (`method="1d"`, `n_grids=8`, `sigma="auto"`, `scale=True`) found:

- `rank(K)=94`; the affine rank of its element rows is 93.
- Largest/smallest singular values: 60.80375 / 0.42116; condition number about 144.37.
- Therefore this full basis is injective over the 94 element weights in exact arithmetic. It does not justify saying that KMD necessarily discards composition identity.
- For three specific random group subsets (NumPy RNG seed 42, sequential draws of 14, 29 and 44 out of 58 groups), each context basis still has row rank 94.
- Relative reconstruction residual `||K_context pinv(K_context) K − K||_F / ||K||_F`: 1.62e−13, 3.41e−15 and 5.26e−15 respectively.

Thus, in these masks, even a linear map can reconstruct the complete descriptor from the context in exact arithmetic. This is a property of the fixed basis, not a fit to experimental targets and not a measured downstream result. Learning that map from finite data is still an optimization problem, but low masked-reconstruction loss alone would provide little evidence of learned materials physics. Latent prediction may still regularize usefully; evaluate it through downstream transfer and against a simple masked-reconstruction baseline.

## 5. Proposed first candidate

Reshape the default KMD descriptor as `(batch, 58 properties, 8 grid values)`. Each property contributes one token:

\[
z_g = A_g x_{g,1:8}+e_g,\quad g=1,\ldots,58.
\]

`A_g` is a learned 8-to-d projection; `e_g` is a learned feature-identity embedding. The eight grid locations retain their identity as projection coordinates. No sinusoidal order is assigned between different properties.

```text
464 KMD values → 58 feature-group tokens + CLS
              → bidirectional Transformer
              → pooled vector → projection to shared latent width → tanh
              → existing task heads and common reconstruction decoder
```

Initial configuration, to be tuned rather than treated as optimal:

| Setting | Starting point |
|---|---|
| Transformer width / depth | 192 / 4 blocks |
| Attention heads | 6, full attention |
| Block | Pre-LayerNorm, residual attention and GELU feed-forward |
| Feed-forward hidden width | 768 |
| Dropout | 0.1; search includes 0 and 0.05 |
| Pooling | CLS, with mean pooling as a later ablation |
| Output | Linear projection to latent width 384, then existing tanh |
| Identity | Learned property embedding; no arbitrary sequence-position encoding |
| Initialization | Independent QKV/FFN initialization per block, explicitly specified |
| Heads / reconstruction | Identical to matched MLP controls |
| Objective | Existing supervised multi-task/replay recipe plus a common reconstruction objective |

At the same width/depth, attention pair counts decrease from `465²` to `59²`, about **62-fold**; token-wise work decreases about eightfold. These are operation-count ratios, not measured wall-clock speedups. This makes feature grouping useful for experimentation even if it does not improve accuracy. With `d≥8`, the per-group linear projection need not compress away the original eight values.

The grouped encoder is permutation invariant to reordering complete `(group values, group identity)` pairs. It must not be invariant to swapping feature values without their identities. A numerical check should enforce the former and distinguish the latter.

Scope is deliberate: this grouping applies to the default KMD layout, not arbitrary sets of 464 columns. Store group metadata/order and descriptor version. For the polymer descriptor data, use feature-specific scalar tokens unless a separately justified grouping is available.

An element-token encoder in the style of CrabNet is the next architectural branch worth considering. It should be compared on identical compositions, with explicit handling of trace fractions and padding. Composition inverse design must retain gradients when element support changes; naively dropping all zero-fraction elements would make that path problematic. A KMD-group encoder avoids this first-round interface change.

Task-specific cross-attention queries are another possible second-stage ablation if shared-CLS competition is demonstrated. They change the shared representation contract and should not be bundled with tokenization in the first experiment.

## 6. Evaluation protocol

### Data and leakage checks before training

Use the corrected `data/qc_ac_te_mp_dos_reformat_20260912.pd.parquet`, plus explicitly versioned magnetic and other task datasets. Do not silently fall back to the older mixed-functional energy targets.

Local metadata/column inspection found 49,034 rows and 49,014 distinct raw composition strings in this QC table. Split labels are 34,322 train / 7,355 validation / 7,357 test. Non-null entries are 33,166 for volume and formation energy, 11,722 for Seebeck and 4,971 for ZT. These are raw rows, not verified usable unique compositions: curve finiteness, canonicalization, duplicate handling and cross-dataset split precedence still need a saved inventory.

The rebuild script explicitly fits scalar normalization on **all non-null values**, not training rows only (`data/scripts/rebuild_mp_gga_20260912.py`, normalization section). Both architectures using those columns would share this preprocessing, but that is not a clean prospective evaluation. For the new benchmark, fit transforms from raw targets on training compositions only; for low-data transfer, fit on the selected target-training subset. Save the transform and inverse-transform predictions for physical-unit metrics. Audit curve preprocessing separately rather than assuming scalar behavior covers every dataset.

Create a global canonical composition split across all tasks. For strict unseen-composition evaluation, test compositions must be absent from source supervised training, replay and self-supervised pretraining. If a same-composition/new-property setting is also useful, report it separately as a transductive setting. Do not mix the two.

### Stage A — validity and pilot

First verify serialization, restoration, finite losses/gradients, missing-label masks, frozen-mode determinism, task addition/replay, predict and both inverse paths. All compared models must have identical heads, latent dimension and reconstruction decoder in the primary experiment.

Use seven pilot tasks: volume, formation energy, Seebeck, ZT, magnetization, magnetic moment, and material type classification. The first six reuse the v2 campaign's task selection, with a classification task added. Freeze the exact data release, split, task order, replay policy, stopping rule and metric definitions before tuning.

| Arm | Purpose |
|---|---|
| Tuned current MLP | Current deployment-quality reference, rerun on the new controlled data |
| Larger MLP | Approximately match the candidate's encoder parameter budget; tests whether capacity explains gains |
| Current scalar Transformer | Reevaluate the existing representation with fair optimization and matched heads |
| Feature-token Transformer | FT-Transformer-style per-scalar learned projections; tests better token identity |
| KMD-group Transformer | Proposed representation: one token per eight-bin property group |
| KMD-group MLP control | Same group tokenizer, per-token feed-forward layers and mean pooling, without attention; tests whether tokenization alone helps |

Use the same Pre-LN/GELU attention backbone for the feature-token and grouped variants so their direct comparison isolates tokenization. This is **FT-Transformer-style**, not an exact reproduction of every activation/normalization choice in the original implementation. Keep an exact published FT-Transformer configuration as an optional reference if results depend on this distinction.

For the legacy arm, matching the output dimension may require a documented final projection; report that adapter separately from an exact historical replay. Do not call a run with changed heads/decoder an exact historical reproduction.

Suggested initial search budget: eight validation-selected configurations × two paired tuning seeds × six arms = **96 pilot training trajectories**. Each trajectory introduces seven tasks: 672 task-introduction stages, plus any common consolidation pass. These are not 96 single-task fits. This is a proposed count, not a submitted campaign or a time estimate.

If grouping wins, add a random-grouping control with the same 58 groups of eight values and the same parameter count, using several preregistered permutations. This separates the benefit of meaningful property groups from the benefit of fewer tokens or a different projection size. Also distinguish backbone normalization from tokenization if the feature-token arm wins: Pre-LN and feature-specific projection should not be credited to each other without an ablation. Inspect sensitivity to feature magnitudes and the initial attention normalization placement; copying a tabular tokenizer while changing the original FT-Transformer's normalization conventions is not an exact baseline reproduction.

Search architecture-appropriate learning rates and regularization with the same trial budget. Suggested Transformer ranges: width `{128,192,256}`, depth `{2,4,6}`, encoder LR `{1e−4,3e−4,1e−3}`, weight decay `{1e−5,1e−3,1e−2}`; use a sampled design, not the full Cartesian product. Include the tuned v2 MLP recipe in the MLP search. Keep batch size and the existing corrected scheduler shared initially; examine warmup/cosine as a separate optimizer ablation if optimization diagnostics warrant it. Do not present warmup/cosine as already supported by the current CLI.

Two complementary comparisons are needed: a matched-latent/head/decoder study to identify mechanisms, and a best-tuned-per-family study to decide deployment. Report parameter counts and actual compute for both. Equal trial counts do not imply equal GPU-hours.

### Stage B — broad multi-task and transfer confirmation

Take the strongest MLP and up to two Transformer variants into five **new paired seeds** on the audited task catalogue, covering scalar regression, classification and functions. Five seeds × three architectures = at most 15 complete pretraining trajectories per chosen protocol. Separate this from per-target transfer fits in budget reporting. Use identical seed-specific task orders across architectures and inspect forgetting after each addition.

Evaluate transfer to genuinely held-out target tasks/domains, including low-data learning curves. For each target, remove its labels from source pretraining; also identify algebraic/near-duplicate target proxies such as closely related energy definitions or unit-rescaled versions. Preserve strict composition exclusion where that is the evaluated question.

For each architecture and target data budget, compare:

1. Independent training on the target data.
2. Source pretraining followed by frozen-encoder head training.
3. Source pretraining followed by full fine-tuning, with encoder/head learning rates tuned separately.

Use nested target-training subsets (e.g. 1%, 5%, 10%, 25%, 50%, 100%, subject to valid minimum counts) and the same held-out test set. Pair subset draws and seeds across architectures. For the eight-compound resistivity data, use its existing 1-to-7 training-compound protocol instead of percentages. Each source-model index also defines the corresponding scratch initialization seed; scratch does not reuse a pretrained checkpoint. Report that scratch replication is a new statistical design choice, not a reinterpretation of the earlier AGIS campaign's run counts.

Rerun the historical polymer benchmark as a separate domain with scalar feature tokens. Do not mix its errors into a materials-KMD aggregate or force KMD grouping onto polymer columns.

### Metrics and decision rule

- Regression: physical-unit RMSE/MAE and R²; inspect both training and held-out errors.
- Functions: average within each composition's curve before averaging compositions, so dense temperature sampling does not dominate. Report scientifically relevant subranges separately.
- Classification: macro-F1 and balanced accuracy; do not average these numerically with R².
- For positive error metrics define `relative change = 100 × (candidate error / baseline error − 1)`. Give per-task values, the distribution and the task-macro summary. Handle zero/near-zero baseline errors explicitly rather than allowing unstable ratios.
- Report paired seed/subset differences and uncertainty intervals. Resample independent compositions, not individual temperature points; preserve task/fold dependence. Repeated evaluations of the same eight compounds do not become hundreds of independent materials.
- Preserve heterogeneous paired effects; an average win must not hide a large systematic loss on an important task.
- Record total GPU-hours, examples/second, peak memory and utilization. Seed variation is not a calibrated predictive interval.

Select using validation only, then open the test results once for confirmation. A default replacement needs a reproducible practical gain under the registered protocol; low-data-only gains can justify a specialized transfer option. Predeclare a practical error reduction threshold with the scientific use case before launching confirmation. If intervals include both useful improvement and meaningful harm, report the result as unresolved and expand seeds only for that comparison.

### Stage C — objective and representation extensions

Only after Stage B, compare the winning backbone with: existing reconstruction, masked group reconstruction, and a T-JEPA-style objective. Match available compositions and the downstream protocol. Mask entire property groups, track representation variance/effective rank and downstream probes, and test collapse. If claiming benefit from additional unlabeled data, use genuinely additional, deduplicated training compositions; synthetic mixtures have deterministic descriptors but no newly measured physics.

Other conditional experiments: element tokens, task-specific queries, and independent head-capacity ablations. These should answer distinct hypotheses rather than turn the first run into a bundle of unrelated changes.

## 7. RIKYU execution preparation

A read-only Slurm query on 2026-10-02 succeeded. The `gpu` partition was up with a four-day limit. This verifies connectivity and partition availability, not our project allocation, queue start time or a reserved GPU budget. The repository's Phase 2 end-of-September note is not sufficient to infer that access stopped.

The package currently runs one GPU per process. Use multiple GPUs for independent configurations and seeds. Before sizing a fleet, benchmark representative MLP, scalar-token and grouped-token runs on allocated compute nodes. Query `sacct` utilization, and calibrate packing factors against an unpacked run. The previous MLP result of eight processes per GPU must not be transferred to attention workloads without measurement. Keep within the enforced 32 CPU cores per allocated GPU.

Estimate campaign time only after this measurement:

\[
T_{\rm compute}\approx
\frac{1}{G}\sum_a\frac{N_a\,t_{a,p_a}}{p_a}
\]

This is an ideal load-balanced estimate for a shared fleet: `N_a` is the trajectory count, `t_{a,p_a}` its measured wall time under packing `p_a`, and `G` the GPU count. For separate concurrent pools, estimate each pool with its assigned GPU count and take the longest finishing time. Add queue, scheduling imbalance and workflow overhead; do not extrapolate old Mac timings or nominal GB200 FLOPS. Record software/container version, dataset hashes, launch commit and resolved config for every run.

Implementation sequence to make the proposal executable: (1) explicit encoder configuration and consistent construction/checkpoint loading, (2) tokenizers/backbones and common decoder controls with focused tests, (3) data/split manifests and pilot runner, then calibration. These are planned changes, not implemented features. Follow the repository's review sequence if this becomes a multi-PR implementation.

## 8. Investigation log and limitations

2026-10-02: inspected encoder, heads, config/workflow construction, data/catalog/split code, replay/inverse interfaces and historical experiment records; checked primary literature and official implementations; recomputed old prediction errors; instantiated current encoders; measured the KMD basis rank and context reconstruction; inspected current parquet metadata and scalar normalization code; ran a read-only RIKYU partition query.

Numerical diagnostics used the existing uv environment without dependency changes. Local environment warnings about the recorded Python patch version and restricted font/CPU-cache probing did not prevent these diagnostics from completing. No GPU benchmark, new training, accuracy claim, production implementation, PR or remote submission is part of this record.

Validation: both Python examples in this document were executed successfully; the historical RMSE changes and basis-rank results reproduced. The new document passed a whitespace/diff check. Production tests were not run because no production code was changed.

The most important unresolved empirical question is whether feature grouping plus attention improves cross-property transfer beyond the same tokenizer with an MLP and beyond a carefully tuned conventional MLP. The proposed controls make that question testable.

### Reproducing the key local numerical diagnostics

Run the following with the repository's `uv run python` environment; it uses existing local inputs and does not train a model:

```python
import numpy as np
import pandas as pd
from foundation_model.utils.kmd_plus import KMD, element_features

K = KMD(element_features.values, method="1d", n_grids=8,
        sigma="auto", scale=True).transform(np.eye(94))
print(K.shape, np.linalg.matrix_rank(K), np.linalg.cond(K))
rng = np.random.default_rng(42)
for n_groups in [14, 29, 44]:
    groups = rng.choice(58, n_groups, replace=False)
    columns = (groups[:, None] * 8 + np.arange(8)).ravel()
    context = K[:, columns]
    residual = np.linalg.norm(context @ np.linalg.pinv(context) @ K - K) / np.linalg.norm(K)
    print(n_groups, np.linalg.matrix_rank(context), residual)

a = pd.read_csv("artifacts/polymers_dynamic_tasks/fine_tune_test/predictions.csv")
b = pd.read_csv("artifacts/polymers_dynamic_tasks_transformer/fine_tune_test/predictions.csv")
pairs = a.merge(b, on=["task", "sample_index"], suffixes=("_mlp", "_tf"),
                validate="one_to_one")
assert len(pairs) == len(a) == len(b) == 436
assert np.array_equal(pairs.actual_mlp, pairs.actual_tf)
for task, frame in pairs.groupby("task"):
    mlp_rmse = np.sqrt(np.mean((frame.predicted_mlp - frame.actual_mlp) ** 2))
    tf_rmse = np.sqrt(np.mean((frame.predicted_tf - frame.actual_tf) ** 2))
    print(task, mlp_rmse, tf_rmse, 100 * (tf_rmse / mlp_rmse - 1))
```


## Frozen readout diagnosis (2026-10-03)

Question: do similar aggregate errors hide encoder differences, or does the common downstream
readout limit their use? This follow-up reuses the **completed main campaign** source checkpoints
at k=1/3/7 for tuned MLP and grouped mean (three seeds). Inputs are explicitly supplied through
`--source-root`; that directory must contain the original `scientific/<arm>_s<seed>` lanes.
Checkpoint, manifest, source campaign revision, protocol and image identities are checked.

`configs/head_probe.toml` registers a separate cached-feature protocol on dielectric only:

- 2 encoders × 3 stages × 3 seeds × 2 target training sizes × 3 readouts = 108 selected fits:
  ridge, MLP 128→64 with identity output, MLP 512→256 with identity output.
- 12 additional k=7 fits reproduce the current LeakyReLU output (the activation comparison has
  24 fits total, of which 12 identity-output fits are shared above).
- 36 pre-tanh ridge probes and 6 raw-descriptor ridge references. Total: **162 selected fits**.
- The 84 neural configurations each search three LRs (252 optimization trials); 78 ridge
  configurations each search seven alphas (546 closed-form fits). Selection uses validation only.
  Also report the predetermined LR 0.002 activation comparison, without choosing LR on test.

The image-installed encoder and its parameters/BatchNorm buffers remain frozen. Both neural
activation arms use identical initial weights and minibatch permutations, AdamW (weight decay
0.001, epsilon 1e-6), batch 128 with drop-last, up to 250 epochs and patience 20. There is no
reconstruction term. Scheduler and best-checkpoint selection monitor target validation MSE;
this intentionally differs from the old workflow and is not pooled with its 756 fits. Preserve
best and last head weights, all validation histories, selected test predictions, last-epoch
predictions, and validation representation statistics. Neural probes use the original post-tanh
features; ridge feature standardization is fitted on training compositions only. Scalers and
train/validation/test memberships come unchanged from the registered data manifest.

This does **not** patch the shared `None` activation semantics: the control uses the installed
`LinearBlock(output_active=None)` and the alternative explicitly uses `nn.Identity()`. Package
0.5.0 and its verified official ARM image are therefore unchanged. A package-level repair is a
separate compatibility/version/image decision. Current work tests sensitivity, not that the
observed activation necessarily caused architectural similarity.

Execution: after review and squash merge, stage scripts/configs only. Launch
`scripts/head_probe.sbatch` with runtime-only `FM_WORKSPACE`, `FM_IMAGE`, `FM_DATA_DIR`,
`FM_SOURCE_WORKSPACE`, `FM_SOURCE_ROOT`, `FM_OUTPUT_ROOT`, and `FM_BENCHMARK_REVISION`.
Case indices 0–8 are MLP (seed-major, then k=1/3/7), 9–17 grouped mean. `SMOKE=1`
uses 256 rows per split and two epochs, in a distinct output root. Calibrate PACK on completed
unpacked cases and compare Slurm accounting before launching the remaining cases. Only
same-identity completed trials may be skipped on resume; an interrupted head trial restarts.
Results and job provenance stay in ignored `artifacts/head_probe` and `results/head_probe`.

Interpretation: report paired encoder-by-readout interactions, physical RMSE/MAE, prediction
disagreement and validation CKA/effective rank/saturation. Three-seed SD is not predictive
uncertainty. Source count still changes task content and cumulative compute. The existing test
set has already been inspected; this is explanatory follow-up, not an untouched confirmatory
benchmark. Token-level readout and new full-fine-tuning/pretraining jobs remain conditional on
this diagnosis. No source training is repeated here.

Execution log (2026-10-03): PR [#74](https://github.com/TsumiNa/foundation_model/pull/74)
was reviewed, fixed and squash-merged as `e9223f0bf98f3693d1c0b300178f494606151f36`
before official-image deployment. Smoke array 162107 passed both encoders. Array 162113 completed
six unpacked full-budget cases; isolated repeat array 162125 calibrated PACK=3 and reproduced
their selected metrics exactly. Array 162133 completed the remaining twelve cases with PACK=3.
All 18 formal lanes / 162 selected fits passed the finite-prediction and frozen-encoder audits;
all 252 neural histories are finite (21–99 actual epochs). The 14 allocated array tasks all
completed with exit 0. Total accounting, including smoke/calibration: 0.140 GPU-hours. PACK=3
throughput was approximately 2.6×; recorded packed utilization was 82–91%, GPU memory ≤3,444 MiB.
No source training was repeated; source and probe checkpoints remain on RIKYU.

Outcome: this frozen dielectric screen does not support insufficient FC-head width as the main
explanation for similar encoder RMSE. At k=7/full training, identity-output 128→64 heads give
RMSE 36.723 (MLP) and 36.712 (Transformer), despite paired prediction disagreement D=0.462
(D is prediction-difference RMSE divided by paired MLP reference-error RMSE). Wider heads do not
reveal a consistent Transformer advantage. Post-tanh ridge gives Transformer RMSE 36.048 versus
MLP 36.775, but MAE/validation and low-data rankings do not agree; this is not a validated winner.
The fixed-LR output-activation effect changes sign across settings. Pre-tanh ridge does not
improve full-training test error. Representation geometry differs and effective rank grows with
source-task count, which remains confounded with task content and compute.

Best-versus-last head checkpoints show a stronger low-data sensitivity: mean test RMSE increases
by 3.19 (MLP) / 1.79 (Transformer) when using last instead of best target-validation weights,
averaged over the 21 selected neural configurations per family at 10% training size. The next
supported confirmation is matched end-to-end checkpoint-selection control, not automatic head
expansion. No token-level or end-to-end confirmation fleet was launched: no readout change won
consistently on validation. This completes the bounded A/B investigation, with three seeds and
one previously inspected target/test set; it does not establish general architectural equivalence.

Full local/remote report and figures: `results/head_probe/REPORT_20261003.md` and
`results/head_probe/readout_scaling_f{010,100}.png`. Audit: `final_validation_audit.json` in that
result directory; Slurm evidence and cost: `artifacts/head_probe_control/`. Generated results,
predictions and caches are synchronized outside Git, separately from smoke and calibration.

Reproduce the final report after synchronizing `artifacts/head_probe` (including feature NPZs,
trial histories and selected-last predictions; model-weight files are not needed for analysis)
and `artifacts/head_probe_control/sacct.txt`:

```bash
EXP=experiments/transformer_encoder_20261002
uv run python "$EXP/analysis/collect_head_probe.py" \
  --root "$EXP/artifacts/head_probe" --config "$EXP/configs/head_probe.toml" \
  --protocol "$EXP/configs/protocol.toml" --output "$EXP/results/head_probe"
uv run python "$EXP/analysis/summarize_head_probe.py" \
  --root "$EXP/artifacts/head_probe" --config "$EXP/configs/head_probe.toml" \
  --protocol "$EXP/configs/protocol.toml" --output "$EXP/results/head_probe" \
  --accounting "$EXP/artifacts/head_probe_control/sacct.txt"
```

The second command requires a complete audited matrix, verifies all optimization histories and
best/last prediction alignment, and regenerates `REPORT_20261003.md`, `final_validation_audit.json`,
`gpu_cost.json`, checkpoint-sensitivity/tail tables, and the final presentation-size figures.
It does not modify or retrain the deployed experiment.

Generate direct composition-level comparisons from the audited paired prediction table:

```bash
uv run python experiments/transformer_encoder_20261002/analysis/plot_composition_comparison.py \
  --input experiments/transformer_encoder_20261002/results/head_probe/paired_composition_predictions.csv \
  --output experiments/transformer_encoder_20261002/results/head_probe/direct_comparison
```

The six-page PDF and PNGs compare frozen tuned MLP and grouped-mean Transformer encoders after
seven source tasks with the identity-output 128→64 head. Both training sizes use the same twelve
examples, chosen at evenly spaced reference-value ranks without using prediction errors.
Bars show seed-mean predictions; individual-seed markers show training variability. Parity plots
include full-range and zoomed views plus a direct model-versus-model panel. CSV exports retain
all compositions and seeds, and provenance records the input/script hashes. These visual means
are distinct from the study's mean-per-seed RMSE; no additional training is performed.
