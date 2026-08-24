*[中文版](README.md)*

# Inverse Design — Teaching Material

A two-lesson, self-contained course on the **inverse design of quasicrystal (QC) alloys**.
Data, model and optimiser all run inside this folder; nothing outside it is required.

> **A note on language:** the narration inside both notebooks is written in **Chinese**.
> This README, all code, and every figure label are in English. See [Language](#language) below.

## How to use it

```bash
uv sync --frozen --all-groups        # once, from the project root, to install dependencies
uv run jupyter lab                   # then open notebooks/teaching_inverse_design/
```

Run them in order:

| Notebook | Content | Runtime |
|---|---|---|
| `01_multitask_model.ipynb` | data → KMD descriptors → multi-task model → training → evaluation | ~1–2 min |
| `02_inverse_design.ipynb` | how the KMD composition-path optimiser works, its parameters, three design scenarios, analysis | ~1 min |

**Run 01 first** — it writes the trained model into `outputs/`, which is where 02 reads it from.

Both `.ipynb` files are committed **with their outputs**, so you can read them as lecture notes
before running a single cell.

## Layout

```
teaching_inverse_design/
├── README.md                              ← Chinese version
├── README.en.md                           ← this file
├── data/
│   └── qc_inverse_design_teaching.parquet ← the only data file (585 KB, 29,802 compositions)
├── prepare_data.py                        ← provenance record for that parquet (never run in class)
├── 01_multitask_model.ipynb
├── 02_inverse_design.ipynb
└── outputs/                               ← written by 01, read by 02; gitignored
    ├── multitask_model.ckpt               ← the trained model
    ├── model_meta.json                    ← architecture parameters + test-split metrics
    ├── resolved_split.parquet             ← composition-level train/val/test split
    ├── scenario*__seed_to_optimized.parquet   ← per-seed design results for each scenario
    └── scenario*__trajectory.npz              ← optimisation trajectory for each scenario
```

## The data

One parquet file; one row = one chemical composition, keyed by `composition`.
The four property columns (`formation_energy` / `magnetization` / `tc` / `klat`) are
**z-scored values, not physical units**; `material_type` is a 3-class label
(`0=AC`, `1=QC`, `2=others`).

Many cells are NaN — the properties come from different databases whose compositions barely
overlap, and the model masks missing targets per task rather than dropping the row.

The table is a merge of four property databases (QC/AC materials, superconductors, magnetics,
thermal conductivity). The full merge / canonicalisation / standardisation procedure is in
`prepare_data.py`, fully commented. That script is **only a provenance record**: the course never
runs it and does not need the four upstream databases — everything the course uses is in this
folder.

## What the two lessons do and do not cover

**Covered:**
- How to build and evaluate a forward model (composition → property), and **how that evaluation
  determines which inverse-design targets you can trust**
- KMD descriptors — in particular why `x = w @ K` being **linear** is the precondition for
  inverse design
- The KMD composition-path optimiser end to end:
  `logits → softmax → w → x → model → loss`, differentiable throughout
- What problem each important parameter solves, with a one-change-at-a-time comparison table
- Three design scenarios, start to finish, with analysis

**Not covered:**
- Data cleaning and assembly (done in advance)
- Multi-task learning as a subject (five tasks are trained together only because the three
  scenarios need those five properties)
- The latent-path optimiser (only the KMD composition path is taught; §2.3 of notebook 02
  explains why)

## Language

The teaching prose in both notebooks is **Chinese**; all code, output and figure labels are
**English**.

Figure labels are English on purpose rather than by omission: matplotlib's default font carries no
CJK glyphs, and substituting a font that does is not portable across machines. So axes and titles
stay English and the explanation lives in the surrounding markdown.
