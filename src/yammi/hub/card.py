# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

"""The model card (``README.md``) of a bundle, generated from ``config.json`` and ``manifest.json``.

The YAML header carries what the Hugging Face Hub reads: the licence, ``library_name`` and tags, and,
for a gated repository, the access form. The body records:

* the heads, with their units and output scale;
* the members;
* how to load, predict, fine-tune and run inverse design;
* the limits of the model, the licence terms and the data citations.

Extra Markdown, for instance the citation list from ``docs/licensing/data_sources.md``, is appended
verbatim.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from .bundle import BundleConfig, Manifest, yammi_version

__all__ = ["HF_LICENSES", "render_card"]

# Licence identifiers the Hub recognises; anything else is published as ``license: other``.
HF_LICENSES = {
    "apache-2.0",
    "mit",
    "bsd-3-clause",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "cc-by-nc-4.0",
    "cc-by-nc-sa-4.0",
    "cc-by-nc-nd-4.0",
    "cc0-1.0",
}
_NON_COMMERCIAL = ("-nc-", "-nc")


def _is_non_commercial(license_id: str) -> bool:
    return any(tag in license_id.lower() for tag in _NON_COMMERCIAL)


def _cc_terms(license_id: str) -> str | None:
    """The derivative terms of a Creative Commons licence id, or ``None`` for any other licence.

    NC: non-commercial use only. SA: adaptations must carry the same licence. ND: adaptations may not be
    shared. A non-CC licence gets no paraphrase, only its name: the card must not invent terms for it.
    """
    lic = license_id.lower()
    if not lic.startswith("cc-by"):
        return None
    parts = set(lic.removesuffix("-4.0").split("-"))
    use = "for non-commercial purposes only" if "nc" in parts else "for any purpose, including commercially"
    terms = f"may be used {use}, with attribution"
    if "nd" in parts:
        return f"{terms}; adapted models (for example fine-tuned ones) may not be shared"
    if "sa" in parts:
        return f"{terms}; models derived from them (for example by fine-tuning) must carry the same licence"
    if "nc" in parts:
        return f"{terms}; models derived from them stay non-commercial"
    return terms


def _image() -> str:
    """The container image of the yammi release writing the card (multi-arch: amd64 and arm64)."""
    version = yammi_version()
    return f"ghcr.io/tsumina/yammi:{version if version != 'unknown' else 'latest'}"


def _writer(version: str) -> str:
    """Name the release that wrote a bundle: before 2.0 the package was published as foundation-model."""
    major = version.split(".", 1)[0]
    return f"foundation-model {version} (now yammi)" if major.isdigit() and int(major) < 2 else f"yammi {version}"


def _yaml_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)  # a JSON string is a valid YAML scalar


def _front_matter(config: BundleConfig, *, gated: bool) -> list[str]:
    lic = config.license.lower()
    lines = ["---"]
    if lic in HF_LICENSES:
        lines.append(f"license: {lic}")
    else:
        lines += ["license: other", f"license_name: {_yaml_str(config.license)}"]
    lines += [
        "library_name: yammi",
        "tags:",
        "  - materials-science",
        "  - chemical-composition",
        "  - multi-task",
        "  - kernel-mean-descriptor",
    ]
    if gated:
        nc = _is_non_commercial(lic)
        terms = _cc_terms(lic)
        prompt = (
            "Access is reviewed by the authors. "
            + (
                f"These weights are licensed under {config.license}: they {terms}. "
                if terms
                else f"These weights are licensed under {config.license}; see its terms. "
            )
            + "Models you train from scratch with the yammi code and your own data are not covered."
        )
        lines += [
            f"extra_gated_prompt: {_yaml_str(prompt)}",
            "extra_gated_fields:",
            "  Name: text",
            "  Affiliation: text",
            "  Intended use: text",
        ]
        if nc:
            label = "I will use these weights and any model derived from them for non-commercial purposes only"
            lines.append(f"  {_yaml_str(label)}: checkbox")
        # The Hub rejects metadata keys containing "." or "$", and licence ids such as cc-by-4.0 have one;
        # the prompt above names the licence, so the checkbox refers to it.
        lines.append(f"  {_yaml_str('I accept the licence terms stated above')}: checkbox")
    lines.append("---")
    return lines


_KIND_ORDER = {"regression": 0, "classification": 1, "kernel_regression": 2}
_MAX_LISTED_CLASSES = 12


def _heads_table(config: BundleConfig) -> list[str]:
    """One row per head, grouped by kind and sorted by name (the stored order is a member's task order)."""
    rows = ["| Head | Kind | Output | Unit |", "|---|---|---|---|"]
    for h in sorted(config.heads, key=lambda h: (_KIND_ORDER.get(h.kind, 3), h.name)):
        if h.kind == "classification":
            assert h.classes is not None
            names = h.classes.classes
            listed = " / ".join(names) if len(names) <= _MAX_LISTED_CLASSES else f"{len(names)} classes"
            output, unit = f"class: {listed}", "—"
        elif h.output_scale == "shape":
            output, unit = "curve shape (trained on curves each divided by its max abs value)", "dimensionless"
        else:
            output = "curve" if h.kind == "kernel_regression" else "value"
            unit = h.unit or "—"
        if h.kind == "kernel_regression" and h.t_range is not None:
            t_unit = f" {h.t_unit}" if h.t_unit else ""
            output += f" over {h.t_name or 't'} ∈ [{h.t_range[0]:g}, {h.t_range[1]:g}]{t_unit}"
        rows.append(f"| `{h.name}` | {h.kind.replace('_', ' ')} | {output} | {unit} |")
    return rows


def render_card(
    config: BundleConfig,
    manifest: Manifest,
    *,
    repo_id: str | None = None,
    title: str | None = None,
    gated: bool = False,
    extra: Sequence[str] = (),
) -> str:
    """The full ``README.md`` text for a bundle."""
    repo = repo_id or "<owner>/<repo>"
    image = _image()
    access = (
        "This is a gated repository: request access on this page, then log in once with `hf auth login`."
        if gated
        else "No login is needed for a public repository; for a private one, log in with `hf auth login`."
    )
    n = len(manifest.members)
    name = title or (repo_id.split("/")[-1] if repo_id else "yammi bundle")
    curve = next((h for h in config.heads if h.kind == "kernel_regression" and h.output_scale == "physical"), None)
    scalar = [h.name for h in config.heads if h.kind == "regression"][:2]
    lines = _front_matter(config, gated=gated)
    lines += [
        "",
        f"# {name}",
        "",
        "A multi-task model that predicts material properties from **chemical composition alone**. The input is "
        f"a kernel mean descriptor (KMD) of the composition ({config.descriptor.n_grids} grids over "
        f"{len(config.descriptor.elements)} elements). One shared encoder feeds {len(config.heads)} task heads.",
        "",
        f"- Bundle schema {config.schema_version}, written by {_writer(config.yammi_version)}.",
        f"- Weights licence: **{config.license}**.",
        f"- {n} member{'s' if n != 1 else ''}; default `{manifest.default}`."
        + (" Members differ only in the order in which pretraining met the tasks." if n > 1 else ""),
        "",
        "## Heads",
        "",
        *_heads_table(config),
        "",
        "Outputs are in the listed units. Shape heads predict on the per-curve max-abs scale they were trained on: "
        "a shape, not an absolute value.",
        "",
        "## Use",
        "",
        access,
        "",
        f"**Container (recommended; nothing to install).** `{image}` runs on x86-64 and on Grace",
        "(AArch64) NVIDIA machines alike; the pull picks the right variant. It contains yammi with CUDA 13",
        "and downloads the weights on first use.",
        "",
        "```bash",
        "# Docker: mount the Hugging Face cache (holds your token and the downloaded weights)",
        'docker run --rm --gpus all -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -v "$PWD:/workspace" \\',
        f'  {image} predict --pretrained {repo} --compositions "Fe2O3,NaCl" --output-dir pred/',
        "",
        "# Apptainer / Singularity (HPC): $HOME, and with it the token and cache, is bound automatically",
        f"apptainer pull yammi.sif docker://{image}",
        f'apptainer exec --nv yammi.sif yammi predict --pretrained {repo} --compositions "Fe2O3,NaCl" --output-dir pred/',
        "```",
        "",
        "**Python** (scripting, or developing on top of yammi). Not on PyPI yet: clone the code repository and",
        "install it with [uv](https://docs.astral.sh/uv/).",
        "",
        "```bash",
        "git clone https://github.com/TsumiNa/yammi.git",
        "cd yammi && uv sync --frozen --no-dev --extra hub     # the locked environment in .venv",
        "source .venv/bin/activate",
        "```",
        "",
        "```python",
        "from yammi.hub import load_pretrained, load_suite",
        "",
    ]
    if n > 1:
        lines += [
            f'suite = load_suite("{repo}")   # all {n} members, combined: mean of in-range members + ":std"',
            f'suite.predict(["Fe2O3", "NaCl"], tasks={scalar!r})',
        ]
        if curve is not None:
            lines.append(
                f'suite.predict_curve(["Bi2Te3"], "{curve.name}")   # on the default {curve.t_name or "t"} grid'
            )
        lines += ["", f'model = load_pretrained("{repo}#m001")   # one member (default: {manifest.default})']
    else:
        lines.append(f'model = load_pretrained("{repo}")')
    lines.append(f'model.predict(["Fe2O3", "NaCl"], tasks={scalar!r})')
    if curve is not None and n == 1:
        lines.append(f'model.predict_curve(["Bi2Te3"], "{curve.name}")   # on the default {curve.t_name or "t"} grid')
    lines += [
        "```",
        "",
        "Command line (the same subcommands inside the container):",
        "",
        "```bash",
        f'yammi predict  --pretrained {repo} --compositions "Fe2O3,NaCl" --output-dir pred/'
        + ("   # members combined; #m001 for one" if n > 1 else ""),
        f"yammi finetune --pretrained {repo} --config my_tasks.toml   # your tasks only, physical units",
        f"yammi inverse  --pretrained {repo} --config design.toml     # targets in physical units",
        "```",
        "",
        "Fine-tuning adds new heads or adapts existing ones on your own data. No pretraining data are released, "
        "so post-training that replays them (rehearsal) is not possible.",
        "",
        "**With an AI coding agent.** The repository ships an agent skill, `yammi-pretrained`, with these "
        "commands, the input and output conventions, and the pitfalls. Install it where your agent reads skills, "
        "for example for Claude Code:",
        "",
        "```bash",
        "git clone --depth 1 https://github.com/TsumiNa/yammi /tmp/yammi",
        "cp -r /tmp/yammi/.claude/skills/yammi-pretrained ~/.claude/skills/   # or <project>/.claude/skills/",
        "```",
        "",
        "Agents that read `.agents/skills/` (Codex and others) take the same folder from `.agents/skills/`. "
        "Without a skill, point the agent at `docs/pretrained_models.md` in the repository.",
        "",
        "## Limitations",
        "",
        "- Input is composition only. Polymorphs and structures of one composition get the same prediction.",
        "- The KMD descriptor uses atomic fractions, so total formula-unit scale is invisible to it. Quantities "
        "that depend on it, such as cell volume or moment per formula unit, carry that caveat.",
        "- Curve heads are trained on the listed `t` ranges. Outside them, `predict_curve` warns that it is "
        "extrapolating.",
        "- Some heads' target transforms cannot be inverted past the training range. A single member returns "
        'such a prediction as NaN with a `SaturationWarning` (`saturated="clip"` gives the most extreme '
        "training value instead); the combined prediction uses the members that stay in range.",
        "- Elements absent from the training data (for example noble gases in most compounds) get predictions "
        "without any warning; treat them as unreliable.",
        "- Regression heads are unbounded: a quantity that cannot be negative (band gap, moduli) can come out "
        "slightly below zero.",
        "",
        "## Licence",
        "",
    ]
    terms = _cc_terms(config.license)
    if terms:
        lines += [f"The weights are licensed under **{config.license}**: they {terms}.", ""]
    else:
        lines += [f"The weights are licensed under **{config.license}**; its own terms apply.", ""]
    lines += [
        "Models trained **from scratch** with the yammi code (Apache-2.0) on your own data are not "
        "covered by this licence.",
        "",
    ]
    if "derived_from" in config.provenance:
        src = config.provenance["derived_from"]
        lines += [
            "## Derived from",
            "",
            f"`{src.get('source')}` (revision `{src.get('revision')}`, member `{src.get('member')}`, "
            f"licence {src.get('license')}).",
            "",
        ]
    for block in extra:
        lines += [block.rstrip(), ""]
    return "\n".join(lines).rstrip() + "\n"


def read_extra(paths: Sequence[str | Path]) -> list[str]:
    return [Path(p).read_text(encoding="utf-8") for p in paths]
