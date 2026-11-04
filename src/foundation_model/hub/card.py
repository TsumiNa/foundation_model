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

from .bundle import BundleConfig, Manifest

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
        "library_name: foundation-model",
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
            + "Models you train from scratch with the foundation-model code and your own data are not covered."
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
        lines.append(f"  {_yaml_str(f'I accept the {config.license} licence terms')}: checkbox")
    lines.append("---")
    return lines


def _heads_table(config: BundleConfig) -> list[str]:
    rows = ["| Head | Kind | Output | Unit |", "|---|---|---|---|"]
    for h in config.heads:
        if h.kind == "classification":
            assert h.classes is not None
            output, unit = f"class: {' / '.join(h.classes.classes)}", "—"
        elif h.output_scale == "shape":
            output, unit = "curve shape (each curve divided by its max abs value)", "dimensionless"
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
    n = len(manifest.members)
    name = title or (repo_id.split("/")[-1] if repo_id else "foundation-model bundle")
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
        f"- Bundle schema {config.schema_version}, written by foundation-model {config.fm_version}.",
        f"- Weights licence: **{config.license}**.",
        f"- {n} member{'s' if n != 1 else ''}; default `{manifest.default}`."
        + (" Members differ only in the order in which pretraining met the tasks." if n > 1 else ""),
        "",
        "## Heads",
        "",
        *_heads_table(config),
        "",
        "Outputs are in the listed units. Shape heads predict a normalised curve shape, not an absolute value.",
        "",
        "## Use",
        "",
        "```bash",
        "pip install 'foundation-model[hub]'",
        "hf auth login   # gated repository: request access on this page first",
        "```",
        "",
        "```python",
        "from foundation_model.hub import load_pretrained",
        "",
        f'model = load_pretrained("{repo}")          # default member; "{repo}#m001" for another',
        f'model.predict(["Fe2O3", "NaCl"], tasks={scalar!r})',
    ]
    if curve is not None:
        lines.append(f'model.predict_curve(["Bi2Te3"], "{curve.name}")   # on the default {curve.t_name or "t"} grid')
    lines += [
        "```",
        "",
        "Command line:",
        "",
        "```bash",
        f'fm predict  --pretrained {repo} --compositions "Fe2O3,NaCl" --output-dir pred/',
        f"fm finetune --pretrained {repo} --config my_tasks.toml   # your tasks only, physical units",
        f"fm inverse  --pretrained {repo} --config design.toml     # targets in physical units",
        "```",
        "",
        "Fine-tuning adds new heads or adapts existing ones on your own data. No pretraining data are released, "
        "so post-training that replays them (rehearsal) is not possible.",
        "",
        "## Limitations",
        "",
        "- Input is composition only. Polymorphs and structures of one composition get the same prediction.",
        "- The KMD descriptor uses atomic fractions, so total formula-unit scale is invisible to it. Quantities "
        "that depend on it, such as cell volume or moment per formula unit, carry that caveat.",
        "- Curve heads are trained on the listed `t` ranges. Outside them, `predict_curve` warns that it is "
        "extrapolating.",
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
        "Models trained **from scratch** with the foundation-model code (Apache-2.0) on your own data are not "
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
