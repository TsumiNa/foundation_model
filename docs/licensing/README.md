# Licensing of the code, the pretrained weights and the training data

Working notes for publishing the pretrained models. Not legal advice; the release terms should be reviewed by
the institute's intellectual-property office before the first public release.

| File | Contents |
|---|---|
| `data_sources.md` | licence of every training-data source, how it was verified, open checks, citations |
| `../../notebooks/teaching_inverse_design/data/README.md` | sources, changes and licence (CC BY 4.0) of the tracked teaching table |

## Decided

- **Hosting:** model code stays in this package. A Hugging Face model repository holds weights,
  `config.json` and the model card (decided 2026-10-30).
- **Access:** the weights are published as a **gated** repository. Users sign in and request access, and the
  owner approves requests and can revoke further downloads. A downloaded copy cannot be recalled; after
  download only the licence binds (decided 2026-11-04).
- **The pretraining data is not released** with the weights. Users can fine-tune existing heads and add new
  ones. Post-training that needs the original data, such as rehearsal or replay during continued pretraining,
  is not supported, and the model card says so (decided 2026-11-04).
  - One exception already exists: the repository tracks a small derived teaching table,
    `notebooks/teaching_inverse_design/data/qc_inverse_design_teaching.parquet`. It covers five properties and
    a down-sampled set of compositions. It is redistributed under CC BY 4.0 with attribution; see
    `data_sources.md`, "Data tracked in this repository".
  - It is not a substitute for the pretraining data, and the "no rehearsal" statement stands.
- **Weights and their derivatives are non-commercial** (decided 2026-11-04). A derivative includes a model
  fine-tuned from the weights.
- **Training data: compatible with this release, provisionally.** Every audited source is CC BY 4.0, read
  from the release actually used, and CC BY 4.0 requires attribution and a statement of changes.
  - Materials Project entries were checked one by one: none carries the non-commercial (`BY-NC`, GNoME)
    terms.
  - 31 Materials Project rows remain whose tag can no longer be read from the API.
  - Until their tag is recovered, the release training excludes those rows (`data_sources.md`, open item 1).

## Weights licence: CC BY-NC-SA 4.0 (decided 2026-11-04)

Scope, as decided:

| Model | Licence | Commercial use |
|---|---|---|
| Weights trained and published by the owner | CC BY-NC-SA 4.0 | no |
| A model derived from those weights, by fine-tuning, adding heads or any other training that starts from them | CC BY-NC-SA 4.0 (ShareAlike: same licence) | no |
| A model a user trains **from scratch** with this code and their own data | not covered. The code is Apache-2.0 and the user's data carries its own terms | the owner places no restriction |

The line between the last two rows is whether training starts from the published weights. The model card and
the gated-access terms state it in those words.

Alternatives considered:

| Option | Derivatives | Why not |
|---|---|---|
| CC BY-NC 4.0 | non-commercial as well: section 2(a)(1)(B) allows Adapted Material to be produced and shared for NonCommercial purposes only | the adapter may license their own contribution on any terms, including terms that do not carry the BY-NC conditions forward to their recipients; ShareAlike (section 3(b)) closes that by requiring adaptations to carry BY-NC-SA itself |
| Custom licence (`license: other`) | as written | needs legal drafting; kept as the fallback if predictions must also be restricted |

Two caveats remain:

- **Copyright in weights is unsettled.** Creative Commons licences rest on copyright and database rights, and
  whether trained weights are protected is not settled law in Japan or the US. The gated repository's access
  form adds a contractual layer: each user accepts the terms before download. That binds the user whether or
  not the weights are copyrightable.
- **Model outputs.** None of the CC licences clearly covers predictions made with the model. If commercial
  use of predictions should also be excluded, that needs a custom licence or explicit terms in the access form.

## Code licence: Apache-2.0 (decided 2026-11-04)

The requirement is that anyone may use, modify, redistribute and sell the code, provided the original licence
and attribution are kept. Apache-2.0 meets it. It was chosen over MIT, which also meets it, to leave room for a
possible commercial version later. On top of MIT it adds three things:

- a `NOTICE` file whose attribution text redistributions must carry;
- a statement in modified files that they were changed;
- an explicit patent licence from contributors.

Before the switch, `pyproject.toml` declared MIT and pointed to a `LICENSE` file that did not exist, while 84
tracked Python files carried `SPDX-License-Identifier: Apache-2.0`. Now:

- `LICENSE` holds the Apache-2.0 text, from apache.org/licenses/LICENSE-2.0.txt, sha256 `cfc7749b…3d30`.
- `NOTICE` holds:
  - the copyright line;
  - the full XenonPy BSD-3-Clause notice for `element_features.csv`, which the BSD conditions require
    redistributions to carry;
  - the CC BY 4.0 status of the teaching table;
  - the statement that the code licence covers neither the weights nor the training data.
- `pyproject.toml` declares `license = "Apache-2.0 AND BSD-3-Clause"`, because the package ships the
  BSD-licensed table (PEP 639 combines bundled licences with `AND`), and `license-files = ["LICENSE", "NOTICE"]`.
  The project's own code is Apache-2.0. The trove licence classifier was dropped: PEP 639 replaces it with the
  licence expression.
- Per-file SPDX headers stay as they are. Apache-2.0 does not require them; files without one are covered by
  `LICENSE`.

Every commit so far is by the repository owner (two git identities), so the switch needed no other
contributor's consent. A commercial version stays possible because the owner holds the copyright and can
license it on other terms. Once outside contributions arrive, keeping that option needs a contributor licence
agreement, since Apache-2.0 alone does not let the owner relicense contributed code.
