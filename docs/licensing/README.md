# Licensing of the code, the pretrained weights and the training data

Working notes for publishing the pretrained models. Not legal advice; the release terms should be reviewed by
the institute's intellectual-property office before the first public release.

| File | Contents |
|---|---|
| `data_sources.md` | licence of every training-data source, how it was verified, open checks, citations |

## Decided

- **Hosting:** model code stays in this package. A Hugging Face model repository holds weights,
  `config.json` and the model card (decided 2026-10-30).
- **Access:** the weights are published as a **gated** repository. Users sign in and request access, and the
  owner approves requests and can revoke further downloads. A downloaded copy cannot be recalled; after
  download only the licence binds (decided 2026-11-04).
- **No training data is released.** Users can fine-tune existing heads and add new ones. Post-training that
  needs the original data, such as rehearsal or replay during continued pretraining, is not supported. The
  model card says so (decided 2026-11-04).
- **Weights and their derivatives are non-commercial** (decided 2026-11-04). A derivative includes a model
  fine-tuned from the weights.
- **Training data:** every audited source is CC BY 4.0, which allows this release. It requires attribution and
  a statement of changes; see `data_sources.md` for the one open item on Materials Project `BY-NC` entries.

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

| Option | Derivatives must stay non-commercial | Why not |
|---|---|---|
| CC BY-NC 4.0 | only the original weights inside a derivative stay non-commercial; the adapter may license their own additions differently | weaker for "derivatives too" |
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
- `NOTICE` holds the copyright line and the XenonPy origin of `element_features.csv`, and states that the code
  licence does not cover the weights or the training data.
- `pyproject.toml` declares `license = "Apache-2.0"` and `license-files = ["LICENSE", "NOTICE"]`. The trove
  licence classifier was dropped: PEP 639 replaces it with the licence expression.
- Per-file SPDX headers stay as they are. Apache-2.0 does not require them; files without one are covered by
  `LICENSE`.

Every commit so far is by the repository owner (two git identities), so the switch needed no other
contributor's consent. A commercial version stays possible because the owner holds the copyright and can
license it on other terms. Once outside contributions arrive, keeping that option needs a contributor licence
agreement, since Apache-2.0 alone does not let the owner relicense contributed code.
