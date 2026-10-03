import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from plot_composition_comparison import composition_cards, main, paired_summary, representatives


def fixture_frame():
    return pd.DataFrame(
        [
            dict(
                composition=f"c{i}",
                seed=s,
                k=7,
                fraction=1.0,
                readout="linear_output",
                true_mlp=float(i),
                true_transformer=float(i),
                pred_mlp=float(i + s),
                pred_transformer=float(i - s),
            )
            for i in range(20)
            for s in range(3)
        ]
    )


def test_paired_summary_and_selection_ignore_predictions_and_input_order():
    data = fixture_frame()
    summary = paired_summary(data.sample(frac=1, random_state=2), 1.0)
    assert summary.loc[0, "mlp"] == 1.0 and summary.loc[0, "transformer"] == -1.0
    assert summary.loc[0, "mlp_sd"] == 1.0
    chosen = representatives(summary)
    assert len(chosen) == 12
    assert chosen.reference_rank.tolist()[0] == 1 and chosen.reference_rank.tolist()[-1] == 20
    changed = data.assign(pred_mlp=99999.0, pred_transformer=-99999.0)
    assert representatives(paired_summary(changed, 1.0)).composition.tolist() == chosen.composition.tolist()


@pytest.mark.parametrize("defect", ["missing_seed", "duplicate", "bad_reference", "nan"])
def test_pairing_rejects_corrupt_inputs(defect):
    data = fixture_frame()
    if defect == "missing_seed":
        data = data.iloc[1:]
    if defect == "duplicate":
        data = pd.concat([data, data.iloc[:1]])
    if defect == "bad_reference":
        data.loc[0, "true_transformer"] = 99.0
    if defect == "nan":
        data.loc[0, "pred_mlp"] = np.nan
    with pytest.raises(ValueError):
        paired_summary(data, 1.0)


def test_selection_boundary():
    summary = paired_summary(fixture_frame(), 1.0)
    assert len(representatives(summary, 100)) == 20
    with pytest.raises(ValueError):
        representatives(summary, 0)


def test_card_titles_preserve_unreduced_composition_identity():
    selected = representatives(paired_summary(fixture_frame(), 1.0))
    selected["composition"] = ["Fe4 O6", *["Si2 O4"] * 11]
    fig = composition_cards(selected, 1.0)
    assert fig.axes[0].get_title() == "Fe$_{4}$O$_{6}$"
    assert fig.axes[1].get_title() == "Si$_{2}$O$_{4}$"
    plt.close(fig)


def test_main_rejects_incomplete_study(tmp_path, monkeypatch):
    frame = fixture_frame()
    source = tmp_path / "input.csv"
    pd.concat([frame, frame.assign(fraction=0.1)]).to_csv(source, index=False)
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["plot", "--input", str(source), "--output", str(output)])
    with pytest.raises(ValueError, match="697"):
        main()
    assert not output.exists()


def test_main_generates_complete_artifacts(tmp_path, monkeypatch):
    frame = fixture_frame().iloc[:3].copy()
    rows = [frame.assign(composition=f"Fe{i + 1} O1", true_mlp=float(i), true_transformer=float(i)) for i in range(697)]
    full = pd.concat(rows, ignore_index=True)
    source = tmp_path / "input.csv"
    pd.concat([full, full.assign(fraction=0.1)], ignore_index=True).to_csv(source, index=False)
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["plot", "--input", str(source), "--output", str(output)])
    main()
    assert len(list(output.glob("*.png"))) == 6
    pdf = (output / "Direct_prediction_comparison_20261003.pdf").read_bytes()
    assert len(re.findall(rb"/Type /Page\b", pdf)) == 6
    a = pd.read_csv(output / "selected_compositions_f100.csv")
    b = pd.read_csv(output / "selected_compositions_f010.csv")
    assert len(a) == 12 and a.composition.tolist() == b.composition.tolist()
    for fraction in ["010", "100"]:
        assert len(pd.read_csv(output / f"all_compositions_f{fraction}.csv")) == 697
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert provenance["fractions"] == [1.0, 0.1]
    assert provenance["test_compositions"] == 697
    assert "pages 1–3 use 100%" in (output / "README.md").read_text()
