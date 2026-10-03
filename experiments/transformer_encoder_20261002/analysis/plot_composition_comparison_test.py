import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from plot_composition_comparison import composition_cards, paired_summary, representatives


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
