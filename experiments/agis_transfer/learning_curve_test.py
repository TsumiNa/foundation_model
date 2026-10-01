# Copyright 2026 TsumiNa.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from learning_curve import aggregate_learning_curve


def test_learning_curve_weights_materials_after_collapsing_checkpoint_repeats() -> None:
    rows = []
    for composition in range(8):
        for pressure in (0, 10, 20):
            for route, repeats in (("scratch", 1), ("direct", 10)):
                for checkpoint in range(repeats):
                    value = float(composition) + (1 if route == "direct" else 0)
                    rows.append(
                        {
                            "n_train": 3,
                            "composition": str(composition),
                            "pressure": pressure,
                            "fold": composition + 1,
                            "route": route,
                            "setting": "unfrozen",
                            "is_anchor": True,
                            "z_rmse": value,
                            "z_rmse_low_T": value,
                            "relative_rmse": value,
                            "r2": value,
                        }
                    )
    summary, material = aggregate_learning_curve(pd.DataFrame(rows))
    aggregate = summary[(summary.scope == "anchor") & (summary.pressure == "all")].set_index("route")
    assert aggregate.loc["scratch", "z_rmse"] == 3.5
    assert aggregate.loc["direct", "z_rmse"] == 4.5
    assert set(material.groupby(["scope", "route"]).size()) == {24}
    with pytest.raises(ValueError, match="eight material"):
        aggregate_learning_curve(pd.DataFrame(rows).query('composition != "0"'))
