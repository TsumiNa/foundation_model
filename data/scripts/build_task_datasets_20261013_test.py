from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _load_module():
    path = Path(__file__).with_name("build_task_datasets_20261013.py")
    spec = importlib.util.spec_from_file_location("build_task_datasets_20261013", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_raw_column_prefers_exact_then_unit_suffix():
    m = _load_module()
    cols = ["composition", "Structure preference", "Curie temperature bcc[K]", "QMOF band gap PBE[eV]"]
    assert m.resolve_raw_column(cols, "Structure preference") == "Structure preference"
    assert m.resolve_raw_column(cols, "Curie temperature bcc") == "Curie temperature bcc[K]"
    assert m.resolve_raw_column(cols, "QMOF band gap PBE") == "QMOF band gap PBE[eV]"


def test_resolve_raw_column_rejects_missing_or_ambiguous():
    m = _load_module()
    with pytest.raises(ValueError):
        m.resolve_raw_column(["kp[W/mK]"], "klat")
    with pytest.raises(ValueError):
        m.resolve_raw_column(["E[eV]", "E[meV]"], "E")


def _index(**over):
    row = dict(
        task="t", file="a.parquet", column="x (normalized)", t_column="", raw_file="a_raw.parquet", raw_column="x[eV]"
    )
    row.update(over)
    return pd.DataFrame([row])


def test_validate_index_accepts_matching_files(tmp_path):
    m = _load_module()
    pd.DataFrame({"composition": ["Fe1"], "x (normalized)": [0.1]}).to_parquet(tmp_path / "a.parquet")
    pd.DataFrame({"composition": ["Fe1"], "x[eV]": [1.0]}).to_parquet(tmp_path / "a_raw.parquet")
    m.validate_index(_index(), tmp_path)


@pytest.mark.parametrize(
    "over", [{"column": "y (normalized)"}, {"raw_column": "x"}, {"t_column": "T (K)"}, {"raw_file": "missing.parquet"}]
)
def test_validate_index_reports_a_wrong_column_or_file(tmp_path, over):
    m = _load_module()
    pd.DataFrame({"composition": ["Fe1"], "x (normalized)": [0.1]}).to_parquet(tmp_path / "a.parquet")
    pd.DataFrame({"composition": ["Fe1"], "x[eV]": [1.0]}).to_parquet(tmp_path / "a_raw.parquet")
    with pytest.raises(ValueError, match="task index does not match"):
        m.validate_index(_index(**over), tmp_path)
