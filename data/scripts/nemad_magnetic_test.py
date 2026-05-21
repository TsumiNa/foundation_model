from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pandas as pd


def _load_module():
    module_path = Path(__file__).with_name("nemad_magnetic.py")
    spec = importlib.util.spec_from_file_location("nemad_magnetic", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_convert_composition_rejects_dummy_species_names():
    module = _load_module()
    composition = module.convert_composition("Fe2O3", as_formula=False)

    assert composition is not None
    assert module.convert_composition("Fe2O3") == composition.formula
    assert module.convert_composition("Y3Fe5O12 (YIG)") is None
    assert module.convert_composition("CoX") is None
    assert module.convert_composition("CoX", require_computable_mass=False, as_formula=False) is not None


def test_clean_magnetic_dataframe_skips_mass_conversion_for_invalid_material_name():
    module = _load_module()
    magnetic = pd.DataFrame(
        {
            "Material_Name": ["Y3Fe5O12 (YIG)", "Fe2O3"],
            "Magnetic_Moment": ["1.2 μB/f.u.", "0.5 μB/f.u."],
            "Magnetization": ["10 emu/g", "10 emu/g"],
            "Curie": ["300", "300"],
            "Curie(Tc)": ["300 K", "300 K"],
            "Neel": ["100", "100"],
            "Neel(Tn)": ["100 K", "100 K"],
        }
    )

    cleaned = module.clean_magnetic_dataframe(magnetic)

    assert math.isnan(cleaned.loc[0, "Magnetization"])
    assert not math.isnan(cleaned.loc[1, "Magnetization"])
    assert cleaned.loc[0, "Curie"] == 300.0
    assert cleaned.loc[0, "Neel"] == 100.0


def test_clean_magnetic_dataframe_keeps_numeric_kelvin_values_without_space():
    module = _load_module()
    magnetic = pd.DataFrame(
        {
            "Material_Name": ["Nd0.5Sr0.5MnO3", "La0.6Ca0.4MnO3"],
            "Magnetic_Moment": ["0.5μB/f.u.", "0.7 μB/f.u."],
            "Magnetization": ["10Am^2/kg", "5 A·m²/mol"],
            "Curie": ["250", "8"],
            "Curie(Tc)": ["250K", "8K"],
            "Neel": ["50", "29"],
            "Neel(Tn)": ["50K", "29K"],
        }
    )

    cleaned = module.clean_magnetic_dataframe(magnetic)

    assert cleaned.loc[0, "Magnetic_Moment"] == 0.5
    assert cleaned.loc[1, "Magnetic_Moment"] == 0.7
    assert not math.isnan(cleaned.loc[0, "Magnetization"])
    assert cleaned.loc[1, "Magnetization"] == 5.0
    assert cleaned.loc[0, "Curie"] == 250.0
    assert cleaned.loc[1, "Curie"] == 8.0
    assert cleaned.loc[0, "Neel"] == 50.0
    assert cleaned.loc[1, "Neel"] == 29.0


def test_clean_magnetic_dataframe_keeps_single_value_magnetic_moment_with_conditions():
    module = _load_module()
    magnetic = pd.DataFrame(
        {
            "Material_Name": ["Fe2O3", "Fe2O3"],
            "Magnetic_Moment": [
                "0.04 μB/f.u. at 2 K and 7 T",
                "1.63 μB/f.u. (ordered), 1.82 μB/f.u. (disordered)",
            ],
            "Magnetization": ["10 emu/g", "10 emu/g"],
            "Curie": ["300", "300"],
            "Curie(Tc)": ["300 K", "300 K"],
            "Neel": ["100", "100"],
            "Neel(Tn)": ["100 K", "100 K"],
        }
    )

    cleaned = module.clean_magnetic_dataframe(magnetic)

    assert cleaned.loc[0, "Magnetic_Moment"] == 0.04
    assert math.isnan(cleaned.loc[1, "Magnetic_Moment"])
