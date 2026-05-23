"""Tests for the full continual-rehearsal + inverse-design runner (config/catalogue/CLI logic).

Training and data loading are exercised by the smoke run, not here; these tests cover the pure
logic that is cheap and worth guarding: the task catalogue, config validation, and TOML/CLI parsing.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from foundation_model.scripts.continual_rehearsal_full import (
    ALLOY_PALETTE,
    DEFAULT_FIXED_TAIL,
    DEFAULT_SEQUENCE,
    INVERSE_PATH_CONFIGS,
    INVERSE_PATHS,
    REG_TASK_TITLES,
    TASK_SPECS,
    ContinualRehearsalFullConfig,
    ContinualRehearsalFullRunner,
    InverseScenario,
    _arrow,
    _display,
    _parse_args,
    _title,
)


def test_default_sequence_is_24_tasks_by_type():
    kinds = [TASK_SPECS[t]["kind"] for t in DEFAULT_SEQUENCE]
    assert len(DEFAULT_SEQUENCE) == 24
    assert kinds.count("reg") == 16
    assert kinds.count("kr") == 7
    assert kinds.count("clf") == 1


def test_catalogue_consistency():
    # Every sequenced task is known; kernel tasks declare a t_column; clf declares num_classes.
    for task in DEFAULT_SEQUENCE:
        spec = TASK_SPECS[task]
        assert spec["kind"] in {"reg", "kr", "clf"}
        if spec["kind"] == "kr":
            assert "t_column" in spec
        if spec["kind"] == "clf":
            assert "num_classes" in spec
    # The fixed tail is the last segment of the default sequence.
    assert DEFAULT_SEQUENCE[-len(DEFAULT_FIXED_TAIL) :] == DEFAULT_FIXED_TAIL
    # material_type is last so the QC classifier is freshest for inverse design.
    assert DEFAULT_SEQUENCE[-1] == "material_type"


def test_inverse_path_configs_match_demo():
    # 8 configurations — 3 latent ae_align_scale points + 5 composition configs — mirroring the
    # demo's paper_inverse_comparison.py so the figures read the same across runners.
    assert len(INVERSE_PATH_CONFIGS) == 8
    methods = [c["method"] for c in INVERSE_PATH_CONFIGS]
    assert methods.count("latent") == 3
    assert methods.count("composition") == 5
    latent_alphas = [c["ae_align_scale"] for c in INVERSE_PATH_CONFIGS if c["method"] == "latent"]
    assert latent_alphas == [0.0, 0.25, 1.0]
    # The key list is a flat str list of unique stable identifiers used as result subdir names.
    assert INVERSE_PATHS == [c["key"] for c in INVERSE_PATH_CONFIGS]
    assert len(set(INVERSE_PATHS)) == len(INVERSE_PATHS)
    # One config row must hit each demo configuration knob.
    keys = set(INVERSE_PATHS)
    assert {
        "latent_align0p0",
        "latent_align0p25",
        "latent_align1p0",
        "comp_seed",
        "comp_seed_blend",
        "comp_seed_blend_palette",
        "comp_seed_blend_palette_lowdiv",
        "comp_random",
    } == keys


def test_reg_task_titles_include_scenario_targets():
    # Every reg task across the three default scenarios should have a paper-style panel title.
    for t in ("formation_energy", "klat", "magnetic_moment", "tc"):
        assert t in REG_TASK_TITLES
        assert "[" in REG_TASK_TITLES[t] and "]" in REG_TASK_TITLES[t]  # units present
        assert REG_TASK_TITLES[t].endswith(("↑", "↓"))


def test_alloy_palette_contents():
    # Plan §5 originally specified 41 elements; extended 2026-05 with the full Hf–Pt 5d TM row
    # (7 symbols) → 48. The three Au-Ga-Ln explicit seeds must still fit.
    assert len(ALLOY_PALETTE) == 48
    for sym in ("Au", "Ga", "Gd", "Tb", "Dy", "Mg", "Pd", "Al"):
        assert sym in ALLOY_PALETTE
    # 5d transition metals (Hf–Pt) — newly added.
    for sym in ("Hf", "Ta", "W", "Re", "Os", "Ir", "Pt"):
        assert sym in ALLOY_PALETTE
    # Radioactive / unwanted symbols deliberately excluded.
    for sym in ("Pu", "Tc", "Pm"):
        assert sym not in ALLOY_PALETTE


def test_default_config_valid_and_inverse_defaults():
    cfg = ContinualRehearsalFullConfig()
    assert len(cfg.inverse_scenarios) == 3
    assert all(isinstance(sc, InverseScenario) for sc in cfg.inverse_scenarios)
    # Plan §5 defaults: 20 seeds (17 strategy + 3 Au-Ga-Ln) + the 41-element palette. The single-
    # value ae_align / seed_blend / diversity knobs are fixed in INVERSE_PATH_CONFIGS, not the
    # config dataclass — see test_inverse_path_configs_match_demo.
    assert cfg.inverse_n_seeds == 20
    assert cfg.inverse_composition_allowed_elements == ALLOY_PALETTE
    assert cfg.inverse_seed_explicit_append == ["Au65 Ga20 Gd15", "Au65 Ga20 Tb15", "Au65 Ga20 Dy15"]


def test_unknown_task_raises():
    with pytest.raises(ValueError, match="Unknown task"):
        ContinualRehearsalFullConfig(task_sequence=["density", "not_a_task", "material_type"])


def test_duplicate_task_raises():
    seq = list(DEFAULT_SEQUENCE) + ["density"]
    with pytest.raises(ValueError, match="duplicates"):
        ContinualRehearsalFullConfig(task_sequence=seq)


def test_fixed_tail_must_be_in_sequence():
    with pytest.raises(ValueError, match="fixed_tail"):
        ContinualRehearsalFullConfig(fixed_tail=["formation_energy", "not_present", "material_type"])


@pytest.mark.parametrize("ratio_kwargs", [{"replay_ratio": -0.1}, {"replay_ratio_high": 1.5}])
def test_replay_ratio_bounds(ratio_kwargs):
    with pytest.raises(ValueError, match="must be in"):
        ContinualRehearsalFullConfig(**ratio_kwargs)


def test_allowed_elements_validation():
    with pytest.raises(ValueError, match="non-empty"):
        ContinualRehearsalFullConfig(inverse_composition_allowed_elements=[])
    with pytest.raises(ValueError, match="not in DEFAULT_ELEMENTS"):
        ContinualRehearsalFullConfig(inverse_composition_allowed_elements=["Mg", "Xx"])


def test_inverse_scenario_length_mismatch():
    with pytest.raises(ValueError, match="equal length"):
        InverseScenario("bad", ["formation_energy"], [-2.0, 2.0])


def test_scenario_task_must_be_regression():
    # material_type is a classification task → cannot be a regression objective.
    bad = InverseScenario("bad", ["material_type"], [1.0])
    with pytest.raises(ValueError, match="must be a"):
        ContinualRehearsalFullConfig(inverse_scenarios=[bad])

    # a kernel-regression task is also not a scalar regression objective.
    bad_kr = InverseScenario("bad_kr", ["dos_density"], [1.0])
    with pytest.raises(ValueError, match="must be a"):
        ContinualRehearsalFullConfig(inverse_scenarios=[bad_kr])


def test_scenario_task_must_be_in_sequence():
    short_seq = ["density", "material_type"]
    bad = InverseScenario("bad", ["formation_energy"], [-2.0])
    with pytest.raises(ValueError, match="not in task_sequence"):
        ContinualRehearsalFullConfig(task_sequence=short_seq, fixed_tail=["material_type"], inverse_scenarios=[bad])


def test_material_type_required():
    seq = [t for t in DEFAULT_SEQUENCE if t != "material_type"]
    with pytest.raises(ValueError, match="material_type"):
        ContinualRehearsalFullConfig(task_sequence=seq, fixed_tail=["formation_energy"], inverse_scenarios=[])


def test_invalid_seed_strategy():
    with pytest.raises(ValueError, match="inverse_seed_strategy"):
        ContinualRehearsalFullConfig(inverse_seed_strategy="bogus")


def test_display_helpers():
    assert _display("formation_energy") == "Formation Energy"
    assert "Density" in _title("density")
    assert "normalized" in _title("density")  # qc scale
    assert "z-scored" in _title("tc")  # raw scale
    assert _arrow(-2.0) == "↓"
    assert _arrow(2.0) == "↑"


def test_element_system_and_dedup():
    # Element-system extraction ignores numeric ratios; dedup keeps the first per element set.
    assert ContinualRehearsalFullRunner._element_system("Au65 Ga20 Gd15") == frozenset({"Au", "Ga", "Gd"})
    assert ContinualRehearsalFullRunner._element_system("Au0.65Ga0.20Gd0.15") == frozenset({"Au", "Ga", "Gd"})
    deduped = ContinualRehearsalFullRunner._dedupe_by_element_system(
        ["Mg2 Zn1 Y1", "Mg1 Zn2 Y1", "Al1 Cu1 Fe1", "Mg3 Zn3 Y2"], n=10
    )
    # Mg-Zn-Y duplicates collapsed to the first occurrence; Al-Cu-Fe kept.
    assert deduped == ["Mg2 Zn1 Y1", "Al1 Cu1 Fe1"]


def test_parse_args_tuple_return_and_toml(tmp_path: Path):
    toml = tmp_path / "cfg.toml"
    toml.write_text(
        textwrap.dedent(
            """
            qc_preprocessing_path = ""
            task_sequence = ["density", "formation_energy", "magnetic_moment", "klat", "tc", "material_type"]
            fixed_tail = ["formation_energy", "magnetic_moment", "tc", "klat", "material_type"]
            replay_ratio_high = 0.2
            inverse_composition_allowed_elements = ["Mg", "Al", "Cu", "Pd"]

            [[inverse_scenarios]]
            name = "s1"
            reg_tasks = ["formation_energy", "klat"]
            reg_targets = [-2.0, 2.0]

            [[inverse_scenarios]]
            name = "s2"
            reg_tasks = ["formation_energy", "tc", "magnetic_moment"]
            reg_targets = [-2.0, 2.0, 2.0]
            """
        ),
        encoding="utf-8",
    )
    cfg, args = _parse_args(["--config-file", str(toml), "--sample-per-dataset", "500", "--max-epochs-per-step", "2"])
    # Empty-string path field becomes None (no dropped_idx filtering).
    assert cfg.qc_preprocessing_path is None
    # inverse_scenarios dicts are coerced to InverseScenario objects.
    assert [sc.name for sc in cfg.inverse_scenarios] == ["s1", "s2"]
    assert all(isinstance(sc, InverseScenario) for sc in cfg.inverse_scenarios)
    # CLI overrides land on the config; the palette override propagates from TOML.
    assert cfg.sample_per_dataset == 500
    assert cfg.max_epochs_per_step == 2
    assert cfg.replay_ratio_high == 0.2
    assert cfg.inverse_composition_allowed_elements == ["Mg", "Al", "Cu", "Pd"]
    # Namespace returned alongside config so main() can read --inverse-only.
    assert args.inverse_only is None


def test_parse_args_inverse_only_flag(tmp_path: Path):
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"placeholder")  # presence-only; loading is exercised by smoke
    _cfg, args = _parse_args(["--inverse-only", str(ckpt)])
    assert args.inverse_only == ckpt


def test_parse_args_unknown_key_ignored(tmp_path: Path):
    toml = tmp_path / "cfg.toml"
    toml.write_text("totally_unknown_key = 7\nreplay_ratio = 0.05\n", encoding="utf-8")
    cfg, _args = _parse_args(["--config-file", str(toml)])
    assert cfg.replay_ratio == 0.05
    assert not hasattr(cfg, "totally_unknown_key")
