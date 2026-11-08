"""The pre-2.0 name ``foundation_model`` keeps working: same module objects, old pickles load."""

import importlib
import io
import pickle
import subprocess
import sys
import warnings

import pytest


def test_old_name_is_the_same_module() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = importlib.import_module("foundation_model.hub.bundle")

    import yammi.hub.bundle as new

    assert old is new
    assert importlib.import_module("foundation_model") is importlib.import_module("yammi")


def test_pickle_written_under_the_old_path_loads() -> None:
    from yammi.data.targets import ClipStep

    blob = pickle.dumps(ClipStep(lower=0.0, upper=1.0), protocol=2)
    blob = blob.replace(b"yammi.data.targets", b"foundation_model.data.targets")
    assert b"foundation_model.data.targets" in blob
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        restored = pickle.load(io.BytesIO(blob))
    assert type(restored) is ClipStep and restored.upper == 1.0


@pytest.mark.parametrize("stmt", ["import foundation_model", "import foundation_model.cli.main"])
def test_old_import_warns(stmt: str) -> None:
    code = f"import warnings; warnings.simplefilter('error', DeprecationWarning)\ntry:\n    {stmt}\nexcept DeprecationWarning as e:\n    print('warned', e)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert "renamed to yammi" in out


def test_old_module_runs_with_python_m() -> None:
    out = subprocess.run(
        [sys.executable, "-W", "ignore", "-m", "foundation_model.cli.main", "--help"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "pretrain" in out and "Usage:" in out


def test_alias_keeps_the_canonical_spec() -> None:
    import importlib.resources

    import yammi.utils

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        importlib.import_module("foundation_model.utils")
    assert yammi.utils.__spec__ is not None and yammi.utils.__spec__.name == "yammi.utils"
    assert importlib.resources.files("yammi.utils").joinpath("element_features.csv").is_file()
    assert importlib.reload(yammi.utils) is yammi.utils
