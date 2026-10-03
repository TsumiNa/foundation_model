import os
from pathlib import Path
import shutil
import subprocess


def test_wrong_image_is_rejected_before_first_container_execution(tmp_path):
    source = Path(__file__).parent
    cfg = tmp_path / "experiments/transformer_scope_20261003/configs/study.toml"
    cfg.parent.mkdir(parents=True)
    shutil.copyfile(source.parent / "configs/study.toml", cfg)
    fake_image = tmp_path / "untrusted.sif"
    fake_image.write_text("not the registered image")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    runtime = bin_dir / "apptainer"
    runtime.write_text('#!/bin/sh\nprintf called > "$TOUCH_MARKER"\nexit 99\n')
    runtime.chmod(0o755)
    checksum = bin_dir / "sha256sum"
    checksum.write_text("#!/bin/sh\nprintf 'incorrect-hash file\\n'\n")
    checksum.chmod(0o755)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(
        FM_WORKSPACE=str(tmp_path),
        FM_IMAGE=str(fake_image),
        FM_DATA_DIR=str(tmp_path / "data"),
        FM_OUTPUT_ROOT=str(tmp_path / "out"),
        FM_BENCHMARK_REVISION="a" * 40,
        PHASE="pilot",
        APPTAINER=str(runtime),
        TOUCH_MARKER=str(tmp_path / "called"),
        PATH=str(bin_dir) + os.pathsep + env["PATH"],
    )
    result = subprocess.run(
        ["bash", "-c", 'module() { :; }; export -f module; bash "$1"', "_", str(source / "array.sbatch")],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2 and "FAIL SIF checksum" in result.stdout
    assert not (tmp_path / "called").exists()
