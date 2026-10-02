from pathlib import Path

import pytest

from make_worklist import write_worklist

PROTOCOL = Path(__file__).parents[1] / "configs/protocol.toml"


def test_disjoint_lanes_cover_full_campaign_and_paired_calibration(tmp_path):
    path = tmp_path / "lanes.tsv"
    assert write_worklist(PROTOCOL, path, None, None) == 24
    lines = path.read_text().splitlines()
    assert len(set(lines)) == 24
    assert write_worklist(PROTOCOL, path, ["grouped_concat"], [0, 1]) == 2
    assert path.read_text() == "grouped_concat\t0\ngrouped_concat\t1\n"


@pytest.mark.parametrize("arms,seeds", [(["missing"], [0]), (["mlp_tuned"], [9]), (["mlp_tuned"] * 2, [0])])
def test_unknown_or_duplicate_lane_is_rejected(tmp_path, arms, seeds):
    with pytest.raises(ValueError):
        write_worklist(PROTOCOL, tmp_path / "bad.tsv", arms, seeds)
