import numpy as np
import pandas as pd
import pytest

from navdiag.data import FEATURES, generate_demo, load_dataset, split_by_trip


def test_trip_split_does_not_leak_and_is_reproducible(tmp_path):
    frame = load_dataset(generate_demo(tmp_path / "demo.csv"))
    splits = split_by_trip(frame)
    ids = [set(part.trip_id) for part in splits.values()]
    assert not ids[0] & ids[1]
    assert not ids[0] & ids[2]
    assert not ids[1] & ids[2]
    assert sum(map(len, splits.values())) == len(frame)
    for name, part in splits.items():
        pd.testing.assert_frame_equal(part, split_by_trip(frame)[name])
    assert "horizontal_error_m" not in FEATURES
    assert "trip_id" not in FEATURES


@pytest.mark.parametrize("column,value", [
    ("horizontal_error_m", -1), ("hdop", np.inf), ("trip_id", ""),
    ("cn0_mean_dbhz", np.nan), ("data_kind", "unknown"),
])
def test_bad_data_is_rejected_before_training(tmp_path, column, value):
    path = generate_demo(tmp_path / "demo.csv")
    frame = pd.read_csv(path)
    frame.loc[0, column] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_dataset(path)


def test_duplicate_epoch_is_rejected(tmp_path):
    path = generate_demo(tmp_path / "demo.csv")
    frame = pd.read_csv(path)
    pd.concat([frame, frame.iloc[[0]]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        load_dataset(path)
