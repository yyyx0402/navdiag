import numpy as np
import pandas as pd
import pytest

pytest.importorskip('torch')
from navdiag.temporal import causal_windows


def test_future_observations_cannot_change_past_windows():
    first = pd.DataFrame({'trip_id': ['a'] * 3, 'timestamp_ms': [0, 1000, 2000]})
    values = np.array([[1], [2], [3]], dtype=float)
    before = causal_windows(first, values, 3)
    extended = pd.concat([first, pd.DataFrame({'trip_id': ['a'], 'timestamp_ms': [3000]})], ignore_index=True)
    after = causal_windows(extended, np.vstack([values, [[999]]]), 3)
    np.testing.assert_array_equal(before, after[:3])
    np.testing.assert_array_equal(before[:, :, 0], [[1, 1, 1], [1, 1, 2], [1, 2, 3]])


def test_windows_reset_for_another_phone_or_time_gap():
    frame = pd.DataFrame({'trip_id': ['a'] * 4, 'device_id': ['p1', 'p1', 'p1', 'p2'],
                          'timestamp_ms': [0, 1000, 10000, 11000]})
    windows = causal_windows(frame, np.arange(4).reshape(-1, 1), 3)
    np.testing.assert_array_equal(windows[2, :, 0], [2, 2, 2])
    np.testing.assert_array_equal(windows[3, :, 0], [3, 3, 3])
