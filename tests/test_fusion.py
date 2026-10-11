import numpy as np
import pytest

from navdiag.fusion import filter_positions, position_metrics


def test_position_accuracy_limits_are_inclusive_and_missing_errors_are_not_zero():
    result = position_metrics(np.array([0.5, 1., 2., 5., 10., np.nan]))
    assert result['epochs'] == 5
    assert result['within_error_limits_m'] == {'1': 0.4, '2': 0.6, '5': 0.8, '10': 1.0}
    assert position_metrics(np.array([np.nan]))['within_error_limits_m']['2'] is None


def test_planar_filter_tracks_motion_and_handles_missing_gnss():
    time = np.arange(20) * 1000
    positions = np.column_stack([np.arange(20) * 2, np.zeros(20)]).astype(float)
    positions[10:13] = np.nan
    result = filter_positions(time, positions, np.full(20, 4), gate_nis=1000)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[-1], [38, 0], atol=0.1)


def test_reference_truth_is_not_a_filter_argument_and_future_fixes_do_not_affect_past():
    time = np.arange(10) * 1000
    positions = np.column_stack([np.arange(10) * 2, np.zeros(10)]).astype(float)
    first = filter_positions(time[:5], positions[:5], np.ones(5))
    positions[8] += 10000
    full = filter_positions(time, positions, np.ones(10))
    np.testing.assert_allclose(first, full[:5])
    with pytest.raises(TypeError):
        filter_positions(time, positions, np.ones(10), ground_truth=positions)


def test_filter_rejects_invalid_timestamps():
    with pytest.raises(ValueError, match='strictly increasing'):
        filter_positions(np.array([0, 0]), np.zeros((2, 2)), np.ones(2))
