import numpy as np
import pytest

from navdiag.doppler import filter_doppler_positions


def test_velocity_observations_constrain_causal_motion_and_future_cannot_change_past():
    t = np.arange(40) * 1000
    truth = np.column_stack([np.arange(40) * 2., np.zeros(40)])
    rng = np.random.default_rng(42)
    fixes = truth + rng.normal(0, 5, truth.shape)
    velocities = np.tile([2., 0.], (40, 1))
    cv = np.tile(np.eye(2) * 0.04, (40, 1, 1))
    result, diagnostic = filter_doppler_positions(t, fixes, velocities, cv, 25.)
    assert np.mean((result - truth)**2) < np.mean((fixes - truth)**2)
    assert diagnostic['velocity_updates'] == 40
    before, _ = filter_doppler_positions(t[:15], fixes[:15], velocities[:15], cv[:15], 25.)
    fixes[30] += 1000
    velocities[30] += 1000
    after, diagnostic = filter_doppler_positions(t, fixes, velocities, cv, 25.)
    np.testing.assert_array_equal(before, after[:15])
    assert diagnostic['velocity_gate_rejections'] > 0
    assert diagnostic['position_gate_rejections'] > 0
    with pytest.raises(TypeError):
        filter_doppler_positions(t, fixes, velocities, cv, 25., ground_truth=truth)


def test_missing_doppler_falls_back_and_large_gap_resets_without_bridging():
    t = np.array([0, 1000, 20000])
    fixes = np.array([[0., 0.], [1., 0.], [100., 50.]])
    velocity = np.full((3, 2), np.nan)
    cv = np.full((3, 2, 2), np.nan)
    result, diagnostic = filter_doppler_positions(t, fixes, velocity, cv, 25.)
    assert diagnostic['missing_velocity_epochs'] == 3
    assert diagnostic['segments'] == 2
    np.testing.assert_array_equal(result[-1], fixes[-1])
    with pytest.raises(ValueError, match='strictly increasing'):
        filter_doppler_positions([0, 0, 1], fixes, velocity, cv, 25.)
