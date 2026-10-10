import numpy as np
import pytest

from navdiag.positioning import (
    align_reference, earth_rotation_correction, ecef_to_enu, ecef_to_lla,
    enu_to_ecef, lla_to_ecef, solve_wls,
)


def test_coordinate_transforms_round_trip_including_pole():
    lla = np.array([[37.4, -122.1, 32], [-33.9, 151.2, 100], [89.999, 30, 100]])
    np.testing.assert_allclose(ecef_to_lla(lla_to_ecef(lla)), lla, atol=1e-6)
    local = np.array([[100, -20, 3], [0, 0, 0]])
    np.testing.assert_allclose(ecef_to_enu(enu_to_ecef(local, lla[0]), lla[0]), local, atol=1e-7)


def test_wls_recovers_receiver_and_independent_constellation_clocks():
    rng = np.random.default_rng(4)
    receiver = lla_to_ecef(np.array([37.4, -122.1, 50]))
    directions = rng.normal(size=(14, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    satellites = receiver + directions * 2.2e7
    systems = np.array(['gps'] * 7 + ['galileo'] * 7)
    biases = np.where(systems == 'gps', 60000, 61234)
    ranges = np.linalg.norm(earth_rotation_correction(satellites, receiver) - receiver, axis=1)
    solution = solve_wls(satellites, ranges + biases, systems)
    np.testing.assert_allclose(solution.xyz, receiver, atol=1e-3)
    assert solution.clocks_m['gps'] == pytest.approx(60000, abs=1e-3)
    assert solution.clocks_m['galileo'] == pytest.approx(61234, abs=1e-3)
    assert np.isfinite(solution.hdop)


def test_degenerate_satellite_geometry_is_rejected():
    satellites = np.full((8, 3), 2e7)
    with pytest.raises(ValueError, match='rank deficient'):
        solve_wls(satellites, np.full(8, 2.5e7))


def test_reference_alignment_does_not_extrapolate_or_bridge_large_gaps():
    reference_times = np.array([0, 1000, 6000])
    points = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    matched, valid = align_reference(np.array([-1, 500, 2000, 6000, 7000]), reference_times, points, 2000)
    assert valid.tolist() == [False, True, False, True, False]
    np.testing.assert_allclose(matched[1], [1.5, 2.5, 3.5])
    assert np.isnan(matched[~valid]).all()
