import numpy as np
import pandas as pd
import pytest

from navdiag.visibility import SkyMaskGrid, geometry_at_truth, match_truth, read_urbannav_truth


def test_reference_labels_conservative_geometry_and_coverage_guards():
    profile = np.full((1, 361), 30.)
    mask = SkyMaskGrid(np.array([[22., 114., 5.]]), profile)
    query = lambda az, el, **kw: mask.reference(22., 114., 5., 1, az, el, **kw)
    assert query(100, 45)['signal_label'] == 'los'
    assert query(100, 15)['signal_label'] == 'nlos'
    assert query(100, 33)['signal_label'] == 'unknown'
    assert mask.reference(22.01, 114., 5., 1, 100, 45)['reference_reason'] == 'outside_mask_coverage'
    assert mask.reference(22., 114., 10., 1, 100, 45)['reference_reason'] == 'height_mismatch'
    assert mask.reference(22., 114., 5., 4, 100, 45)['reference_reason'] == 'truth_quality'
    # Public 0/360 discrepancies must increase uncertainty, not change the file.
    mask.profiles[0, 360] = 70.
    assert query(0, 50)['signal_label'] == 'unknown'
    assert mask.profiles[0, 0] == 30


def test_truth_geometry_cardinal_directions_and_zenith():
    # At lat=lon=0, ECEF +Y is east, +Z is north and +X is up.
    earth = 6378137.
    positions = np.array([[earth, 2e7, 0], [earth, 0, 2e7], [earth + 2e7, 0, 0]])
    az, el = geometry_at_truth(positions, 0., 0., 0.)
    assert az[:2] == pytest.approx([90., 0.])
    assert el == pytest.approx([0., 0., 90.])


def test_truth_dms_time_and_no_index_alignment(tmp_path):
    p = tmp_path / 'truth.txt'
    row = '1621218775.00 2158 95593 22 18 04.31949 114 10 44.60559 3.472 0 0 0 0 0 0 0 0 0 2\n'
    p.write_text('header\n' + row)
    truth = read_urbannav_truth(p)
    assert truth.latitude_deg.iloc[0] == pytest.approx(22.301199858333334)
    ids, valid = match_truth(np.array([1621218775.006, 1621218776.]), truth)
    assert ids.tolist() == [0, 0] and valid.tolist() == [True, False]
    with pytest.raises(ValueError, match='Multiple'):
        match_truth(np.array([1621218775.001, 1621218775.002]), truth)
    p.write_text(row.replace('1621218775.00', '1621218793.00'))
    with pytest.raises(ValueError, match='disagree'):
        read_urbannav_truth(p)


def test_invalid_mask_grid_row_does_not_become_a_label(tmp_path):
    p = tmp_path / 'mask.csv'
    bad = [np.nan, np.nan, 5., *([42.] * 361)]
    good = [22., 114., 5., *([30.] * 361)]
    pd.DataFrame([good, bad]).to_csv(p, index=False)
    grid = SkyMaskGrid.read(p)
    assert grid.audit['grid_points'] == 1
    assert grid.audit['excluded_invalid_or_inside_building_grid_rows'] == 1
