import numpy as np
import pandas as pd
from navdiag.gsdc import GPS_EPOCH_UNIX_MS, GPS_MINUS_UTC_MS, read_observations, read_reference
from navdiag.positioning import ecef_to_lla


def test_legacy_derived_corrections_and_frequency_deduplication(tmp_path):
    frame = pd.DataFrame({
        'millisSinceGpsEpoch': [1e12, 1e12+1000, 1e12+1000],
        'receivedSvTimeInGpsNanos': [1e18-70e6, 1e18-70e6, 1e18-70e6],
        'constellationType': [1, 1, 1], 'svid': [2, 2, 2],
        'xSatPosM': [2e7]*3, 'ySatPosM': [1e7]*3, 'zSatPosM': [2e7]*3,
        'rawPrM': [2.4e7]*3, 'rawPrUncM': [3, 3, 5], 'satClkBiasM': [100]*3,
        'isrbM': [5]*3, 'ionoDelayM': [10]*3, 'tropoDelayM': [2]*3,
    })
    path = tmp_path / 'device_derived.csv'
    frame.to_csv(path, index=False)
    observations = read_observations(path)
    assert len(observations) == 1
    assert observations.pseudorange_m.iloc[0] == 2.4e7 + 83
    assert np.isnan(observations.cn0_dbhz.iloc[0])  # Never fabricate unavailable signal strength.
    assert observations.timestamp_ms.iloc[0] == 1e12 + GPS_EPOCH_UNIX_MS - GPS_MINUS_UTC_MS


def test_legacy_epoch_mapping_respects_irregular_intervals_and_rejects_bad_flight_time(tmp_path):
    base = 1e12
    frame = pd.DataFrame({
        'millisSinceGpsEpoch': [base, base+999, base+2000, base+2000],
        'receivedSvTimeInGpsNanos': [base*1e6-70e6, base*1e6-70e6,
                                     (base+999)*1e6-70e6, (base+999)*1e6-400e6],
        'constellationType': 1, 'svid': [1, 1, 1, 2],
        'xSatPosM': 2e7, 'ySatPosM': 1e7, 'zSatPosM': 2e7,
        'rawPrM': 2.4e7, 'rawPrUncM': 3, 'satClkBiasM': 100,
        'isrbM': 5, 'ionoDelayM': 10, 'tropoDelayM': 2,
    })
    path = tmp_path / 'legacy.csv'; frame.to_csv(path, index=False)
    observed = read_observations(path)
    np.testing.assert_allclose(observed.timestamp_ms - GPS_EPOCH_UNIX_MS + GPS_MINUS_UTC_MS,
                               [base, base+999], rtol=0, atol=0.001)
    assert observed.attrs['time_corrections']['first_or_invalid_epoch_rows_removed'] == 1
    assert observed.attrs['time_corrections']['timing_outlier_rows_removed'] == 1


def test_legacy_reference_height_correction_does_not_shift_reference_time(tmp_path):
    legacy = pd.DataFrame({'millisSinceGpsEpoch': [1e12,1e12+1000],
                           'latDeg': [37.,37.], 'lngDeg': [-122.,-122.],
                           'heightAboveWgs84EllipsoidM': [100.,101.]})
    path = tmp_path / 'truth.csv'; legacy.to_csv(path,index=False)
    times, xyz = read_reference(path)
    assert times[0] == 1e12 + GPS_EPOCH_UNIX_MS - GPS_MINUS_UTC_MS
    np.testing.assert_allclose(ecef_to_lla(xyz)[:,2], [39.,40.], atol=1e-6)
    modern = legacy.rename(columns={'millisSinceGpsEpoch':'UnixTimeMillis','latDeg':'LatitudeDegrees',
                                    'lngDeg':'LongitudeDegrees','heightAboveWgs84EllipsoidM':'AltitudeMeters'})
    modern.to_csv(path,index=False)
    times, xyz = read_reference(path)
    assert times[0] == 1e12
    np.testing.assert_allclose(ecef_to_lla(xyz)[:,2], [100.,101.], atol=1e-6)
