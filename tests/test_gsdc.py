import numpy as np
import pandas as pd

from navdiag.gsdc import read_observations


def test_legacy_derived_corrections_and_frequency_deduplication(tmp_path):
    frame = pd.DataFrame({
        'millisSinceGpsEpoch': [1e12, 1e12], 'constellationType': [1, 1], 'svid': [2, 2],
        'xSatPosM': [2e7, 2e7], 'ySatPosM': [1e7, 1e7], 'zSatPosM': [2e7, 2e7],
        'rawPrM': [2.4e7, 2.4e7], 'rawPrUncM': [3, 5], 'satClkBiasM': [100, 100],
        'isrbM': [5, 5], 'ionoDelayM': [10, 10], 'tropoDelayM': [2, 2],
    })
    path = tmp_path / 'device_derived.csv'
    frame.to_csv(path, index=False)
    observations = read_observations(path)
    assert len(observations) == 1
    assert observations.pseudorange_m.iloc[0] == 2.4e7 + 83
    assert np.isnan(observations.cn0_dbhz.iloc[0])  # Never fabricate unavailable signal strength.
