"""Align supervised/scored epochs with a complete, observation-only causal history."""
import numpy as np
import pandas as pd

from navdiag.data import FEATURES


def context_indices(scored: pd.DataFrame, context: pd.DataFrame) -> np.ndarray:
    """Return positions in context, rejecting duplicate, missing or changed observations.

    Reference columns are deliberately ignored. Extra observation epochs may have no labels.
    """
    keys = ['trip_id', 'timestamp_ms']
    if 'device_id' in scored or 'device_id' in context:
        keys.insert(1, 'device_id')
    for frame in (scored, context):
        if not set(keys) <= set(frame) or frame[keys].isna().any().any():
            raise ValueError('Context requires nonmissing epoch keys')
        if frame.duplicated(keys).any():
            raise ValueError('Duplicate context epoch keys')
    lookup = pd.MultiIndex.from_frame(context[keys])
    indices = lookup.get_indexer(pd.MultiIndex.from_frame(scored[keys]))
    if (indices < 0).any():
        raise ValueError('Scored epochs are missing from observation context')
    shared = [name for name in [*FEATURES, 'wls_e_m', 'wls_n_m'] if name in scored]
    if not set(shared) <= set(context):
        raise ValueError('Context is missing observation columns')
    if not np.allclose(scored[shared].to_numpy(float), context.iloc[indices][shared].to_numpy(float),
                       rtol=0, atol=1e-8, equal_nan=True):
        raise ValueError('Context observations differ from scored epochs')
    return indices
