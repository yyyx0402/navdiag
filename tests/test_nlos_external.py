import numpy as np
import pandas as pd
import pytest

from navdiag.nlos import NLOS_FEATURES
from navdiag.nlos_external import area_manifests, external_evaluation


def test_all_district_permutations_keep_target_out_of_training_and_validation():
    frame = pd.DataFrame({'trip_id': ['runA', 'runB', 'runC'], 'scene_id': ['A', 'B', 'C']})
    specs = area_manifests(frame, ['A', 'B', 'C'])
    assert len(specs) == 6
    for spec in specs:
        assert len(set(sum(spec['manifest'].values(), []))) == 3
        assert spec['test_area'] not in [spec['train_area'], spec['validation_area']]
    bad = pd.concat([frame, pd.DataFrame({'trip_id': ['runD'], 'scene_id': ['A']})])
    with pytest.raises(ValueError, match='one whole acquisition'):
        area_manifests(bad, ['A', 'B', 'C'])


def test_external_evaluation_keeps_source_thresholds_and_excludes_unknown(monkeypatch):
    frame = pd.DataFrame({'scene_id': ['A'] * 3, 'trip_id': ['runA'] * 3,
                          'timestamp_ms': [0, 1000, 2000], 'signal_label': ['los', 'nlos', 'unknown']})
    monkeypatch.setattr('navdiag.nlos_external.nlos_scores', lambda *args: np.array([.55, .7, .99]))
    bundle = {'features': NLOS_FEATURES, 'model_features': {'forest': NLOS_FEATURES}}
    source = {'models': {'forest': {'validation': {'threshold': .6}, 'test': {'f1': .8}}},
              'selected_model': 'forest', 'splits': {}, 'protocol': 'source'}
    report, predictions = external_evaluation(frame, bundle, source)
    metrics = report['models']['forest']['per_area']['A']
    assert report['models']['forest']['threshold'] == .6
    assert metrics['f1'] == 1 and metrics['rows'] == 2 and metrics['unknown_rows'] == 1
    assert len(predictions) == 3
    bundle['features'] = ['reference_elevation_deg']
    with pytest.raises(ValueError, match='truth'):
        external_evaluation(frame, bundle, source)
