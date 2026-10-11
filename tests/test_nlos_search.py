import json

import joblib
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')
from navdiag.nlos import CLASSIC_DEFAULTS, NLOS_FEATURES
from navdiag.nlos_neural import NEURAL_MODELS, build_network
from navdiag.nlos_search import tune_nlos, validate_search_plan


def test_search_plan_rejects_duplicate_and_invalid_candidates():
    plans = {name: [{'id': 'one', 'parameters': {}}] for name in [*CLASSIC_DEFAULTS, *NEURAL_MODELS]}
    plan = {'neural_epochs': 2, 'models': plans}
    assert validate_search_plan(plan)['models']['cnn'][0]['parameters']['window'] == 32
    plans['mlp'].append({'id': 'duplicate_effective', 'parameters': {'window': 32}})
    with pytest.raises(ValueError, match='Duplicate effective'):
        validate_search_plan(plan)
    plans['mlp'] = [{'id': 'invalid', 'parameters': {'learning_rate': 0}}]
    with pytest.raises(ValueError, match='Invalid dropout'):
        validate_search_plan(plan)


@pytest.mark.parametrize('name', NEURAL_MODELS)
def test_nondefault_network_shapes_support_short_history(name):
    network = build_network(name, 5, window=1, config={
        'width': 16, 'layers': 1, 'heads': 2, 'dropout': 0.2, 'kernel_size': 3}).eval()
    with torch.no_grad():
        result = network(torch.ones(3, 1, 5))
    assert result.shape == (3,) and torch.isfinite(result).all()


def test_search_is_blind_to_test_labels_features_and_freezes_before_evaluation(tmp_path, monkeypatch):
    import navdiag.nlos_search as search
    rng = np.random.default_rng(9)
    parts = []
    for group in ['train', 'validation', 'test']:
        part = pd.DataFrame({
            'trip_id': group, 'scene_id': group, 'device_id': 'D', 'data_kind': 'real',
            'timestamp_ms': np.repeat(np.arange(12) * 1000, 2),
            'satellite_id': np.tile(['G01', 'G02'], 12),
            'signal_label': np.tile(['los', 'nlos'], 12),
            'signal_label_source': 'dataset_annotation',
        })
        for feature in NLOS_FEATURES:
            part[feature] = rng.uniform(1, 50, len(part))
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    csv = tmp_path / 'satellites.csv'
    frame.to_csv(csv, index=False)
    manifest = {name: [name] for name in ['train', 'validation', 'test']}
    proposals = {
        'logistic_regression': [{'id': 'a', 'parameters': {'C': .1}}, {'id': 'b', 'parameters': {'C': 1.}}],
        'random_forest': [{'id': 'a', 'parameters': {'n_estimators': 5, 'max_depth': 2}},
                          {'id': 'b', 'parameters': {'n_estimators': 5, 'max_depth': 4}}],
    }
    for name in NEURAL_MODELS:
        proposals[name] = [
            {'id': 'a', 'parameters': {'window': 4, 'width': 16, 'layers': 1}},
            {'id': 'b', 'parameters': {'window': 8, 'width': 16, 'layers': 2}},
        ]
    plan = {'neural_epochs': 2, 'models': proposals}
    actual_training = search.train_neural
    actual_final = search.train_nlos

    def guarded_training(train, validation, *args, **kwargs):
        assert set(train.trip_id) == {'train'} and set(validation.trip_id) == {'validation'}
        return actual_training(train, validation, *args, **kwargs)

    def guarded_final(path, output, *args, **kwargs):
        frozen = json.loads((output / 'selection_before_test.json').read_text(encoding='utf-8'))
        assert frozen['selection_data'] == 'validation only'
        trace = json.loads((output / 'search_trace.json').read_text(encoding='utf-8'))
        assert trace['status'] == 'selection_frozen_before_test'
        assert all('test' not in trial for trial in trace['trials'])
        return actual_final(path, output, *args, **kwargs)

    monkeypatch.setattr(search, 'train_neural', guarded_training)
    monkeypatch.setattr(search, 'train_nlos', guarded_final)
    first = tune_nlos(csv, tmp_path / 'first', manifest, plan)
    # Change all held-out labels and values, retaining keys/splits only.
    frame.loc[frame.trip_id == 'test', 'signal_label'] = 'los'
    frame.loc[frame.trip_id == 'test', NLOS_FEATURES] = 100000.
    frame.to_csv(csv, index=False)
    second = tune_nlos(csv, tmp_path / 'second', manifest, plan)
    assert first['hyperparameter_search']['selected_candidates'] == second['hyperparameter_search']['selected_candidates']
    assert first['selected_model'] == second['selected_model']
    a = joblib.load(tmp_path / 'first/models.joblib')
    b = joblib.load(tmp_path / 'second/models.joblib')
    for name in NEURAL_MODELS:
        for key, value in a['models'][name].network.state_dict().items():
            torch.testing.assert_close(value, b['models'][name].network.state_dict()[key], atol=0, rtol=0)
