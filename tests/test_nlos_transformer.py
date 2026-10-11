import joblib
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')
from navdiag.nlos import NLOS_FEATURES, nlos_scores, train_nlos
from navdiag.nlos_neural import NEURAL_MODELS
from navdiag.nlos_transformer import NLOSTransformer, satellite_windows


def test_satellite_windows_isolate_keys_gaps_order_and_future():
    frame = pd.DataFrame({
        'trip_id': ['A', 'A', 'A', 'A', 'A', 'B', 'A'],
        'device_id': ['D', 'D', 'D', 'D', 'D', 'D', 'other'],
        'satellite_id': ['G01', 'G02', 'G01', 'G01', 'G01', 'G01', 'G01'],
        'timestamp_ms': [1000, 1000, 0, 2000, 10000, 2000, 2000],
    }, index=[7, 9, 8, 1, 3, 2, 4])
    values = np.array([[2], [99], [1], [3], [4], [88], [77]], dtype=np.float32)
    windows = satellite_windows(frame, values, 3)
    np.testing.assert_array_equal(windows[0, :, 0], [1, 1, 2])
    np.testing.assert_array_equal(windows[3, :, 0], [1, 2, 3])
    np.testing.assert_array_equal(windows[4, :, 0], [4, 4, 4])
    for index, value in [(1, 99), (2, 1), (5, 88), (6, 77)]:
        np.testing.assert_array_equal(windows[index, :, 0], [value] * 3)
    values[3:, 0] = -1000
    np.testing.assert_array_equal(satellite_windows(frame, values, 3)[:3], windows[:3])


def test_attention_cannot_read_future_and_warmup_is_finite():
    torch.manual_seed(8)
    network = NLOSTransformer(features=5, window=8).eval()
    inputs = torch.randn(2, 8, 5)
    with torch.no_grad():
        before = network.encode(inputs)
        inputs[:, 4:] += 100
        after = network.encode(inputs)
        torch.testing.assert_close(before[:, :4], after[:, :4], atol=1e-6, rtol=1e-6)
        assert torch.isfinite(network(torch.zeros(1, 8, 5))).all()


def test_five_neural_models_persist_and_never_select_using_test_labels(tmp_path):
    rng = np.random.default_rng(3)
    parts = []
    for area in ['train', 'val', 'test']:
        part = pd.DataFrame({
            'trip_id': area, 'device_id': 'D', 'scene_id': area, 'data_kind': 'real',
            'timestamp_ms': np.repeat(np.arange(16) * 1000, 2),
            'satellite_id': np.tile(['G01', 'G02'], 16),
            'signal_label': np.tile(['los', 'nlos'], 16),
            'signal_label_source': 'dataset_annotation',
        })
        for feature in NLOS_FEATURES:
            part[feature] = rng.uniform(1, 50, len(part))
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    frame.loc[0, 'signal_label'] = 'unknown'
    path = tmp_path / 'satellites.csv'
    frame.to_csv(path, index=False)
    manifest = {'train': ['train'], 'validation': ['val'], 'test': ['test']}
    report = train_nlos(path, tmp_path / 'first', manifest, with_neural_models=True, neural_epochs=2)
    bundle = joblib.load(tmp_path / 'first/models.joblib')
    assert 'rf_signal_geometry' not in report['models']
    assert report['splits']['train']['excluded_unknown'] == 1
    assert tuple(report['neural_models']) == NEURAL_MODELS
    assert set(NEURAL_MODELS) <= set(report['models'])
    assert report['best_neural_model'] == max(NEURAL_MODELS, key=lambda name: report['models'][name]['validation']['f1'])
    pred = pd.read_csv(tmp_path / 'first/predictions.csv').sample(frac=1, random_state=4)
    observed = pred.drop(columns=['signal_label', 'signal_label_source', 'scene_id', 'data_kind'])
    prefix = observed[observed.timestamp_ms < 5000]
    for name in NEURAL_MODELS:
        model = bundle['models'][name]
        np.testing.assert_allclose(model.imputer.statistics_, frame.loc[frame.trip_id == 'train', NLOS_FEATURES].median())
        scores = nlos_scores(bundle, observed, name)
        assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
        np.testing.assert_allclose(scores, pred[f'nlos_score_{name}'], atol=1e-6)
        np.testing.assert_allclose(nlos_scores(bundle, prefix, name), scores[observed.timestamp_ms < 5000], atol=1e-6)
    # Changing held-out labels cannot change trained weights, threshold or winner.
    frame.loc[frame.trip_id == 'test', 'signal_label'] = 'los'
    frame.to_csv(path, index=False)
    second = train_nlos(path, tmp_path / 'second', manifest, with_neural_models=True, neural_epochs=2)
    second_bundle = joblib.load(tmp_path / 'second/models.joblib')
    assert second['selected_model'] == report['selected_model']
    for name in NEURAL_MODELS:
        assert second['models'][name]['validation'] == report['models'][name]['validation']
        assert second['models'][name]['validation_threshold_tradeoff'] == report['models'][name]['validation_threshold_tradeoff']
        for key, value in bundle['models'][name].network.state_dict().items():
            torch.testing.assert_close(value, second_bundle['models'][name].network.state_dict()[key], atol=0, rtol=0)
