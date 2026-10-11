import json
import shutil
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd
import pytest

from navdiag.agent import EvidenceTools, ask
from navdiag.dashboard import make_server
from navdiag.data import generate_demo
from navdiag.learning import train_and_evaluate
from navdiag.nlos import NLOS_FEATURES, cn0_variability, nlos_scores, train_nlos, validate_observations


def test_signal_variability_ignores_zero_sentinels_and_is_causal_across_gaps():
    frame = pd.DataFrame({'trip_id': 'trip', 'device_id': 'receiver', 'satellite_id': 'G01',
                          'timestamp_ms': [0, 1000, 2000, 3000, 10000],
                          'cn0_dbhz': [30., 32., 0., 34., 40.]})
    values = cn0_variability(frame)
    assert np.isnan(values.iloc[0]) and np.isnan(values.iloc[-1])
    assert values.iloc[2] == pytest.approx(np.std([30, 32], ddof=1))
    assert values.iloc[3] == pytest.approx(np.std([30, 32, 34], ddof=1))
    frame.loc[3:, 'cn0_dbhz'] = 5.
    np.testing.assert_allclose(cn0_variability(frame).iloc[:3], values.iloc[:3], equal_nan=True)


@pytest.fixture(scope='module')
def nlos_experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp('nlos')
    rng = np.random.default_rng(7)
    parts = []
    for area in ['A', 'B', 'C']:
        n = 80
        part = pd.DataFrame({
            'trip_id': f'run_{area}', 'scene_id': area, 'device_id': 'fixture',
            'timestamp_ms': np.repeat(np.arange(20) * 1000, 4),
            'satellite_id': np.tile(['G01', 'G02', 'C01', 'C02'], 20),
            'system': np.tile(['G', 'G', 'C', 'C'], 20), 'data_kind': 'real',
            'signal_label': np.tile(['los', 'nlos'], n // 2),
            'signal_label_source': 'dataset_annotation',
        })
        for feature in NLOS_FEATURES:
            part[feature] = rng.uniform(1, 50, n)
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    path = root / 'satellites.csv'
    frame.to_csv(path, index=False)
    manifest = {'train': ['run_A'], 'validation': ['run_B'], 'test': ['run_C']}
    report = train_nlos(path, root / 'klt_nlos', manifest)
    train_and_evaluate(generate_demo(root / 'epochs.csv', trips=5, epochs=60), root / 'epoch_risk')
    return root, frame, manifest, report


def test_nlos_training_keeps_scene_groups_and_selects_on_validation(nlos_experiment):
    root, frame, manifest, report = nlos_experiment
    assert report['task_type'] == 'satellite_nlos'
    winner = report['selected_model']
    assert report['models'][winner]['validation']['f1'] == max(m['validation']['f1'] for m in report['models'].values())
    assert report['models'][winner]['test']['rows'] == 80
    assert report['source']['description'].startswith('User-provided')
    assert set(report['features']) == set(NLOS_FEATURES)
    assert not set(report['features']) & {'signal_label', 'scene_id', 'satellite_id'}
    overlapping = frame.copy()
    overlapping.loc[overlapping.trip_id == 'run_B', 'scene_id'] = 'A'
    overlapping.to_csv(root / 'overlap.csv', index=False)
    with pytest.raises(ValueError, match='Scene groups overlap'):
        train_nlos(root / 'overlap.csv', root / 'invalid', manifest)


def test_position_error_and_unverified_causes_cannot_become_nlos_truth(nlos_experiment):
    _, frame, _, _ = nlos_experiment
    invalid = frame.copy()
    invalid['signal_label_source'] = 'position_error_over_10m'
    with pytest.raises(ValueError, match='independent label provenance'):
        validate_observations(invalid)
    invalid = frame.copy()
    invalid.loc[0, 'signal_label'] = 'building'
    with pytest.raises(ValueError, match='Signal labels'):
        validate_observations(invalid)
    duplicate = pd.concat([frame, frame.iloc[[0]]])
    with pytest.raises(ValueError, match='Duplicate satellite'):
        validate_observations(duplicate)


def test_nlos_inference_needs_no_camera_truth_or_scene_and_preserves_order(nlos_experiment):
    root, _, _, _ = nlos_experiment
    pred = pd.read_csv(root / 'klt_nlos/predictions.csv').sample(frac=1, random_state=2)
    bundle = joblib.load(root / 'klt_nlos/models.joblib')
    observed = pred.drop(columns=['signal_label', 'signal_label_source', 'scene_id', 'data_kind'])
    np.testing.assert_allclose(nlos_scores(bundle, observed), pred.predicted_nlos_score)
    invalid = observed.copy()
    invalid.loc[invalid.index[0], 'cn0_dbhz'] = np.inf
    with pytest.raises(ValueError, match='Non-finite satellite feature'):
        nlos_scores(bundle, invalid)


def test_modified_prepared_table_cannot_keep_stale_provenance(nlos_experiment, tmp_path):
    _, frame, manifest, _ = nlos_experiment
    frame.to_csv(tmp_path / 'satellites.csv', index=False)
    (tmp_path / 'preprocess_report.json').write_text(json.dumps({'prepared_sha256': 'incorrect'}))
    with pytest.raises(ValueError, match='provenance hash'):
        train_nlos(tmp_path / 'satellites.csv', tmp_path / 'invalid', manifest)


def test_nlos_evidence_is_separate_from_epoch_risk_and_keeps_real_provenance(nlos_experiment):
    root, _, _, _ = nlos_experiment
    tools = EvidenceTools(root / 'epoch_risk')
    evidence = tools.invoke('compare_nlos', {})
    assert evidence['data_kind'] == 'real'
    assert evidence['payload']['reference_labels_used'] is True
    result = ask('遮挡识别模型效果如何？', root / 'epoch_risk')
    assert result['evidence'][0]['tool'] == 'compare_nlos'
    assert '真实卫星 LOS/NLOS' in result['answer']
    assert '不区分高楼' in result['answer']
    missing = EvidenceTools(root / 'epoch_risk', root / 'missing').invoke('compare_nlos', {})
    assert missing['payload']['available'] is False


def test_diagnostic_task_context_routes_generic_metrics_to_the_correct_task(nlos_experiment):
    root, _, _, report = nlos_experiment
    satellite = ask('比较模型的 F1', root / 'epoch_risk', task_context='satellite_nlos')
    assert satellite['evidence'][0]['tool'] == 'compare_nlos'
    assert satellite['evidence'][0]['data_kind'] == 'real'
    assert '当前是模拟实验' not in satellite['answer']
    assert 'random_forest' in satellite['answer'] and 'logistic_regression' in satellite['answer']
    position = ask('比较模型的 F1', root / 'epoch_risk', task_context='position_error')
    assert position['evidence'][0]['tool'] == 'compare_models'
    assert position['evidence'][0]['data_kind'] == 'synthetic'
    counts = ask('当前数据如何划分？', root / 'epoch_risk', task_context='satellite_nlos')
    assert f"{report['splits']['test']['rows']} 条卫星观测" in counts['answer']
    with pytest.raises(ValueError, match='Task context'):
        ask('question', root / 'epoch_risk', task_context='other')


def test_cli_evidence_can_load_the_current_protocol_suite_and_explicit_historical_fold(nlos_experiment, tmp_path):
    root, _, _, report = nlos_experiment
    suite = tmp_path / 'klt_protocols'
    folder = suite / 'folds' / 'in_domain_A'
    folder.mkdir(parents=True)
    changed = {**report, 'experiment': {'id': 'in_domain_A'}}
    (folder / 'report.json').write_text(json.dumps(changed), encoding='utf-8')
    (suite / 'suite_report.json').write_text(json.dumps({'task_type': 'satellite_nlos_suite',
        'experiments': [{'id': 'in_domain_A'}], 'protocols': {}, 'limitations': []}), encoding='utf-8')
    result = EvidenceTools(root / 'epoch_risk', suite).invoke('compare_nlos', {})
    assert result['payload']['experiment']['id'] == 'in_domain_A'
    assert result['payload']['comparison_test_sets_shared'] is True
    assert all('training_history' not in model for model in result['payload']['models'].values())
    historical = EvidenceTools(root / 'epoch_risk', root / 'klt_nlos').invoke('compare_nlos', {})
    assert 'experiment' not in historical['payload']
    position = tmp_path / 'epoch_risk'
    position.mkdir()
    for name in ['report.json', 'predictions.csv']:
        shutil.copyfile(root / 'epoch_risk' / name, position / name)
    current = EvidenceTools(position).invoke('compare_nlos', {})
    assert current['payload']['experiment']['id'] == 'in_domain_A'


def test_dashboard_nlos_rows_show_prediction_and_independent_reference(nlos_experiment):
    root, _, _, _ = nlos_experiment
    generalization = root / 'generalization'
    generalization.mkdir(exist_ok=True)
    (generalization / 'generalization_report.json').write_text(json.dumps(
        {'task_type': 'satellite_nlos_generalization', 'external_transfer': {'rows': 123},
         'district_holdout': {}, 'limitations': []}))
    server = make_server(root / 'epoch_risk', port=0, nlos_artifacts=root / 'klt_nlos',
                         generalization_artifacts=generalization)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(base + '/api/nlos/report') as response:
            assert json.load(response)['report']['task_type'] == 'satellite_nlos'
        with urlopen(base + '/api/nlos/generalization') as response:
            result = json.load(response)
            assert result['available'] and result['report']['external_transfer']['rows'] == 123
        with urlopen(base + '/api/nlos/epoch?trip=run_C&index=1') as response:
            epoch = json.load(response)
            assert epoch['epoch_index'] == 1 and len(epoch['satellites']) == 4
            assert epoch['satellites'][0]['signal_label_source'] == 'dataset_annotation'
        body = json.dumps({'question': '比较模型的 F1', 'provider': 'offline',
                           'task_context': 'satellite_nlos'}).encode('utf-8')
        with urlopen(Request(base + '/api/ask', body, {'Content-Type': 'application/json'})) as response:
            assert json.load(response)['evidence'][0]['tool'] == 'compare_nlos'
        body = json.dumps({'question': 'question', 'task_context': 'unknown'}).encode('utf-8')
        with pytest.raises(HTTPError) as caught:
            urlopen(Request(base + '/api/ask', body, {'Content-Type': 'application/json'}))
        assert caught.value.code == 400
        for query in ['trip=unknown', 'trip=run_C&index=-1', 'trip=run_C&index=bad']:
            with pytest.raises(HTTPError) as caught:
                urlopen(base + '/api/nlos/epoch?' + query)
            assert caught.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_dashboard_protocol_selection_routes_reports_predictions_and_rejects_unknown_ids(nlos_experiment, tmp_path):
    root, _, _, report = nlos_experiment
    suite = tmp_path / 'protocols'
    for name in ['in_domain_A', 'cross_region_A']:
        folder = suite / 'folds' / name
        folder.mkdir(parents=True)
        shutil.copyfile(root / 'klt_nlos/report.json', folder / 'report.json')
        shutil.copyfile(root / 'klt_nlos/predictions.csv', folder / 'predictions.csv')
    cross = pd.read_csv(suite / 'folds/cross_region_A/predictions.csv')
    cross['predicted_nlos_score'] = .123
    cross.to_csv(suite / 'folds/cross_region_A/predictions.csv', index=False)
    changed = {**report, 'selected_model': 'cn0_rule'}
    (suite / 'folds/cross_region_A/report.json').write_text(json.dumps(changed))
    (suite / 'suite_report.json').write_text(json.dumps({'task_type': 'satellite_nlos_suite',
        'experiments': [{'id': 'in_domain_A'}, {'id': 'cross_region_A'}]}))
    server = make_server(root / 'epoch_risk', port=0, nlos_artifacts=suite)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(base + '/api/nlos/report?experiment=cross_region_A') as response:
            result = json.load(response)
            assert result['report']['selected_model'] == 'cn0_rule'
            assert result['experiment_id'] == 'cross_region_A'
        with urlopen(base + '/api/nlos/epoch?experiment=cross_region_A&trip=run_C&index=0') as response:
            assert all(s['predicted_nlos_score'] == .123 for s in json.load(response)['satellites'])
        with pytest.raises(HTTPError) as caught:
            urlopen(base + '/api/nlos/report?experiment=../other')
        assert caught.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
