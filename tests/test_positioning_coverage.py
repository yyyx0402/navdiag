import numpy as np
import pandas as pd
import pytest

from navdiag.availability import availability_summary, coverage_position_metrics
from navdiag.context import context_indices
from navdiag.fusion import _apply


def test_dashboard_and_evidence_expose_raw_coverage_and_matching_doppler_cohort(tmp_path):
    import hashlib
    import json
    import threading
    from urllib.request import urlopen
    from navdiag.agent import EvidenceTools
    from navdiag.dashboard import make_server
    from navdiag.data import generate_demo
    from navdiag.learning import train_and_evaluate
    from navdiag.fusion import position_metrics

    directory = tmp_path / 'run'
    report = train_and_evaluate(generate_demo(tmp_path / 'data.csv', trips=5, epochs=10), directory)
    predictions = pd.read_csv(directory / 'predictions.csv')
    trip = predictions.trip_id.iloc[0]
    metric = position_metrics(predictions.horizontal_error_m.to_numpy())
    report['positioning'] = {'methods': {'wls': metric}}
    report['coverage'] = {'test': {trip: {'raw_epochs': 11, 'solution_epochs': 10, 'scored_epochs': 10}}}
    report['baseline_solver'] = 'fixture SPP'
    report['evaluation_limitations'] = ['Test fixture, not real GNSS']
    report_path = directory / 'report.json'
    report_path.write_text(json.dumps(report), encoding='utf-8')
    digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    doppler = {'policy': {'limitations': ['Independent approximate velocity baseline']}, 'folds': [{
        'id': 'fixture', 'test_trip': trip, 'source_report_sha256': digest, 'metrics': metric,
        'diagnostics_on_all_successful_fixes': {'epochs': 10}}]}
    (directory / 'doppler_report.json').write_text(json.dumps(doppler), encoding='utf-8')
    additional = predictions[['trip_id', 'device_id', 'timestamp_ms']].copy() if 'device_id' in predictions else predictions[['trip_id', 'timestamp_ms']].copy()
    if 'device_id' not in predictions:
        predictions['device_id'] = 'fixture'
        predictions.to_csv(directory / 'predictions.csv', index=False)
        additional['device_id'] = 'fixture'
    additional[['doppler_filter_e_m', 'doppler_filter_n_m']] = 0.
    additional['doppler_filter_error_m'] = predictions.horizontal_error_m
    additional.to_csv(directory / 'doppler_predictions.csv', index=False)
    # The raw acquisition starts one second before the first scorable fix.
    times = [predictions.timestamp_ms.min() - 1000, *predictions.timestamp_ms.tolist()]
    pd.DataFrame({'trip_id': trip, 'timestamp_ms': times, 'solution_available': [False, *([True] * 10)],
                  'reference_time_matched': True, 'reference_quality_allowed': True,
                  'scorable': [False, *([True] * 10)]}).to_csv(directory / 'availability.csv', index=False)
    server = make_server(directory, port=0, connections_file=tmp_path / 'connections.json')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(base + '/api/positioning/availability') as response:
            body = json.load(response)
            assert len(body['points']) == 11 and body['points'][0]['solution_available'] is False
            assert body['coverage']['raw_epochs'] == 11
        with urlopen(base + '/api/trajectory') as response:
            body = json.load(response)
            assert body['points'][0]['time_s'] == 1
            assert 'doppler_filter_error_m' in body['points'][0]
        with urlopen(base + '/api/report') as response:
            assert 'doppler_filter' in json.load(response)['positioning']['methods']
        with urlopen(base + '/api/positioning/rotation') as response:
            assert json.load(response)['available'] is False
        evidence = EvidenceTools(directory).invoke('compare_positioning', {})['payload']
        assert evidence['test_coverage'][trip]['raw_epochs'] == 11
        assert evidence['methods']['doppler_filter']['epochs'] == 10
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)
    doppler['folds'][0]['source_report_sha256'] = '0' * 64
    (directory / 'doppler_report.json').write_text(json.dumps(doppler), encoding='utf-8')
    with pytest.raises(ValueError, match='does not match'):
        make_server(directory, port=0, connections_file=tmp_path / 'connections.json')
    with pytest.raises(ValueError, match='does not match'):
        EvidenceTools(directory).invoke('compare_positioning', {})


def test_missing_solutions_do_not_disappear_from_accuracy_denominator():
    events = pd.DataFrame({'timestamp_ms': np.arange(6) * 1000,
        'solution_available': [1, 0, 0, 0, 1, 1], 'reference_time_matched': [1] * 6,
        'reference_quality_allowed': [1, 1, 1, 1, 1, 0], 'scorable': [1, 0, 0, 0, 1, 0]})
    result = availability_summary(events)
    assert result['solution_availability'] == 0.5
    assert result['eligible_reference_epochs'] == 5
    assert result['solution_availability_on_eligible_reference'] == 0.4
    assert result['longest_solution_outage_s'] == 3
    accuracy = coverage_position_metrics(np.array([1., 3.]), 5)
    assert accuracy['within_error_limits_m']['2'] == 0.5
    assert accuracy['available_and_within_error_limits_m']['2'] == 0.2
    with pytest.raises(ValueError):
        coverage_position_metrics(np.array([1., 3.]), 1)
    events.loc[1, 'scorable'] = 1
    with pytest.raises(ValueError, match='Scorable'):
        availability_summary(events)


def test_unlabelled_observation_history_survives_reference_scoring_mask():
    context = pd.DataFrame({'trip_id': ['a'] * 4, 'timestamp_ms': [0, 1000, 2000, 3000],
                            'wls_e_m': [0., 5., 8., 9.], 'wls_n_m': [0.] * 4})
    scored = context.iloc[[3, 0]].copy()
    ids = context_indices(scored, context)
    np.testing.assert_array_equal(ids, [3, 0])
    full = _apply(context, np.ones(4), 1)
    # Changing reference labels cannot change the observation-only filtering history.
    context['horizontal_error_m'] = 999999.
    context['gt_e_m'] = -999999.
    np.testing.assert_array_equal(full[ids], _apply(context, np.ones(4), 1)[ids])
    removed = _apply(context.iloc[[0, 3]], np.ones(2), 1)
    assert not np.allclose(full[3], removed[1])
    with pytest.raises(ValueError, match='missing'):
        context_indices(scored, context.iloc[:3])
    altered = scored.copy()
    altered.loc[3, 'wls_e_m'] = 100.
    with pytest.raises(ValueError, match='differ'):
        context_indices(altered, context)
    with pytest.raises(ValueError, match='Duplicate'):
        context_indices(scored, pd.concat([context, context.iloc[:1]]))
