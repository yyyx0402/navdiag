import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from navdiag.data import split_by_trip
from navdiag.nlos import validate_nlos_protocol
from navdiag.nlos_protocols import protocol_manifests, summarize_protocols


ROOT = Path(__file__).resolve().parents[1]


def acquisition_metadata():
    plan = json.loads((ROOT / 'config/nlos_protocols.json').read_text())
    runs = plan['folds'][0]['in_domain_train'] + plan['folds'][0]['validation'] + plan['folds'][0]['test']
    return pd.DataFrame({'trip_id': runs, 'scene_id': [r.split('_')[1] for r in runs]}), plan


def test_paired_protocols_preserve_complete_acquisitions_and_identical_holdouts():
    frame, plan = acquisition_metadata()
    specs = protocol_manifests(frame, plan)
    for within, cross in zip(specs[::2], specs[1::2]):
        assert within['manifest']['test'] == cross['manifest']['test']
        assert within['manifest']['validation'] == cross['manifest']['validation']
        assert set(cross['manifest']['train']) < set(within['manifest']['train'])
        excluded = set(within['manifest']['train']) - set(cross['manifest']['train'])
        assert all(within['test_area'] in run for run in excluded)
        for spec in [within, cross]:
            included = sum(spec['manifest'].values(), [])
            parts = split_by_trip(frame[frame.trip_id.isin(included)], manifest=spec['manifest'])
            validate_nlos_protocol(parts, spec['protocol'])
    bad = copy.deepcopy(plan)
    bad['folds'][0]['validation'][0] = bad['folds'][0]['test'][0]
    with pytest.raises(ValueError, match='exactly once'):
        protocol_manifests(frame, bad)


def test_protocol_checks_reject_target_area_leakage_and_unseen_in_domain_tests():
    parts = {name: pd.DataFrame({'scene_id': areas}) for name, areas in {
        'train': ['A', 'B'], 'validation': ['A', 'B'], 'test': ['C']}.items()}
    validate_nlos_protocol(parts, 'cross_region')
    with pytest.raises(ValueError, match='represented'):
        validate_nlos_protocol(parts, 'in_domain')
    parts['test'] = pd.DataFrame({'scene_id': ['A']})
    validate_nlos_protocol(parts, 'in_domain')
    with pytest.raises(ValueError, match='absent'):
        validate_nlos_protocol(parts, 'cross_region')


def test_summary_selection_cannot_be_changed_by_test_scores():
    reports = {}
    for protocol in ['in_domain', 'cross_region']:
        for area in ['A', 'B', 'C']:
            reports[f'{protocol}_{area}'] = {'protocol': protocol, 'experiment': {'test_area': area},
                'models': {name: {'validation': {'f1': val},
                                  'test': {metric: test for metric in ['precision', 'recall', 'f1', 'accuracy', 'average_precision']}}
                           for name, val, test in [('forest', .8, .7), ('network', .7, .9)]}}
    summary = summarize_protocols(reports)
    for p in summary['protocols'].values():
        assert p['selected_model'] == 'forest'
        assert p['observed_test_order'][0] == 'network'
    for r in reports.values():
        r['models']['network']['test'] = {m: 0. for m in r['models']['network']['test']}
    changed = summarize_protocols(reports)
    assert all(p['selected_model'] == 'forest' for p in changed['protocols'].values())
    reports['cross_region_A']['models']['network']['test']['precision'] = None
    missing = summarize_protocols(reports)
    assert missing['protocols']['cross_region']['models']['network']['test_macro']['precision'] is None
    assert missing['cross_minus_in_domain']['network']['precision'] is None


def test_timestamp_alignment_rejects_shifted_duplicate_and_missing_label_epochs():
    pytest.importorskip('pyrtklib')
    spec = importlib.util.spec_from_file_location('klt_prepare_adapter', ROOT / 'tools/klt/prepare.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    times = np.arange(1000., 1006.)
    gps = times[1:-1] + 18. - .0001
    assert module.check_timestamp_alignment(times, gps)['epochs'] == 4
    assert module.check_timestamp_alignment(times + .005, gps)['max_time_difference_ms'] == pytest.approx(5.1)
    for invalid in [gps + .2, np.array([1019., 1019.]), gps + 20.]:
        with pytest.raises(ValueError):
            module.check_timestamp_alignment(times, invalid)
