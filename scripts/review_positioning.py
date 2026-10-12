"""Replay trusted local positioning artifacts; no training, tuning or model API calls."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from navdiag.availability import availability_summary
from navdiag.context import context_indices
from navdiag.fusion import _apply
from navdiag.learning import evaluate, predict_frame

ROOT = Path(__file__).resolve().parents[1]


def review(data, output):
    rotation = json.loads((output / 'positioning_rotation_report.json').read_text(encoding='utf-8'))
    observations, events = (pd.read_csv(data / name) for name in ['observations.csv', 'availability.csv'])
    checks = 0
    for fold in rotation['folds']:
        folder = (output / fold['report_path']).parent
        report = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
        frame = pd.read_csv(folder / 'predictions.csv')
        # Only load the experiment's own trusted local bundle.
        bundle = joblib.load(folder / 'models.joblib')
        context = observations[observations.trip_id.isin(frame.trip_id.unique())]
        ids = context_indices(frame, context)
        for name in report['models']:
            risk = predict_frame(bundle, context, name)[ids]
            np.testing.assert_allclose(risk, frame['risk_' + name], atol=1e-7)
            metrics = evaluate(frame.is_unreliable.to_numpy(), risk, report['models'][name]['test']['threshold'])
            assert metrics['confusion_matrix'] == report['models'][name]['test']['confusion_matrix']
            checks += 2
        position = report['positioning']
        replay = _apply(context, np.full(len(context), position['fixed_sigma_m']**2), position['acceleration_noise_mps2'])[ids]
        np.testing.assert_allclose(replay, frame[['fixed_filter_e_m', 'fixed_filter_n_m']], atol=1e-7)
        risk = predict_frame(bundle, context)
        variances = np.array(position['risk_bin_variances_m2'])
        bins = np.clip(np.searchsorted(position['risk_bin_edges'], risk, side='right') - 1, 0, len(variances) - 1)
        replay = _apply(context, variances[bins], position['acceleration_noise_mps2'])[ids]
        np.testing.assert_allclose(replay, frame[['adaptive_filter_e_m', 'adaptive_filter_n_m']], atol=1e-7)
        assert availability_summary(events[events.trip_id == fold['test_trip']]) == fold['coverage']
        assert report['selected_model'] == max(report['models'], key=lambda name: report['models'][name]['validation']['f1'])
        checks += 4
        frozen = json.loads((folder / 'frozen_plan.json').read_text(encoding='utf-8'))
        for path, digest in frozen['files'].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
            checks += 1
    print(f'Expanded experiment replay: {checks} checks passed; full-context scores, confusion counts, filters, coverage and frozen inputs agree.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/processed/urbannav_positioning_v3')
    parser.add_argument('--artifacts', type=Path, default=ROOT / 'artifacts/urbannav_positioning')
    args = parser.parse_args()
    review(args.data, args.artifacts)
