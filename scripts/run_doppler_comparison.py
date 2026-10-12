"""Evaluate a literature-inspired Doppler baseline on the unchanged three scored cohorts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from navdiag.availability import coverage_position_metrics
from navdiag.context import context_indices
from navdiag.doppler import filter_doppler_positions

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply(frame, sigma, noise, policy):
    cv = np.empty((len(frame), 2, 2))
    cv[:, 0, 0] = frame.doppler_ee_m2ps2
    cv[:, 1, 1] = frame.doppler_nn_m2ps2
    cv[:, 0, 1] = cv[:, 1, 0] = frame.doppler_en_m2ps2
    return filter_doppler_positions(frame.timestamp_ms.to_numpy(), frame[['wls_e_m', 'wls_n_m']].to_numpy(),
        frame[['doppler_e_mps', 'doppler_n_mps']].to_numpy(), cv, sigma**2, noise,
        reset_gap_s=policy['reset_gap_s'], position_gate_nis=policy['position_gate_nis'],
        velocity_gate_nis=policy['velocity_gate_nis'], covariance_floor=policy['velocity_covariance_eigenvalue_floor_m2ps2'])


def run(data, velocity, output):
    policy_path = ROOT / 'config/urbannav_doppler.json'
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    rotation = json.loads((output / 'positioning_rotation_report.json').read_text(encoding='utf-8'))
    paths = [policy_path, data / 'epochs.csv', data / 'observations.csv', velocity / 'velocity.csv',
             velocity / 'preprocessing_plan.json', output / 'positioning_rotation_report.json',
             ROOT / 'src/navdiag/doppler.py', Path(__file__)]
    plan = {'policy': policy, 'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}}
    frozen = output / 'doppler_frozen_plan.json'
    if frozen.exists() and json.loads(frozen.read_text(encoding='utf-8')) != plan:
        raise ValueError('Doppler comparison inputs changed; preserve prior evidence')
    if (output / 'doppler_report.json').exists():
        raise ValueError('Completed Doppler comparison already exists')
    frozen.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    observations = pd.read_csv(data / 'observations.csv')
    velocities = pd.read_csv(velocity / 'velocity.csv')
    context = observations.merge(velocities, on=['trip_id', 'device_id', 'timestamp_ms'], validate='one_to_one', how='left')
    if context.doppler_valid.isna().any() or len(context) != len(velocities):
        raise ValueError('Doppler extraction changed observation population')
    scored = pd.read_csv(data / 'epochs.csv')
    records = []
    for fold in rotation['folds']:
        source_report = output / fold['report_path']
        original = json.loads(source_report.read_text(encoding='utf-8'))
        parts = {role: scored[scored.trip_id.isin(original['splits'][role]['trips'])].copy() for role in ['validation', 'test']}
        histories = {role: context[context.trip_id.isin(part.trip_id.unique())].sort_values('timestamp_ms').reset_index(drop=True)
                     for role, part in parts.items()}
        indices = {role: context_indices(part, histories[role]) for role, part in parts.items()}
        best, best_score = None, float('inf')
        for sigma in policy['position_sigma_candidates_m']:
            for noise in policy['acceleration_noise_candidates_mps2']:
                filtered, _ = apply(histories['validation'], sigma, noise, policy)
                errors = np.linalg.norm(filtered[indices['validation']] - parts['validation'][['gt_e_m', 'gt_n_m']].to_numpy(), axis=1)
                rmse = float(np.sqrt(np.mean(errors**2)))
                if rmse < best_score:
                    best, best_score = (sigma, noise), rmse
        filtered, diagnostics = apply(histories['test'], *best, policy)
        xy = filtered[indices['test']]
        errors = np.linalg.norm(xy - parts['test'][['gt_e_m', 'gt_n_m']].to_numpy(), axis=1)
        metric = coverage_position_metrics(errors, fold['coverage']['eligible_reference_epochs'])
        result = parts['test'][['trip_id', 'device_id', 'timestamp_ms']].copy()
        result[['doppler_filter_e_m', 'doppler_filter_n_m']] = xy
        result['doppler_filter_error_m'] = errors
        result.to_csv(source_report.parent / 'doppler_predictions.csv', index=False)
        record = {'id': fold['id'], 'test_trip': fold['test_trip'], 'source_report_sha256': sha(source_report),
                  'metrics': metric, 'diagnostics_on_all_successful_fixes': diagnostics,
                  'settings': {'position_sigma_m': best[0], 'acceleration_noise_mps2': best[1],
                               'selected_on': 'validation', 'validation_rmse_m': best_score},
                  'baseline_cohort_unchanged': True, 'alert_models_retrained': False}
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    report = {'version': 1, 'task_type': 'doppler_positioning_comparison', 'policy': policy,
              'folds': records, 'input_files_sha256': plan['files']}
    (output / 'doppler_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/processed/urbannav_positioning_v3')
    parser.add_argument('--velocity', type=Path, default=ROOT / 'data/processed/urbannav_doppler')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/urbannav_positioning')
    args = parser.parse_args()
    run(args.data, args.velocity, args.output)
