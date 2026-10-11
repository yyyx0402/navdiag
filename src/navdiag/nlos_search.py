"""Bounded NLOS hyperparameter search; trials never evaluate the test split."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from navdiag.data import split_by_trip
from navdiag.learning import choose_threshold, evaluate
from navdiag.nlos import (CLASSIC_DEFAULTS, NLOS_FEATURES, build_classic_model,
                          classic_config, train_nlos, validate_observations)
from navdiag.nlos_neural import NEURAL_MODELS, neural_config, train_neural


def _write(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def validate_search_plan(plan: dict) -> dict:
    if not isinstance(plan.get('neural_epochs'), int) or plan['neural_epochs'] < 1:
        raise ValueError('Search plan needs positive neural_epochs')
    expected = set(CLASSIC_DEFAULTS) | set(NEURAL_MODELS)
    if set(plan.get('models', {})) != expected:
        raise ValueError('Search plan must cover LR, RF and all five neural models')
    normalized = {}
    for name, proposals in plan['models'].items():
        if not proposals or len({proposal['id'] for proposal in proposals}) != len(proposals):
            raise ValueError('Every model needs nonempty, uniquely named candidates')
        normalized[name] = []
        seen = set()
        for proposal in proposals:
            params = (classic_config(name, proposal['parameters']) if name in CLASSIC_DEFAULTS
                      else neural_config(proposal['parameters']))
            fingerprint = json.dumps(params, sort_keys=True)
            if fingerprint in seen:
                raise ValueError(f'Duplicate effective candidate for {name}')
            seen.add(fingerprint)
            normalized[name].append({'id': proposal['id'], 'parameters': params})
    return {**plan, 'models': normalized}


def tune_nlos(csv_path: Path, output: Path, manifest: dict, plan: dict, seed: int = 42,
              baseline_report: Path | None = None, progress: bool = False) -> dict:
    plan = validate_search_plan(plan)
    frame = validate_observations(pd.read_csv(csv_path, dtype={
        key: 'string' for key in ['trip_id', 'device_id', 'satellite_id']}))
    splits = split_by_trip(frame, seed, manifest)
    areas = {name: set(part.scene_id) for name, part in splits.items()}
    for name, own in areas.items():
        if any(own & other for other_name, other in areas.items() if other_name != name):
            raise ValueError('Scene groups overlap between splits')
    # Only these two frames enter any trial training, threshold search or ranking.
    train, validation = splits['train'], splits['validation']
    known_train = train.signal_label.ne('unknown').to_numpy()
    known_val = validation.signal_label.ne('unknown').to_numpy()
    y_train = train.loc[known_train, 'signal_label'].eq('nlos').to_numpy(dtype=int)
    y_val = validation.loc[known_val, 'signal_label'].eq('nlos').to_numpy(dtype=int)
    if len(np.unique(y_train)) != 2 or len(np.unique(y_val)) != 2:
        raise ValueError('Train and validation need both LOS and NLOS')
    features = [name for name in NLOS_FEATURES if train.loc[known_train, name].notna().mean() >= 0.5]
    if not features:
        raise ValueError('No observed training features')
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    previous = None
    if baseline_report is not None:
        previous = json.loads(baseline_report.read_text(encoding='utf-8'))
        if previous['dataset_sha256'] != digest or previous['seed'] != seed:
            raise ValueError('Baseline dataset or seed does not match the search')
        for name, part in splits.items():
            if previous['splits'][name]['trips'] != sorted(part.trip_id.unique().tolist()):
                raise ValueError('Baseline trip splits differ from the search')
    output.mkdir(parents=True, exist_ok=True)
    frozen_plan = {**plan, 'seed': seed, 'dataset_sha256': digest, 'split_manifest': manifest}
    _write(output / 'search_plan.json', frozen_plan)
    trace = {'status': 'running', 'seed': seed, 'dataset_sha256': digest,
             'search_plan_sha256': hashlib.sha256((output / 'search_plan.json').read_bytes()).hexdigest(),
             'trial_selection_data': 'validation only', 'trial_test_metrics_recorded': False,
             'neural_epochs': plan['neural_epochs'], 'trials': [], 'selected_candidates': {}}
    if previous is not None:
        _write(output / 'baseline_report.json', previous)
    _write(output / 'search_trace.json', trace)
    total = sum(len(candidates) for candidates in plan['models'].values())
    for name, candidates in plan['models'].items():
        best = None
        for proposal in candidates:
            start = time.monotonic()
            params = proposal['parameters']
            if name in CLASSIC_DEFAULTS:
                model = build_classic_model(name, params, seed)
                model.fit(train.loc[known_train, features], y_train)
                scores = model.predict_proba(validation.loc[known_val, features])[:, 1]
                training = None
            else:
                model, training = train_neural(train, validation, features, name,
                                               seed=seed, epochs=plan['neural_epochs'], config=params)
                scores = model.predict_proba(validation)[:, 1][known_val]
            threshold = choose_threshold(y_val, scores)
            metrics = evaluate(y_val, scores, threshold)
            trial = {'model': name, 'id': proposal['id'], 'parameters': params,
                     'validation': metrics, 'elapsed_s': time.monotonic() - start}
            if training is not None:
                trial['training'] = training
            trace['trials'].append(trial)
            # Stable first-proposal tie break; no test scores appear in the key.
            if best is None or metrics['f1'] > best['validation']['f1']:
                best = trial
            _write(output / 'search_trace.json', trace)
            if progress:
                display = lambda value: 'N/A' if value is None else f'{value:.4f}'
                print(f"[{len(trace['trials'])}/{total}] {name}/{proposal['id']}: "
                      f"Val F1={display(metrics['f1'])} P={display(metrics['precision'])} R={display(metrics['recall'])}", flush=True)
        trace['selected_candidates'][name] = {
            key: best[key] for key in ['id', 'parameters', 'validation']}
        del model
    trace['status'] = 'selection_frozen_before_test'
    _write(output / 'search_trace.json', trace)
    _write(output / 'selection_before_test.json', {
        'selected_candidates': trace['selected_candidates'], 'selection_data': 'validation only',
        'search_plan_sha256': trace['search_plan_sha256'], 'dataset_sha256': digest})
    if progress:
        print('Selections frozen. Refit selected configurations and evaluate once on held-out area.', flush=True)
    if hashlib.sha256(csv_path.read_bytes()).hexdigest() != digest:
        raise ValueError('Dataset changed during search; frozen selection cannot be evaluated')
    configs = {name: candidate['parameters'] for name, candidate in trace['selected_candidates'].items()}
    report = train_nlos(csv_path, output, manifest, seed, with_neural_models=True,
                        neural_epochs=plan['neural_epochs'],
                        neural_configs={name: configs[name] for name in NEURAL_MODELS},
                        classic_configs={name: configs[name] for name in CLASSIC_DEFAULTS})
    for name, candidate in trace['selected_candidates'].items():
        actual = report['models'][name]['validation']
        if actual['threshold'] != candidate['validation']['threshold'] or not np.isclose(
                actual['f1'], candidate['validation']['f1'], atol=1e-10, rtol=0):
            raise ValueError('Refit validation metric differs from frozen selection')
    report['hyperparameter_search'] = {
        'trials': total, 'trials_per_model': {name: len(candidates) for name, candidates in plan['models'].items()},
        'selection_metric': 'validation F1 after validation threshold selection',
        'search_plan_sha256': trace['search_plan_sha256'],
        'selected_candidates': trace['selected_candidates'],
        'test_status': 'Development holdout inspected in earlier experiments; a new final holdout is needed.',
    }
    if previous is not None:
        report['baseline_comparison'] = {
            'selected_model_before': previous['selected_model'],
            'models': {name: {'before': previous['models'][name], 'after': report['models'][name]}
                       for name in report['models'] if name in previous['models']},
        }
    report['comparison_policy'] = ('Eight predefined candidates per tuned family in the default plan; '
                                   'same train/validation areas and early stopping budget. Window/capacity may differ. '
                                   'No component ablations or test-driven trial additions; this is a bounded development search.')
    _write(output / 'report.json', report)
    trace['status'] = 'completed'
    trace['selected_model'] = report['selected_model']
    _write(output / 'search_trace.json', trace)
    return report
