"""Unseen-district evaluation with fixed configurations and frozen thresholds."""
import hashlib
import itertools
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from navdiag.data import split_by_trip
from navdiag.learning import evaluate
from navdiag.nlos import CLASSIC_DEFAULTS, NLOS_FEATURES, nlos_scores, validate_nlos_protocol, validate_observations
from navdiag.nlos_protocols import _run_fold


def area_manifests(frame: pd.DataFrame, target_areas: list[str]) -> list[dict]:
    areas = frame.groupby('trip_id').scene_id.first()
    if set(areas) != set(target_areas) or len(areas) != 3 or areas.nunique() != 3:
        raise ValueError('UrbanNav protocol requires one whole acquisition in each of three districts')
    by_area = {area: trip for trip, area in areas.items()}
    specs = []
    for test_area in target_areas:
        source_areas = [area for area in target_areas if area != test_area]
        for train_area, validation_area in itertools.permutations(source_areas):
            manifest = {'train': [by_area[train_area]], 'validation': [by_area[validation_area]],
                        'test': [by_area[test_area]]}
            validate_nlos_protocol(split_by_trip(frame, manifest=manifest), 'area_disjoint')
            specs.append({'id': f'test_{test_area}_train_{train_area}', 'protocol': 'area_disjoint',
                          'test_area': test_area, 'train_area': train_area,
                          'validation_area': validation_area, 'manifest': manifest, 'excluded_runs': []})
    return specs


def macro_metrics(metrics: list[dict]) -> dict:
    keys = ['precision', 'recall', 'f1', 'accuracy', 'average_precision']
    return {key: None if any(m[key] is None for m in metrics) else float(np.mean([m[key] for m in metrics]))
            for key in keys}


def external_evaluation(frame: pd.DataFrame, bundle: dict, source: dict) -> tuple[dict, pd.DataFrame]:
    if set(bundle['features']) - set(NLOS_FEATURES):
        raise ValueError('Frozen classifier contains non-radio/truth features')
    predictions = frame.copy()
    reports = {}
    known = frame.signal_label.ne('unknown')
    for name in source['models']:
        if name != 'cn0_rule' and set(bundle['model_features'][name]) - set(NLOS_FEATURES):
            raise ValueError('Model contains unsupported or truth-derived input features')
        scores = nlos_scores(bundle, frame, name)
        predictions[f'nlos_score_{name}'] = scores
        threshold = source['models'][name]['validation']['threshold']
        area_metrics = {}
        for area, part in frame.groupby('scene_id', sort=False):
            idx = part.index[part.signal_label.ne('unknown')]
            area_metrics[str(area)] = {
                **evaluate((frame.loc[idx, 'signal_label'] == 'nlos').to_numpy(int), scores[idx], threshold),
                'rows': len(idx), 'total_rows': len(part), 'unknown_rows': int(part.signal_label.eq('unknown').sum()),
                'label_coverage': len(idx) / len(part), 'epochs': frame.loc[idx, 'timestamp_ms'].nunique(),
                'trip_id': str(part.trip_id.iloc[0])}
        reports[name] = {'threshold_origin': 'Frozen source validation', 'threshold': threshold,
                         'source_test': source['models'][name]['test'],
                         'per_area': area_metrics, 'test_macro': macro_metrics(list(area_metrics.values()))}
    return {'selected_model': source['selected_model'], 'models': reports,
            'source_splits': source['splits'], 'source_protocol': source['protocol'],
            'selection_policy': 'Source validation selection retained; all new districts are test-only',
            'rows': int(known.sum()), 'total_rows': len(frame)}, predictions


def run_generalization(csv_path: Path, output: Path, plan: dict, progress: bool = False,
                       workers: int = 2) -> dict:
    from navdiag.nlos_neural import DEFAULT_NEURAL_CONFIG, NEURAL_MODELS

    frame = validate_observations(pd.read_csv(csv_path))
    metadata = json.loads((csv_path.parent / 'preprocess_report.json').read_text(encoding='utf-8'))
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    if digest != metadata['prepared_sha256'] or metadata['reference_policy'] != plan['reference_policy']:
        raise ValueError('Target observations or visibility-reference policy differ from the declared plan')
    specs = area_manifests(frame, plan['target_areas'])
    source_root = Path(plan['source_bundle'])
    source = json.loads((source_root / 'report.json').read_text(encoding='utf-8'))
    if set(frame.scene_id) & set(sum([s['scene_groups'] for s in source['splits'].values()], [])):
        raise ValueError('External target areas overlap source areas')
    freeze = {'plan': plan, 'dataset_sha256': digest, 'experiments': specs,
              'source_model_sha256': hashlib.sha256((source_root / 'models.joblib').read_bytes()).hexdigest(),
              'source_report_sha256': hashlib.sha256((source_root / 'report.json').read_bytes()).hexdigest(),
              'classic_configs': CLASSIC_DEFAULTS, 'neural_config': DEFAULT_NEURAL_CONFIG,
              'neural_models': list(NEURAL_MODELS),
              'implementation_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                 for name in ['nlos.py', 'nlos_neural.py', 'nlos_transformer.py', 'nlos_external.py', 'visibility.py']}}
    output.mkdir(parents=True, exist_ok=True)
    frozen_path = output / 'experiment_plan.json'
    if frozen_path.exists() and json.loads(frozen_path.read_text(encoding='utf-8')) != freeze:
        raise ValueError('Existing generalization artifacts use another frozen plan; use a new output directory')
    frozen_path.write_text(json.dumps(freeze, indent=2, ensure_ascii=False), encoding='utf-8')
    if progress:
        print('Evaluating frozen KLT models on three unseen districts; source thresholds unchanged', flush=True)
    bundle = joblib.load(source_root / 'models.joblib')
    external, predictions = external_evaluation(frame, bundle, source)
    predictions.to_csv(output / 'external_predictions.csv', index=False)
    (output / 'external_report.json').write_text(json.dumps(external, indent=2), encoding='utf-8')
    reports = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks = {pool.submit(_run_fold, output, frame, metadata, digest, plan, spec): spec for spec in specs}
        for task in as_completed(tasks):
            spec = tasks[task]
            reports[spec['id']] = task.result()
            if progress:
                print(f"[{len(reports)}/6] district holdout {spec['id']} complete", flush=True)
    reports = {spec['id']: reports[spec['id']] for spec in specs}
    models = {}
    for name in reports[specs[0]['id']]['models']:
        per_area = {}
        for area in plan['target_areas']:
            area_reports = [r for r in reports.values() if r['experiment']['test_area'] == area]
            per_area[area] = {'test_mean': macro_metrics([r['models'][name]['test'] for r in area_reports]),
                              'directions': {r['experiment']['train_area']: r['models'][name]['test'] for r in area_reports}}
        models[name] = {'per_area': per_area,
                        'test_macro': macro_metrics([v['test_mean'] for v in per_area.values()]),
                        'validation_f1_mean': float(np.mean([r['models'][name]['validation']['f1'] for r in reports.values()]))}
    summary = {'task_type': 'satellite_nlos_generalization', 'dataset': metadata, 'plan': freeze,
               'external_transfer': external,
               'district_holdout': {'models': models,
                                    'selected_model': max(models, key=lambda n: models[n]['validation_f1_mean']),
                                    'experiments': [{'id': spec['id'], **spec,
                                                      'selected_model': reports[spec['id']]['selected_model'],
                                                      'splits': reports[spec['id']]['splits']}
                                                     for spec in specs]},
               'limitations': metadata['limitations'] + [
                   'KLT-to-UrbanNav transfer changes geography, acquisition dates, antenna and reference-label method together.',
                   'Within UrbanNav, receiver/label method match, but district and acquisition date remain confounded.',
                   'Six train/validation directions reuse three drives; they are not six independent test datasets.',
                   'Scores condition on accepted static-building references; unknown rows remain unscored.',
                   'One seed and no independent-drive confidence interval; no cross-city or universal model winner claim.']}
    if hashlib.sha256(csv_path.read_bytes()).hexdigest() != digest:
        raise ValueError('Target observations changed during evaluation')
    (output / 'generalization_report.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return summary
