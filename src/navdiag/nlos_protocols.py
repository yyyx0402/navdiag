"""Fixed-configuration, paired acquisition/region experiments. No search loop."""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from navdiag.data import split_by_trip
from navdiag.nlos import CLASSIC_DEFAULTS, train_nlos, validate_nlos_protocol, validate_observations


def protocol_manifests(frame: pd.DataFrame, plan: dict) -> list[dict]:
    """Validate acquisition membership using metadata, never classifier scores."""
    if plan.get('version') != 1 or not plan.get('folds'):
        raise ValueError('Protocol plan needs version 1 and paired folds')
    if (frame.groupby('trip_id').scene_id.nunique() != 1).any():
        raise ValueError('Every acquisition must belong to one area')
    trip_areas = frame.groupby('trip_id').scene_id.first().to_dict()
    all_areas = set(trip_areas.values())
    specs, ids, test_trips = [], set(), set()
    for fold in plan['folds']:
        area, fold_id = fold['test_area'], fold['id']
        if fold_id in ids or not fold_id.replace('_', '').isalnum():
            raise ValueError('Fold IDs must be unique safe names')
        ids.add(fold_id)
        manifest = {'train': fold['in_domain_train'], 'validation': fold['validation'], 'test': fold['test']}
        split_by_trip(frame, manifest=manifest)  # Require all reliable runs exactly once.
        if {trip_areas[t] for t in manifest['test']} != {area}:
            raise ValueError('A paired fold must test exactly its declared target area')
        if {trip_areas[t] for t in manifest['train']} != all_areas:
            raise ValueError('In-domain training must cover all available areas')
        if {trip_areas[t] for t in manifest['validation']} != all_areas - {area}:
            raise ValueError('Shared validation must cover the source areas, excluding the target')
        if test_trips & set(manifest['test']):
            raise ValueError('Test acquisitions cannot be repeated across target-area folds')
        test_trips.update(manifest['test'])
        for protocol in ['in_domain', 'cross_region']:
            split = {name: list(trips) for name, trips in manifest.items()}
            if protocol == 'cross_region':
                split['train'] = [t for t in split['train'] if trip_areas[t] != area]
            included = [t for trips in split.values() for t in trips]
            subset = frame[frame.trip_id.isin(included)]
            parts = split_by_trip(subset, manifest=split)
            validate_nlos_protocol(parts, protocol)
            specs.append({'id': f'{protocol}_{fold_id}', 'protocol': protocol,
                          'test_area': area, 'manifest': split,
                          'excluded_runs': sorted(set(trip_areas) - set(included))})
    if {fold['test_area'] for fold in plan['folds']} != all_areas:
        raise ValueError('Each available area needs a paired test fold')
    return specs


def summarize_protocols(reports: dict[str, dict]) -> dict:
    """Rank observed results separately from validation-only model selection."""
    summary = {}
    def average(values):
        # An area with no positive predictions has undefined precision.
        # Do not silently drop it from a supposedly three-area macro average.
        return None if any(value is None for value in values) else float(np.mean(values))
    for protocol in ['in_domain', 'cross_region']:
        folds = [r for r in reports.values() if r['protocol'] == protocol]
        models = {}
        for name in folds[0]['models']:
            models[name] = {
                'validation_f1_mean': float(np.mean([r['models'][name]['validation']['f1'] for r in folds])),
                'test_macro': {metric: average([r['models'][name]['test'][metric] for r in folds])
                               for metric in ['precision', 'recall', 'f1', 'accuracy', 'average_precision']},
                'per_area': {r['experiment']['test_area']: r['models'][name]['test'] for r in folds},
            }
        summary[protocol] = {
            'models': models,
            'selected_model': max(models, key=lambda n: models[n]['validation_f1_mean']),
            'observed_test_order': sorted(models, key=lambda n: models[n]['test_macro']['f1'], reverse=True),
        }
    differences = {}
    for name in summary['in_domain']['models']:
        within = summary['in_domain']['models'][name]['test_macro']
        cross = summary['cross_region']['models'][name]['test_macro']
        differences[name] = {m: cross[m] - within[m] if cross[m] is not None and within[m] is not None else None
                             for m in ['precision', 'recall', 'f1']}
    return {'protocols': summary, 'cross_minus_in_domain': differences}


def _run_fold(output: Path, frame: pd.DataFrame, metadata: dict, digest: str, plan: dict, spec: dict) -> dict:
    fold_root = output / 'folds' / spec['id']
    report_path = fold_root / 'report.json'
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if report.get('experiment') != spec:
            raise ValueError('Cached fold differs from frozen protocol')
        return report
    input_dir = fold_root / 'input'
    input_dir.mkdir(parents=True, exist_ok=True)
    subset = frame[frame.trip_id.isin(sum(spec['manifest'].values(), []))]
    subset_path = input_dir / 'satellites.csv'
    subset.to_csv(subset_path, index=False)
    provenance = {**metadata, 'prepared_sha256': hashlib.sha256(subset_path.read_bytes()).hexdigest(),
                  'parent_prepared_sha256': digest}
    (input_dir / 'preprocess_report.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    report = train_nlos(subset_path, fold_root, spec['manifest'], plan['seed'],
                        with_neural_models=True, neural_epochs=plan['neural_epochs'],
                        protocol=spec['protocol'])
    report['experiment'] = spec
    total = sum(s['epochs'] for s in report['splits'].values())
    for split in report['splits'].values():
        split['epoch_fraction'] = split['epochs'] / total
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report


def run_protocols(csv_path: Path, output: Path, plan: dict, progress: bool = False, workers: int = 2) -> dict:
    from navdiag.nlos_neural import DEFAULT_NEURAL_CONFIG, NEURAL_MODELS

    frame = validate_observations(pd.read_csv(csv_path))
    if workers not in [1, 2, 3]:
        raise ValueError('Protocol workers must be 1, 2 or 3 separate processes')
    specs = protocol_manifests(frame, plan)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    metadata = json.loads((csv_path.parent / 'preprocess_report.json').read_text(encoding='utf-8'))
    if metadata.get('prepared_sha256') != digest:
        raise ValueError('Expanded observations differ from their provenance hash')
    if metadata.get('excluded_unreliable_runs') != ['1116_KLT3_649']:
        raise ValueError('Expanded KLT experiment must explicitly exclude the unreliable acquisition')
    freeze = {'dataset_sha256': digest, 'plan': plan, 'experiments': specs,
              'classic_configs': CLASSIC_DEFAULTS, 'neural_config': DEFAULT_NEURAL_CONFIG,
              'neural_models': list(NEURAL_MODELS),
              'selection_policy': 'Per-fold threshold and checkpoint from validation; no test tuning',
              'aggregation_policy': 'Equal-weight mean of three target-area folds; descriptive, no significance claim'}
    freeze['implementation_sha256'] = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ['nlos.py', 'nlos_neural.py', 'nlos_transformer.py', 'nlos_protocols.py']}
    output.mkdir(parents=True, exist_ok=True)
    frozen_path = output / 'experiment_plan.json'
    if frozen_path.exists() and json.loads(frozen_path.read_text(encoding='utf-8')) != freeze:
        raise ValueError('Existing protocol artifacts use another frozen plan; use a new output directory')
    frozen_path.write_text(json.dumps(freeze, indent=2, ensure_ascii=False), encoding='utf-8')
    reports = {}
    if workers == 1:
        for spec in specs:
            if progress:
                print(f"{spec['id']} fixed-parameter training", flush=True)
            reports[spec['id']] = _run_fold(output, frame, metadata, digest, plan, spec)
    else:
        # Separate processes isolate PyTorch random seeds and each artifact directory.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            tasks = {pool.submit(_run_fold, output, frame, metadata, digest, plan, spec): spec for spec in specs}
            if progress:
                print(f'{len(specs)} fixed experiments dispatched to {workers} processes; no search', flush=True)
            for task in as_completed(tasks):
                spec = tasks[task]
                reports[spec['id']] = task.result()
                if progress:
                    print(f"[{len(reports)}/{len(specs)}] completed {spec['id']}: validation-selected {reports[spec['id']]['selected_model']}", flush=True)
    if hashlib.sha256(csv_path.read_bytes()).hexdigest() != digest:
        raise ValueError('Source observations changed during the fixed experiments')
    reports = {spec['id']: reports[spec['id']] for spec in specs}
    summary = {'task_type': 'satellite_nlos_suite', 'dataset_sha256': digest,
               'dataset': metadata, 'plan': freeze,
               'experiments': [{'id': spec['id'], 'protocol': spec['protocol'], 'test_area': spec['test_area'],
                                'splits': reports[spec['id']]['splits'],
                                'selected_model': reports[spec['id']]['selected_model']}
                               for spec in specs],
               **summarize_protocols(reports),
               'limitations': [
                   'Only three nearby KLT areas, three dates and one receiver; no cross-city/device claim.',
                   'Paired tests and validation acquisitions are identical; removing target-area training also reduces training size.',
                   'Complete acquisition separation, not strict chronological or leave-date-out evaluation; some same-day runs are adjacent.',
                   'Earlier development inspected November 9 runs; this suite is an exploratory comparison, not an untouched final benchmark.',
                   'One seed, three correlated area folds; small differences are not evidence of a statistically stable winner.',
                   'KLT3-649 excluded because the source declares its labels unreliable; no synthetic epochs added.',
               ]}
    (output / 'suite_report.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return summary
