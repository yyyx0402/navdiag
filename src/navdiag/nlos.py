"""Satellite-level LOS/NLOS detection, separate from epoch-level position error.

Consumes a published numeric KLT table; no upstream implementation is vendored.
Label provenance and geographic grouping are mandatory and never model inputs.
"""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from navdiag.data import split_by_trip
from navdiag.learning import choose_threshold, evaluate

NLOS_FEATURES = ['cn0_dbhz', 'elevation_deg', 'residual_normalized',
                 'pr_rate_consistency_mps', 'cn0_std_dbhz']
KEYS = ['trip_id', 'device_id', 'timestamp_ms', 'satellite_id']
SOURCE_URL = 'https://github.com/fossettlab/gnss-nlos-notebook'
SOURCE_COMMIT = 'b13aa856f723a472bd35d45171017580642054cf'
SOURCE_SHA256 = '4955326b4e25ffd9be6f86bc3111c2fecd6e5db1aa33e975d15b83aa4304434f'

CLASSIC_DEFAULTS = {
    'logistic_regression': {'C': 1., 'class_weight': 'balanced'},
    'random_forest': {'n_estimators': 200, 'max_depth': 12, 'min_samples_leaf': 8,
                      'max_features': 'sqrt', 'class_weight': 'balanced'},
}


def classic_config(name: str, config: dict | None = None) -> dict:
    if name not in CLASSIC_DEFAULTS:
        raise ValueError(f'Unknown classical model: {name}')
    config = config or {}
    if set(config) - set(CLASSIC_DEFAULTS[name]):
        raise ValueError(f'Unknown {name} hyperparameters')
    return {**CLASSIC_DEFAULTS[name], **config}


def build_classic_model(name: str, config: dict | None = None, seed: int = 42):
    cfg = classic_config(name, config)
    if name == 'logistic_regression':
        return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                             LogisticRegression(**cfg, max_iter=1000, random_state=seed))
    return make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(
        **cfg, random_state=seed, n_jobs=1))


def cn0_variability(frame: pd.DataFrame, window: int = 5) -> pd.Series:
    """Trailing observed-signal std; zero sentinels are missing and >5s gaps reset."""
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, part in frame.groupby(['trip_id', 'device_id', 'satellite_id'], sort=False):
        part = part.sort_values('timestamp_ms')
        values = pd.to_numeric(part.cn0_dbhz).where(part.cn0_dbhz > 0)
        arcs = part.timestamp_ms.diff().gt(5000).cumsum()
        # One observed value cannot establish variability; preserve NaN at warm-up.
        result.loc[part.index] = values.groupby(arcs).transform(
            lambda series: series.rolling(window, min_periods=2).std())
    return result


def prepare_klt(path: Path, output: Path) -> dict:
    """Adapt the pinned external feature table; retain its labels and missing values."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError('KLT feature table differs from the pinned public source')
    raw = pd.read_csv(path)
    required = ['run', 'epoch', 'utc', 'sat', 'sys', 'nlos', 'cn0', 'elev_deg',
                'pr_resid_norm', 'pr_rate_consistency', 'cn0_std']
    if set(required) - set(raw):
        raise ValueError('KLT source schema has changed')
    if not raw.nlos.isin([0, 1]).all():
        raise ValueError('KLT labels must be 0=LOS, 1=NLOS')
    frame = raw.rename(columns={
        'run': 'trip_id', 'sat': 'satellite_id', 'sys': 'system', 'cn0': 'cn0_dbhz',
        'elev_deg': 'elevation_deg', 'pr_resid_norm': 'residual_normalized',
        'pr_rate_consistency': 'pr_rate_consistency_mps', 'cn0_std': 'cn0_std_dbhz',
    })
    frame['timestamp_ms'] = np.rint(frame.utc * 1000).astype('int64')
    frame['device_id'] = 'ublox_f9p'
    unavailable_cn0 = int((frame.cn0_dbhz <= 0).sum())
    frame['cn0_std_dbhz'] = cn0_variability(frame)
    frame['data_kind'] = 'real'
    frame['scene_id'] = frame.trip_id.str.extract(r'_(KLT[123])_', expand=False)
    frame['scene_label'] = 'urban_unspecified'
    frame['scene_label_source'] = 'dataset_description'
    frame['signal_label'] = np.where(frame.nlos == 1, 'nlos', 'los')
    frame['signal_label_source'] = 'fisheye_sky_mask'
    frame['obstruction_type'] = 'unknown'
    # Do not turn the author's urban collection description into building/tree labels.
    frame = validate_observations(frame)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / 'satellites.csv', index=False)
    report = {
        'task_type': 'satellite_nlos', 'data_kind': 'real', 'source_sha256': digest,
        'prepared_sha256': hashlib.sha256((output / 'satellites.csv').read_bytes()).hexdigest(),
        'source_url': SOURCE_URL, 'source_commit': SOURCE_COMMIT, 'license': 'GPL-3.0',
        'original_dataset': 'https://github.com/ebhrz/KLTDataset',
        'rows': len(frame), 'runs': sorted(frame.trip_id.unique().tolist()),
        'areas': sorted(frame.scene_id.unique().tolist()),
        'label_definition': '1=NLOS (direct path blocked), 0=LOS; source fisheye sky mask',
        'features_origin': 'Published KLT radio/SPP features; C/N0 zero sentinels masked and trailing std recomputed locally; no raw RINEX re-extraction',
        'unavailable_cn0_rows': unavailable_cn0,
        'cn0_policy': 'C/N0 <= 0 treated as unavailable; five trailing satellite observations, >=2 observed values, reset on >5s gap',
        'label_limitations': 'No building/tree/tunnel object labels; LOS can contain multipath. Only GPS/BeiDou are labelled.',
        'per_run': {str(trip): {'rows': len(part), 'epochs': part.timestamp_ms.nunique(),
                               'nlos_fraction': float(part.nlos.mean()), 'scene_id': str(part.scene_id.iloc[0])}
                    for trip, part in frame.groupby('trip_id', sort=True)},
    }
    (output / 'preprocess_report.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report


def validate_observations(frame: pd.DataFrame, require_labels: bool = True, sort: bool = True) -> pd.DataFrame:
    required = [*KEYS, *NLOS_FEATURES]
    if require_labels:
        required += ['data_kind', 'scene_id', 'signal_label', 'signal_label_source']
    if set(required) - set(frame):
        raise ValueError(f'Missing satellite columns: {sorted(set(required) - set(frame))}')
    frame = frame.copy()
    if frame.empty:
        raise ValueError('Satellite table is empty')
    for name in [*KEYS[:-2], 'satellite_id', *(['scene_id'] if require_labels else [])]:
        if frame[name].isna().any() or frame[name].astype(str).str.strip().eq('').any():
            raise ValueError(f'Non-empty {name} is required')
    for name in ['timestamp_ms', *NLOS_FEATURES]:
        frame[name] = pd.to_numeric(frame[name], errors='raise')
        if np.isinf(frame[name]).any() or (name == 'timestamp_ms' and frame[name].isna().any()):
            raise ValueError(f'Non-finite satellite feature: {name}')
    if frame.duplicated(KEYS).any():
        raise ValueError('Duplicate satellite observations')
    if require_labels:
        if set(frame.data_kind) != {'real'}:
            raise ValueError('This NLOS experiment requires real data')
        if not frame.signal_label.isin(['los', 'nlos', 'unknown']).all():
            raise ValueError('Signal labels must be los, nlos, or unknown')
        known = frame.signal_label != 'unknown'
        if frame.loc[known, 'signal_label_source'].isna().any() or not frame.loc[known, 'signal_label_source'].isin(
                ['fisheye_sky_mask', 'lidar_sky_mask', 'static_3d_sky_mask', 'dataset_annotation']).all():
            raise ValueError('Known signal labels need independent label provenance')
        if (frame.groupby('trip_id').scene_id.nunique() != 1).any():
            raise ValueError('Every collection run must belong to one scene group')
    frame['cn0_dbhz'] = frame.cn0_dbhz.where(frame.cn0_dbhz > 0)
    return frame.sort_values(KEYS).reset_index(drop=True) if sort else frame


def nlos_scores(bundle: dict, frame: pd.DataFrame, name: str | None = None) -> np.ndarray:
    """Predict from radio/geometry features. No camera, labels or truth needed."""
    name = name or bundle['selected_model']
    frame = validate_observations(frame, require_labels=False, sort=False)
    if name == 'cn0_rule':
        return (pd.to_numeric(frame.cn0_dbhz) < 30).to_numpy(dtype=float)
    if name not in bundle['models']:
        raise ValueError(f'Unknown NLOS model: {name}')
    if name in bundle.get('neural_models', ['transformer']):
        return bundle['models'][name].predict_proba(frame)[:, 1]
    features = bundle['model_features'][name]
    return bundle['models'][name].predict_proba(frame[features].apply(pd.to_numeric))[:, 1]


def _groups(frame: pd.DataFrame, scores: np.ndarray, threshold: float, column: str) -> list[dict]:
    result = []
    copy = frame.assign(_score=scores)
    for group, part in copy.groupby(column, sort=True, observed=True):
        result.append({'group': str(group), 'rows': len(part),
                       **evaluate((part.signal_label == 'nlos').to_numpy(dtype=int), part._score.to_numpy(), threshold)})
    return result


def threshold_tradeoff(labels: np.ndarray, scores: np.ndarray) -> list[dict]:
    """Validation-only operating points; no test labels determine thresholds."""
    return [evaluate(labels, scores, float(threshold)) for threshold in np.arange(0.05, 0.96, 0.05)]


def validate_nlos_protocol(splits: dict, protocol: str = 'area_disjoint') -> dict:
    """Run separation is mandatory; geographic separation depends on the question."""
    areas = {name: set(part.scene_id) for name, part in splits.items()}
    if protocol == 'area_disjoint':
        for name in areas:
            for other in areas:
                if name != other and areas[name] & areas[other]:
                    raise ValueError('Scene groups overlap between splits; keep an area in exactly one split')
    elif protocol == 'in_domain':
        if not areas['validation'] <= areas['train'] or not areas['test'] <= areas['train']:
            raise ValueError('In-domain validation and test areas must be represented in training')
    elif protocol == 'cross_region':
        if areas['test'] & (areas['train'] | areas['validation']):
            raise ValueError('Cross-region test areas must be absent from training and validation')
        if not areas['validation'] <= areas['train']:
            raise ValueError('Cross-region validation must use source areas represented in training')
    else:
        raise ValueError(f'Unknown NLOS protocol: {protocol}')
    return areas


def train_nlos(csv_path: Path, output: Path, manifest: dict, seed: int = 42,
               with_transformer: bool = False, transformer_epochs: int = 30,
               with_neural_models: bool = False, neural_epochs: int = 30,
               neural_configs: dict | None = None, classic_configs: dict | None = None,
               protocol: str = 'area_disjoint') -> dict:
    frame = validate_observations(pd.read_csv(csv_path, dtype={key: 'string' for key in ['trip_id', 'device_id', 'satellite_id']}))
    splits_all = split_by_trip(frame, seed, manifest)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    source = {'description': 'User-provided annotated numeric table; upstream source not asserted'}
    metadata = csv_path.parent / 'preprocess_report.json'
    if metadata.exists():
        provenance = json.loads(metadata.read_text(encoding='utf-8'))
        if provenance.get('prepared_sha256') != digest:
            raise ValueError('Prepared table no longer matches its provenance hash; prepare it again')
        source = {key: provenance[key] for key in ['source_url', 'source_commit', 'source_sha256',
                                                  'original_dataset', 'license', 'features_origin']}
    source['redistributed'] = False
    area_sets = validate_nlos_protocol(splits_all, protocol)
    splits = {name: part[part.signal_label != 'unknown'].copy() for name, part in splits_all.items()}
    labels = {name: (part.signal_label == 'nlos').to_numpy(dtype=int) for name, part in splits.items()}
    for name in ['train', 'validation']:
        if len(np.unique(labels[name])) != 2:
            raise ValueError(f'{name} needs independently labelled LOS and NLOS examples')
    if not len(splits['test']):
        raise ValueError('Test split has no independently labelled observations')
    features = [key for key in NLOS_FEATURES if splits['train'][key].notna().mean() >= 0.5]
    if not features:
        raise ValueError('No observed training features')
    classic_configs, neural_configs = classic_configs or {}, neural_configs or {}
    models = {name: build_classic_model(name, classic_configs.get(name), seed) for name in CLASSIC_DEFAULTS}
    model_features = {name: features for name in models}
    report = {
        'version': '0.3.0', 'task_type': 'satellite_nlos', 'task': 'signal_label == nlos',
        'data_kind': 'real', 'unit': 'satellite_observation', 'seed': seed,
        'sklearn_version': sklearn.__version__,
        'dataset_sha256': digest,
        'features': features, 'dropped_features': [key for key in NLOS_FEATURES if key not in features],
        'missing_feature_fraction_train': {key: float(splits['train'][key].isna().mean()) for key in NLOS_FEATURES},
        'label_sources': sorted(frame.loc[frame.signal_label != 'unknown', 'signal_label_source'].unique().tolist()),
        'splits': {name: {'rows': len(part), 'epochs': len(part.drop_duplicates(['trip_id', 'device_id', 'timestamp_ms'])),
                           'trips': sorted(part.trip_id.unique().tolist()), 'scene_groups': sorted(area_sets[name]),
                           'positive_rate': float(labels[name].mean()), 'excluded_unknown': len(splits_all[name]) - len(part)}
                   for name, part in splits.items()},
        'protocol': protocol,
        'split_policy': {
            'area_disjoint': 'Whole collection runs; train/validation/test have disjoint area groups',
            'in_domain': 'Whole collection runs; validation and test areas are represented by other runs in training',
            'cross_region': 'Whole collection runs; test areas are absent from both training and validation',
        }[protocol],
        'models': {}, 'sanity_baselines': {},
        'selection_policy': 'Highest validation F1; threshold fixed on validation; no test tuning',
        'comparison_policy': 'Complete classifiers, no component ablations. Included neural models share 32 past/current observations, train-only preprocessing, optimizer and early-stopping budget. LR/RF/rule use current features; this is not an isolated neural-versus-tree architecture comparison or per-model optimized benchmark.',
        'warning': (provenance.get('label_warning', f'Real NLOS reference labels from {frame.trip_id.nunique()} acquisitions; results apply to the stated areas, devices and label method.')
                    if metadata.exists() else 'Performance applies only to this scene-grouped annotated table; upstream collection is not asserted.'),
        'evaluation_limitations': [
            'Satellite rows within an epoch and run are correlated, not independent trials.',
            f'Test covers {splits["test"].trip_id.nunique()} run(s) and {len(area_sets["test"])} area group(s); no independent-drive confidence interval estimated.',
            'LOS includes possible multipath. No obstruction object type or untracked-satellite label.',
            'Reference label accuracy and upstream feature extraction are not independently verified by this classifier.',
            'No positioning improvement established by this classification-only table.',
            'Test area has been inspected during development; further tuning requires a new untouched final holdout.',
        ],
        'source': source,
    }
    test = splits['test'].copy()
    for name, model in models.items():
        used = model_features[name]
        model.fit(splits['train'][used], labels['train'])
        val_scores = model.predict_proba(splits['validation'][used])[:, 1]
        threshold = choose_threshold(labels['validation'], val_scores)
        scores = model.predict_proba(test[used])[:, 1]
        test[f'nlos_score_{name}'] = scores
        report['models'][name] = {
            'configuration': classic_config(name, classic_configs.get(name)),
            'features': used, 'train': evaluate(labels['train'], model.predict_proba(splits['train'][used])[:, 1], threshold),
            'validation': evaluate(labels['validation'], val_scores, threshold),
            'validation_threshold_tradeoff': threshold_tradeoff(labels['validation'], val_scores),
            'test': evaluate(labels['test'], scores, threshold),
            'per_test_run': _groups(test, scores, threshold, 'trip_id'),
        }
    neural_models = []
    if with_neural_models or with_transformer:
        from navdiag.nlos_neural import NEURAL_MODELS, train_neural
        neural_models = list(NEURAL_MODELS) if with_neural_models else ['transformer']
    report['neural_models'] = neural_models
    for name in neural_models:
        model, training = train_neural(splits_all['train'], splits_all['validation'],
                                      features, name, seed,
                                      neural_epochs if with_neural_models else transformer_epochs,
                                      config=neural_configs.get(name))
        models[name] = model
        model_features[name] = features
        sequence_scores = {
            split: model.predict_proba(part)[:, 1][part.signal_label.ne('unknown').to_numpy()]
            for split, part in splits_all.items()
        }
        threshold = choose_threshold(labels['validation'], sequence_scores['validation'])
        scores = sequence_scores['test']
        test[f'nlos_score_{name}'] = scores
        report['models'][name] = {
            'features': features, 'training': training,
            'validation_threshold_tradeoff': threshold_tradeoff(labels['validation'], sequence_scores['validation']),
            **{split: evaluate(labels[split], sequence_scores[split], threshold) for split in splits},
            'per_test_run': _groups(test, scores, threshold, 'trip_id'),
        }
    rule = lambda part: (part.cn0_dbhz < 30).to_numpy(dtype=float)
    test['nlos_score_cn0_rule'] = rule(test)
    report['models']['cn0_rule'] = {
        'features': ['cn0_dbhz'], 'validation': evaluate(labels['validation'], rule(splits['validation']), 0.5),
        'test': evaluate(labels['test'], rule(test), 0.5),
        'per_test_run': _groups(test, rule(test), 0.5, 'trip_id'),
    }
    for name, value in [('always_los', 0), ('always_nlos', 1)]:
        report['sanity_baselines'][name] = {split: evaluate(labels[split], np.full(len(part), value), 0.5)
                                         for split, part in splits.items() if split != 'train'}
    selected = max(report['models'], key=lambda name: report['models'][name]['validation']['f1'])
    if neural_models:
        report['best_neural_model'] = max(neural_models, key=lambda name: report['models'][name]['validation']['f1'])
    threshold = report['models'][selected]['validation']['threshold']
    report['selected_model'] = selected
    test['predicted_nlos_score'] = test[f'nlos_score_{selected}']
    test['predicted_nlos'] = test.predicted_nlos_score >= threshold
    test['prediction_status'] = np.where(test.predicted_nlos, 'nlos', 'los')
    test['outcome'] = np.select([
        (test.signal_label == 'nlos') & test.predicted_nlos,
        (test.signal_label == 'los') & test.predicted_nlos,
        (test.signal_label == 'nlos') & ~test.predicted_nlos,
    ], ['true_positive', 'false_positive', 'false_negative'], default='true_negative')
    buckets = test.assign(cn0_bin=pd.cut(test.cn0_dbhz, [-np.inf, 25, 35, 45, np.inf]),
                          elevation_bin=pd.cut(test.elevation_deg, [-np.inf, 15, 30, 60, np.inf]))
    for column in ['cn0_bin', 'elevation_bin']:
        buckets[column] = buckets[column].astype('string').fillna('missing')
    report['error_analysis'] = {column: _groups(buckets, test.predicted_nlos_score.to_numpy(), threshold, column)
                              for column in ['system', 'cn0_bin', 'elevation_bin'] if column in buckets}
    bundle = {'task_type': 'satellite_nlos', 'models': models, 'model_features': model_features,
              'neural_models': neural_models,
              'selected_model': selected, 'threshold': threshold, 'features': features}
    output.mkdir(parents=True, exist_ok=True)
    test.to_csv(output / 'predictions.csv', index=False)
    joblib.dump(bundle, output / 'models.joblib')
    (output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    return report
