"""Read-only consistency audit of completed local experiments before RAG work.

This checks saved data/report relationships; it neither fits models nor treats
internal consistency as an independent verification of reference-label truth.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['cn0_dbhz', 'elevation_deg', 'residual_normalized',
            'pr_rate_consistency_mps', 'cn0_std_dbhz']
KEYS = ['trip_id', 'device_id', 'satellite_id', 'timestamp_ms']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    checks = []

    def check(name, condition, detail=''):
        checks.append({'check': name, 'passed': bool(condition), 'detail': detail})

    def check_fold(folder):
        r = read(folder / 'report.json')
        check(folder.name + ': feature whitelist', set(r['features']) == set(FEATURES))
        sets = [set(s['trips']) for s in r['splits'].values()]
        check(folder.name + ': acquisition isolation', all(not a & b for i, a in enumerate(sets) for b in sets[i + 1:]))
        validation = {name: value['validation']['f1'] for name, value in r['models'].items()}
        check(folder.name + ': validation model selection',
              validation[r['selected_model']] == max(value for value in validation.values() if value is not None))
        for name, model in r['models'].items():
            test, val = model['test'], model['validation']
            matrix = np.array(test['confusion_matrix'])
            check(f'{folder.name}/{name}: scored denominator', matrix.sum() == r['splits']['test']['rows'])
            check(f'{folder.name}/{name}: frozen threshold', test['threshold'] == val['threshold'])
            tn, fp, fn, tp = matrix.ravel()
            check(f'{folder.name}/{name}: error counts',
                  fp == test['false_positives'] and fn == test['false_negatives'])
            expected = 2 * tp / (2 * tp + fp + fn) if tp + fn and 2 * tp + fp + fn else None
            check(f'{folder.name}/{name}: F1 from counts',
                  test['f1'] is None if expected is None else np.isclose(test['f1'], expected, atol=1e-10))
        return r

    suite_root = root / 'artifacts/klt_protocols'
    suite = read(suite_root / 'suite_report.json')
    folds = {item['id']: check_fold(suite_root / 'folds' / item['id']) for item in suite['experiments']}
    for area in ['KLT1', 'KLT2', 'KLT3']:
        frames = []
        for protocol in ['in_domain', 'cross_region']:
            folder = suite_root / 'folds' / f'{protocol}_{area.lower()}'
            frames.append(pd.read_csv(folder / 'predictions.csv').sort_values(KEYS).reset_index(drop=True))
        a, b = frames
        check(area + ': identical paired test inputs and labels',
              a[KEYS + FEATURES + ['signal_label']].equals(b[KEYS + FEATURES + ['signal_label']]))
        check(area + ': fixed C/N0 rule invariant', a.nlos_score_cn0_rule.equals(b.nlos_score_cn0_rule))
    for protocol, summary in suite['protocols'].items():
        subset = [r for r in folds.values() if r['protocol'] == protocol]
        for name, model in summary['models'].items():
            for metric in ['precision', 'recall', 'f1']:
                expected = np.mean([r['models'][name]['test'][metric] for r in subset])
                check(f'{protocol}/{name}: macro {metric}', np.isclose(model['test_macro'][metric], expected, atol=1e-10))

    gen_root = root / 'artifacts/urbannav_generalization'
    r = read(gen_root / 'generalization_report.json')
    plan = read(gen_root / 'experiment_plan.json')
    source_root = root / plan['plan']['source_bundle']
    source = read(source_root / 'report.json')
    check('UrbanNav: frozen source model', sha(source_root / 'models.joblib') == plan['source_model_sha256'])
    check('UrbanNav: frozen source report', sha(source_root / 'report.json') == plan['source_report_sha256'])
    check('UrbanNav: prepared data unchanged', sha(root / 'data/processed/urbannav/satellites.csv') == plan['dataset_sha256'])
    for name, digest in plan['implementation_sha256'].items():
        check('UrbanNav: frozen implementation ' + name, sha(root / 'src/navdiag' / name) == digest)
    dataset = r['dataset']
    known = sum(item['label_counts']['los'] + item['label_counts']['nlos'] for item in dataset['areas'].values())
    unknown = sum(item['label_counts']['unknown'] for item in dataset['areas'].values())
    check('UrbanNav: known/unknown conservation', known + unknown == dataset['rows'])
    check('UrbanNav: unknown excluded from external scoring', known == r['external_transfer']['rows'])
    check('UrbanNav: source selection retained', r['external_transfer']['selected_model'] == source['selected_model'])
    for name, model in r['external_transfer']['models'].items():
        for area, metrics in model['per_area'].items():
            check(f'external/{area}/{name}: threshold unchanged', metrics['threshold'] == source['models'][name]['validation']['threshold'])
            check(f'external/{area}/{name}: scored coverage', metrics['rows'] == sum(dataset['areas'][area]['label_counts'][k] for k in ['los', 'nlos']))
    district_folds = {item['id']: check_fold(gen_root / 'folds' / item['id']) for item in plan['experiments']}
    for section in ['external_transfer', 'district_holdout']:
        for name, model in r[section]['models'].items():
            for area, value in model['per_area'].items():
                if section == 'district_holdout':
                    reports = [fold for fold in district_folds.values() if fold['experiment']['test_area'] == area]
                    for metric in ['precision', 'recall', 'f1']:
                        expected = np.mean([fold['models'][name]['test'][metric] for fold in reports])
                        check(f'district/{area}/{name}: mean {metric}', np.isclose(value['test_mean'][metric], expected, atol=1e-10))
            for metric in ['precision', 'recall', 'f1']:
                expected = np.mean([(value.get('test_mean') or value)[metric] for value in model['per_area'].values()])
                check(f'{section}/{name}: equal-area {metric}', np.isclose(model['test_macro'][metric], expected, atol=1e-10))
    return {'scope': 'Saved experimental consistency, not independent label accuracy or statistical significance.',
            'passed': sum(item['passed'] for item in checks), 'checks': len(checks),
            'all_passed': all(item['passed'] for item in checks), 'results': checks,
            'research_limits': ['All target regions are in Hong Kong; cross-city transfer is unverified.',
                'UrbanNav references are static-map geometry, not manually verified reception truth.',
                'One drive per UrbanNav district; no independent-drive confidence interval.',
                'No evidence yet that NLOS classification improves positioning.',
                'Live LLM quality has not been evaluated; current checks are offline/tool protocol checks.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/project_review.json')
    args = parser.parse_args()
    report = audit(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"Consistency checks: {report['passed']}/{report['checks']} | {args.out}")
    for result in report['results']:
        if not result['passed']:
            print('FAILED:', result['check'])
    raise SystemExit(0 if report['all_passed'] else 1)


if __name__ == '__main__':
    main()
