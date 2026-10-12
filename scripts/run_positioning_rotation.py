"""Rotate three fixed train/validation/test acquisition roles, preserving the primary run."""
import argparse
import json
from pathlib import Path

import numpy as np

from run_positioning_expansion import ROOT, digest, run


def rotate(data, output):
    base = json.loads((ROOT / 'config/urbannav_positioning.json').read_text(encoding='utf-8'))
    path = ROOT / 'config/urbannav_positioning_rotation.json'
    policy = json.loads(path.read_text(encoding='utf-8'))
    plan = {'protocol': policy, 'protocol_sha256': digest(path), 'runner_sha256': digest(Path(__file__)),
            'primary_report_sha256': digest(output / 'report.json'),
            'data_sha256': {name: digest(data / name) for name in ['epochs.csv', 'observations.csv', 'availability.csv']}}
    frozen = output / 'rotation_frozen_plan.json'
    if frozen.exists() and json.loads(frozen.read_text(encoding='utf-8')) != plan:
        raise ValueError('Rotation inputs changed; preserve existing evidence')
    if (output / 'positioning_rotation_report.json').exists():
        raise ValueError('Completed rotation exists; use another output directory')
    frozen.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    folds = []
    for item in policy['folds']:
        if item.get('existing_primary'):
            folder = output
            report = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
        else:
            folder = output / 'folds' / item['id']
            folder.mkdir(parents=True, exist_ok=True)
            protocol = {**base, 'splits': {name: [item[name]] for name in ['train', 'validation', 'test']},
                        'scope': policy['scope'], 'selection_basis': policy['selection_basis']}
            manifest = folder / 'protocol.json'
            manifest.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            report = run(data, folder, manifest)
        folds.append({'id': item['id'], 'report_path': (folder / 'report.json').relative_to(output).as_posix(),
                      'report_sha256': digest(folder / 'report.json'), 'test_trip': item['test'],
                      'selected_model': report['selected_model'], 'models': report['models'],
                      'coverage': report['coverage']['test'][item['test']], 'positioning': report['positioning']})
    fields = ['precision', 'recall', 'f1', 'average_precision']
    def mean(values):
        finite = [value for value in values if value is not None]
        return float(np.mean(finite)) if finite else None
    result = {'version': 1, 'task_type': 'epoch_position_error_rotation', 'data_kind': 'real',
              'protocol': policy, 'folds': folds,
              'total_raw_test_epochs': sum(item['coverage']['raw_epochs'] for item in folds),
              'total_scored_test_epochs': sum(item['coverage']['scored_epochs'] for item in folds),
              'distinct_test_acquisitions': len(folds),
              'models_macro': {name: {key: mean([fold['models'][name]['test'][key] for fold in folds])
                                      for key in fields} for name in folds[0]['models']},
              'limitations': base['limitations'] + [policy['scope'], policy['aggregation']]}
    (output / 'positioning_rotation_report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ['total_raw_test_epochs', 'total_scored_test_epochs',
                                                 'distinct_test_acquisitions', 'models_macro']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/processed/urbannav_positioning_v3')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/urbannav_positioning')
    args = parser.parse_args()
    rotate(args.data, args.output)
