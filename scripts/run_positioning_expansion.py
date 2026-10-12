"""Run the fixed full-acquisition positioning/alert protocol once; never tune on test."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import pandas as pd

from navdiag.availability import availability_summary, coverage_position_metrics
from navdiag.learning import train_and_evaluate

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(data: Path, output: Path, plan_path: Path):
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    preprocessing = json.loads((data / 'preprocess_report.json').read_text(encoding='utf-8'))
    source_paths = [plan_path, data / 'epochs.csv', data / 'observations.csv',
                    data / 'availability.csv', data / 'preprocessing_plan.json', data / 'preprocess_report.json',
                    Path(__file__), *[ROOT / 'src/navdiag' / f'{name}.py'
                                      for name in ['learning', 'temporal', 'fusion', 'context', 'availability']]]
    frozen = {'protocol': plan, 'python_version': platform.python_version(),
              'files': {path.relative_to(ROOT).as_posix(): digest(path) for path in source_paths}}
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'report.json').exists():
        raise ValueError('Completed results already exist; preserve them and choose a new output directory')
    frozen_path = output / 'frozen_plan.json'
    if frozen_path.exists() and json.loads(frozen_path.read_text(encoding='utf-8')) != frozen:
        raise ValueError('Frozen implementation/data changed; choose a new output directory')
    frozen_path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    observations = pd.read_csv(data / 'observations.csv', dtype={'trip_id': 'string'})
    events = pd.read_csv(data / 'availability.csv', dtype={'trip_id': 'string'})
    report = train_and_evaluate(data / 'epochs.csv', output, error_limit_m=plan['error_limit_m'],
        seed=plan['seed'], with_gru=plan['with_gru'], gru_epochs=plan['gru_epochs'],
        split_manifest=plan['splits'], observation_context=observations)
    coverage = {}
    for split, trip_ids in plan['splits'].items():
        coverage[split] = {trip: availability_summary(events[events.trip_id == trip]) for trip in trip_ids}
    report.update({'protocol': plan, 'coverage': coverage, 'baseline_solver': plan['solver'],
                   'preprocessing': preprocessing['preprocessing'],
                   'evaluation_limitations': plan['limitations'],
                   'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                   'artifact_inputs_sha256': frozen['files']})
    predictions = pd.read_csv(output / 'predictions.csv')
    columns = {'wls': 'horizontal_error_m', 'fixed_filter': 'fixed_filter_error_m',
               'adaptive_filter': 'adaptive_filter_error_m'}
    eligible = sum(item['eligible_reference_epochs'] for item in coverage['test'].values())
    report['positioning']['methods'] = {name: coverage_position_metrics(predictions[column].to_numpy(), eligible)
                                         for name, column in columns.items()}
    report['positioning']['coverage_policy'] = 'No position or alert output is credited at failed SPP epochs. Accuracy is conditional on reference-scored fixes; the joint availability/accuracy fraction uses all quality-eligible reference epochs.'
    report['positioning']['test_independent_acquisitions'] = len(plan['splits']['test'])
    report['warning'] = 'One previously explored Hong Kong drive. Longer recording is not independent replication; SPP outages and partial/low-quality references limit score coverage.'
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    selected = report['models'][report['selected_model']]['test']
    print(json.dumps({'selected_model': report['selected_model'], 'test_alert': selected,
                      'test_coverage': coverage['test'], 'positioning': report['positioning']},
                     ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/processed/urbannav_positioning_v3')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/urbannav_positioning')
    parser.add_argument('--plan', type=Path, default=ROOT / 'config/urbannav_positioning.json')
    args = parser.parse_args()
    run(args.data, args.output, args.plan)
