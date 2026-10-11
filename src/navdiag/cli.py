"""A small CLI: follow main() to see the complete experiment flow."""

import argparse
import json
import sys
from pathlib import Path

from navdiag.data import generate_demo
from navdiag.evidence import summarize_trip
from navdiag.learning import train_and_evaluate


def main() -> None:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description="GNSS quality prediction: a learning-first baseline")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Generate SYNTHETIC data and run an experiment")
    demo.add_argument("--out", type=Path, default=Path("artifacts/demo"))
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument('--navigation', action='store_true', help='Simulate code observations and run actual WLS + filtering')
    demo.add_argument('--with-gru', action='store_true')
    demo.add_argument('--gru-epochs', type=int, default=20)
    train = commands.add_parser("train", help="Train on an epoch table conforming to data/schema.csv")
    train.add_argument("--csv", type=Path, required=True)
    train.add_argument("--out", type=Path, default=Path("artifacts/experiment"))
    train.add_argument("--error-limit-m", type=float, default=10)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument('--allow-missing-features', action='store_true')
    train.add_argument('--with-gru', action='store_true')
    train.add_argument('--gru-epochs', type=int, default=20)
    train.add_argument('--split-manifest', type=Path)
    prepare = commands.add_parser('prepare-gsdc', help='Parse GSDC derived CSVs, run positioning and align references')
    prepare.add_argument('--root', type=Path, required=True)
    prepare.add_argument('--out', type=Path, default=Path('data/processed/gsdc'))
    prepare.add_argument('--max-reference-gap-ms', type=float, default=2000)
    klt = commands.add_parser('prepare-klt', help='Adapt the published real LOS/NLOS-labelled KLT feature table')
    klt.add_argument('--csv', type=Path, default=Path('data/raw/klt_features/nlos_features.csv.gz'))
    klt.add_argument('--out', type=Path, default=Path('data/processed/klt'))
    nlos = commands.add_parser('train-nlos', help='Train satellite NLOS detection with disjoint area groups')
    nlos.add_argument('--csv', type=Path, default=Path('data/processed/klt/satellites.csv'))
    nlos.add_argument('--out', type=Path, default=Path('artifacts/klt_nlos'))
    nlos.add_argument('--split-manifest', type=Path, default=Path('config/klt_split.json'))
    nlos.add_argument('--seed', type=int, default=42)
    nlos.add_argument('--with-transformer', action='store_true', help='Include causal per-satellite Transformer (requires PyTorch)')
    nlos.add_argument('--transformer-epochs', type=int, default=30)
    nlos.add_argument('--with-neural-models', action='store_true', help='Compare MLP, CNN, LSTM, GRU and Transformer (requires PyTorch)')
    nlos.add_argument('--neural-epochs', type=int, default=30)
    nlos.add_argument('--protocol', choices=['area_disjoint', 'in_domain', 'cross_region'], default='area_disjoint')
    protocols = commands.add_parser('compare-nlos-protocols', help='Run fixed-parameter paired in-domain/cross-region experiments')
    protocols.add_argument('--csv', type=Path, default=Path('data/processed/klt_extended/satellites.csv'))
    protocols.add_argument('--out', type=Path, default=Path('artifacts/klt_protocols'))
    protocols.add_argument('--plan', type=Path, default=Path('config/nlos_protocols.json'))
    generalization = commands.add_parser('evaluate-nlos-regions', help='Evaluate frozen source models and whole-drive unseen-district experiments')
    generalization.add_argument('--csv', type=Path, default=Path('data/processed/urbannav/satellites.csv'))
    generalization.add_argument('--out', type=Path, default=Path('artifacts/urbannav_generalization'))
    generalization.add_argument('--plan', type=Path, default=Path('config/nlos_generalization.json'))
    tuning = commands.add_parser('tune-nlos', help='Bounded validation-only hyperparameter search for all NLOS models')
    tuning.add_argument('--csv', type=Path, default=Path('data/processed/klt/satellites.csv'))
    tuning.add_argument('--out', type=Path, default=Path('artifacts/klt_nlos_tuned'))
    tuning.add_argument('--split-manifest', type=Path, default=Path('config/klt_split.json'))
    tuning.add_argument('--search-plan', type=Path, default=Path('config/nlos_search.json'))
    tuning.add_argument('--baseline-report', type=Path, default=Path('artifacts/klt_nlos/report.json'))
    tuning.add_argument('--seed', type=int, default=42)
    predict = commands.add_parser('predict', help='Predict risk from an observation table without reference errors')
    predict.add_argument('--csv', type=Path, required=True)
    predict.add_argument('--model', type=Path, required=True)
    predict.add_argument('--out', type=Path, required=True)
    agent = commands.add_parser('ask', help='Run offline diagnostics or a configured API tool-calling agent')
    agent.add_argument('question')
    agent.add_argument('--artifacts', type=Path, default=Path('artifacts/navigation_demo'))
    agent.add_argument('--provider', choices=['offline', 'openai', 'deepseek', 'compatible'], default=None)
    agent.add_argument('--env-file', type=Path, default=Path('.env'))
    agent.add_argument('--connection', help='Saved connection ID from the local dashboard')
    agent.add_argument('--trip', default=None)
    agent.add_argument('--nlos-artifacts', type=Path, help='NLOS fold or protocol suite; defaults to the current local suite when present')
    agent.add_argument('--generalization-artifacts', type=Path)
    agent.add_argument('--task-context', choices=['satellite_nlos', 'position_error'])
    agent.add_argument('--out', type=Path)
    evaluation = commands.add_parser('evaluate-agent', help='Check diagnostic tool routing, citations and note retrieval')
    evaluation.add_argument('--artifacts', type=Path, default=Path('artifacts/navigation_demo'))
    evaluation.add_argument('--out', type=Path, default=Path('artifacts/agent_eval.json'))
    evaluation.add_argument('--provider', choices=['offline', 'openai', 'deepseek', 'compatible'], default=None)
    evaluation.add_argument('--env-file', type=Path, default=Path('.env'))
    evaluation.add_argument('--connection', help='Evaluate with a saved dashboard connection')
    evaluation.add_argument('--cases', type=Path)
    evaluation.add_argument('--nlos-artifacts', type=Path)
    evaluation.add_argument('--generalization-artifacts', type=Path)
    ui = commands.add_parser('serve', help='Open a local experiment dashboard')
    ui.add_argument('--artifacts', type=Path, default=Path('artifacts/navigation_demo'))
    ui.add_argument('--port', type=int, default=8765)
    ui.add_argument('--env-file', type=Path, default=Path('.env'))
    ui.add_argument('--nlos-artifacts', type=Path, default=Path('artifacts/klt_protocols'))
    ui.add_argument('--generalization-artifacts', type=Path, default=Path('artifacts/urbannav_generalization'))
    diagnose = commands.add_parser("diagnose", help="Run the offline, read-only evidence tool")
    diagnose.add_argument("--artifacts", type=Path, default=Path("artifacts/demo"))
    diagnose.add_argument("--trip", default=None)
    args = parser.parse_args()
    try:
        if args.command == 'prepare-klt':
            from navdiag.nlos import prepare_klt
            print(json.dumps(prepare_klt(args.csv, args.out), indent=2, ensure_ascii=False))
            return
        if args.command == 'tune-nlos':
            from navdiag.nlos_search import tune_nlos
            report = tune_nlos(args.csv, args.out,
                              json.loads(args.split_manifest.read_text(encoding='utf-8')),
                              json.loads(args.search_plan.read_text(encoding='utf-8')),
                              args.seed, args.baseline_report, progress=True)
            print(f"Validation-selected model: {report['selected_model']}")
            for name, result in report['models'].items():
                display = lambda value: 'N/A' if value is None else f'{value:.4f}'
                print(f"{name:22s} Val F1={display(result['validation']['f1'])} Test F1={display(result['test']['f1'])}")
            return
        if args.command == 'compare-nlos-protocols':
            from navdiag.nlos_protocols import run_protocols
            result = run_protocols(args.csv, args.out, json.loads(args.plan.read_text(encoding='utf-8')), progress=True)
            for protocol, summary in result['protocols'].items():
                print(protocol, 'validation-selected:', summary['selected_model'])
                for name, model in summary['models'].items():
                    print(f"{name:22s} macro Test F1={model['test_macro']['f1']:.4f}")
            return
        if args.command == 'evaluate-nlos-regions':
            from navdiag.nlos_external import run_generalization
            report = run_generalization(args.csv, args.out, json.loads(args.plan.read_text(encoding='utf-8')), progress=True)
            for section in ['external_transfer', 'district_holdout']:
                print(section, 'validation-selected:', report[section]['selected_model'])
                for name, result in report[section]['models'].items():
                    print(f"{name:22s} macro Test F1={result['test_macro']['f1']:.4f}")
            return
        if args.command == 'train-nlos':
            from navdiag.nlos import train_nlos
            report = train_nlos(args.csv, args.out,
                                json.loads(args.split_manifest.read_text(encoding='utf-8')), args.seed,
                                args.with_transformer, args.transformer_epochs,
                                args.with_neural_models, args.neural_epochs, protocol=args.protocol)
            print(f"Task: satellite NLOS | validation-selected model: {report['selected_model']}")
            for name, result in report['models'].items():
                v = result['test']
                display = lambda value: 'N/A' if value is None else f'{value:.3f}'
                print(f"{name:22} precision={display(v['precision'])} recall={display(v['recall'])} F1={display(v['f1'])}")
            print(report['warning'])
            print(f'Artifacts: {args.out.resolve()}')
            return
        if args.command == 'prepare-gsdc':
            from navdiag.gsdc import preprocess_root
            print(json.dumps(preprocess_root(args.root, args.out, args.max_reference_gap_ms), indent=2, ensure_ascii=False))
            return
        if args.command == 'predict':
            import joblib
            import pandas as pd
            from navdiag.learning import predict_frame
            frame = pd.read_csv(args.csv, dtype={'trip_id': 'string'})
            bundle = joblib.load(args.model)
            if bundle.get('task_type') == 'satellite_nlos':
                from navdiag.nlos import nlos_scores
                frame['predicted_nlos_score'] = nlos_scores(bundle, frame)
                frame['predicted_nlos'] = frame.predicted_nlos_score >= bundle['threshold']
                args.out.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(args.out, index=False)
                print(f'NLOS predictions: {args.out.resolve()}')
                return
            frame['predicted_risk'] = predict_frame(bundle, frame)
            frame['predicted_unreliable'] = frame.predicted_risk >= bundle['threshold']
            args.out.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(args.out, index=False)
            print(f'Predictions: {args.out.resolve()}')
            return
        if args.command == 'ask':
            from navdiag.agent import LLMConfig, ask
            from navdiag.llm_connections import ConnectionStore, connection_path
            if args.connection and args.provider:
                raise ValueError('Choose --connection or --provider, not both')
            config = (ConnectionStore(connection_path(args.env_file)).config(args.connection) if args.connection
                      else LLMConfig.from_environment(args.provider, args.env_file))
            result = ask(args.question, args.artifacts, config, args.trip,
                         args.nlos_artifacts, args.generalization_artifacts, args.task_context)
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return
        if args.command == 'evaluate-agent':
            from navdiag.agent import LLMConfig
            from navdiag.agent_eval import evaluate_agent
            from navdiag.llm_connections import ConnectionStore, connection_path
            if args.connection and args.provider:
                raise ValueError('Choose --connection or --provider, not both')
            config = (ConnectionStore(connection_path(args.env_file)).config(args.connection) if args.connection
                      else LLMConfig.from_environment(args.provider, args.env_file))
            report = evaluate_agent(args.artifacts, args.out, config, args.cases,
                                    args.nlos_artifacts, args.generalization_artifacts)
            print(f"Backend: {report['backend']} | cases: {report['passed']}/{report['cases']} | {args.out.resolve()}")
            print(report['limitations'])
            if report['passed'] != report['cases']:
                parser.exit(1)
            return
        if args.command == 'serve':
            from navdiag.dashboard import serve
            serve(args.artifacts, args.port, args.env_file, args.nlos_artifacts, args.generalization_artifacts)
            return
        if args.command == "diagnose":
            print(json.dumps(summarize_trip(args.artifacts, args.trip), indent=2, ensure_ascii=False))
            return
        if args.command == "demo":
            if args.navigation:
                from navdiag.navigation_demo import generate_navigation_demo
                csv_path = generate_navigation_demo(args.out / 'synthetic_navigation.csv', seed=args.seed)
            else:
                csv_path = generate_demo(args.out / "synthetic_epochs.csv", seed=args.seed)
            report = train_and_evaluate(csv_path, args.out, seed=args.seed, with_gru=args.with_gru, gru_epochs=args.gru_epochs)
        else:
            report = train_and_evaluate(args.csv, args.out, args.error_limit_m, args.seed,
                                       args.allow_missing_features, args.with_gru, args.gru_epochs,
                                       json.loads(args.split_manifest.read_text(encoding='utf-8')) if args.split_manifest else None)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(f"Data: {report['data_kind']} | model selected on validation: {report['selected_model']}")
    print(report["warning"])
    for name, result in report["models"].items():
        metrics = result["test"]
        display = lambda value: 'N/A' if value is None else f'{value:.3f}'
        print(f"{name:22} precision={display(metrics['precision'])} recall={display(metrics['recall'])} F1={display(metrics['f1'])}")
    print(f"Artifacts: {args.out.resolve()}")
