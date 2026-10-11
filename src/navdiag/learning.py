"""Three baselines and an honest evaluation: select on validation, report on test."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from navdiag.data import ERROR_COLUMN, FEATURES, load_dataset, split_by_trip


def rule_scores(frame: pd.DataFrame) -> np.ndarray:
    """An intentionally simple, fixed baseline; thresholds are not tuned on test."""
    poor = ((frame["cn0_mean_dbhz"] < 30) | (frame["hdop"] > 4)
            | (frame["residual_rms_m"] > 12))
    return poor.to_numpy(dtype=float)


def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Choose the F1 threshold on VALIDATION only; freeze it before testing."""
    choices = np.arange(0.05, 0.96, 0.05)
    return float(max(choices, key=lambda t: (
        f1_score(labels, scores >= t, zero_division=0), -abs(t - 0.5),
    )))


def evaluate(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = scores >= threshold
    both_classes = len(np.unique(labels)) == 2
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "threshold": threshold,
        "rows": len(labels),
        "positives": int(tp + fn),
        "predicted_positives": int(tp + fp),
        "true_positives": int(tp), "true_negatives": int(tn),
        "false_positives": int(fp), "false_negatives": int(fn),
        "accuracy": float((tp + tn) / len(labels)),
        "balanced_accuracy": float((tp / (tp + fn) + tn / (tn + fp)) / 2) if both_classes else None,
        "precision": float(precision_score(labels, predicted, zero_division=0)) if tp + fp else None,
        "recall": float(recall_score(labels, predicted, zero_division=0)) if tp + fn else None,
        "f1": float(f1_score(labels, predicted, zero_division=0)) if tp + fn else None,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "average_precision": float(average_precision_score(labels, scores)) if both_classes else None,
        "roc_auc": float(roc_auc_score(labels, scores)) if both_classes else None,
        "brier_score": float(brier_score_loss(labels, scores)),
        "confusion_matrix": matrix.tolist(),
    }


def train_and_evaluate(csv_path: Path, output: Path, error_limit_m: float = 10, seed: int = 42,
                       allow_missing_features: bool = False, with_gru: bool = False,
                       gru_epochs: int = 20, split_manifest: dict | None = None,
                       min_feature_coverage: float = 0.5) -> dict:
    if not np.isfinite(error_limit_m) or error_limit_m <= 0:
        raise ValueError("error_limit_m must be finite and positive.")
    frame = load_dataset(csv_path, allow_missing_features=allow_missing_features)
    if not 0 < min_feature_coverage <= 1:
        raise ValueError('min_feature_coverage must be in (0, 1]')
    splits = split_by_trip(frame, seed, split_manifest)
    active_features = [name for name in FEATURES if splits['train'][name].notna().mean() >= min_feature_coverage]
    if not active_features:
        raise ValueError("No observed training features.")
    labels = {name: (part[ERROR_COLUMN].to_numpy() > error_limit_m).astype(int)
              for name, part in splits.items()}
    for name in ["train", "validation"]:
        if len(np.unique(labels[name])) < 2:
            raise ValueError(f"{name} needs both classes. Add independent trips or review the task threshold.")

    models = {
        "logistic_regression": make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed)),
        "random_forest": make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(
            n_estimators=150, max_depth=8, min_samples_leaf=6,
            class_weight="balanced", random_state=seed, n_jobs=1)),
    }
    report = {
        "version": "0.3.0", "data_kind": str(frame["data_kind"].iloc[0]),
        "dataset_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "seed": seed, "sklearn_version": sklearn.__version__,
        "task": f"horizontal_error_m > {error_limit_m:g}",
        "error_limit_m": error_limit_m, "features": active_features,
        "missing_feature_fraction_train": {name: float(splits['train'][name].isna().mean()) for name in FEATURES},
        "dropped_features": [name for name in FEATURES if name not in active_features],
        "min_feature_coverage_train": min_feature_coverage,
        "evaluation_limitations": ['Test split has no unreliable epochs; recall and F1 cannot be assessed.'] if not labels['test'].sum() else [],
        "splits": {name: {"rows": len(part), "trips": sorted(part["trip_id"].unique().tolist()),
                           "positive_rate": float(labels[name].mean())} for name, part in splits.items()},
        "models": {},
        "task_type": "epoch_position_error",
        "label_definition": "Reference horizontal error only; no obstruction or LOS/NLOS ground truth.",
        "sanity_baselines": {name: {split: evaluate(labels[split], np.full(len(part), value), 0.5)
                                   for split, part in splits.items() if split != 'train'}
                             for name, value in [('always_reliable', 0), ('always_unreliable', 1)]},
        "warning": "Synthetic demo results are not real GNSS performance." if frame["data_kind"].iloc[0] == "synthetic"
                   else "Performance applies only to this dataset and split; check unseen routes/devices.",
    }
    test = splits["test"].copy()
    test["is_unreliable"] = labels["test"]
    rule_metrics = evaluate(labels["test"], rule_scores(test), 0.5)
    rule_validation = evaluate(labels["validation"], rule_scores(splits["validation"]), 0.5)
    report["models"]["fixed_rules"] = {"validation": rule_validation, "test": rule_metrics}
    test["risk_fixed_rules"] = rule_scores(test)
    for name, model in models.items():
        model.fit(splits["train"][active_features], labels["train"])
        validation_scores = model.predict_proba(splits["validation"][active_features])[:, 1]
        threshold = choose_threshold(labels["validation"], validation_scores)
        test_scores = model.predict_proba(test[active_features])[:, 1]
        test[f"risk_{name}"] = test_scores
        report["models"][name] = {
            "validation": evaluate(labels["validation"], validation_scores, threshold),
            "test": evaluate(labels["test"], test_scores, threshold),
        }
    if with_gru:
        from navdiag.temporal import train_gru
        model, history = train_gru(splits['train'], splits['validation'], active_features, error_limit_m,
                                   seed=seed, epochs=gru_epochs)
        models['gru'] = model
        validation_scores = model.predict_proba(splits['validation'])[:, 1]
        threshold = choose_threshold(labels['validation'], validation_scores)
        test_scores = model.predict_proba(test)[:, 1]
        test['risk_gru'] = test_scores
        report['models']['gru'] = {
            'validation': evaluate(labels['validation'], validation_scores, threshold),
            'test': evaluate(labels['test'], test_scores, threshold), 'training_history': history,
            'window_epochs': model.window,
        }
    # The winner can be the simple rule baseline. Test scores never pick the winner.
    selected = max(report["models"], key=lambda name: report["models"][name]["validation"]["f1"])
    report["selected_model"] = selected
    test["predicted_risk"] = test[f"risk_{selected}"]
    threshold = report["models"][selected]["validation"]["threshold"]
    test["predicted_unreliable"] = test["predicted_risk"] >= threshold
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    test.to_csv(output / "predictions.csv", index=False)
    bundle = {"models": models, "selected_model": selected, "threshold": threshold,
              "features": active_features, "data_kind": report["data_kind"], "report_version": '0.3.0'}
    joblib.dump(bundle, output / "models.joblib")
    if all(name in frame for name in ['wls_e_m', 'wls_n_m', 'gt_e_m', 'gt_n_m']):
        from navdiag.fusion import evaluate_positioning
        report['positioning'] = evaluate_positioning(splits['validation'], test, bundle)
        (output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        test.to_csv(output / 'predictions.csv', index=False)
    write_plot(test, report, output / "risk_plot.png")
    return report


def predict_frame(bundle: dict, frame: pd.DataFrame, name: str | None = None) -> np.ndarray:
    """Inference consumes observations only: reference error is never required."""
    selected = bundle['selected_model'] if name is None else name
    if selected != 'fixed_rules' and selected not in bundle['models']:
        raise ValueError(f'Unknown model: {selected}')
    required = ['cn0_mean_dbhz', 'hdop', 'residual_rms_m'] if selected == 'fixed_rules' else bundle['features']
    if selected == 'gru':
        required = [*required, 'trip_id', 'timestamp_ms']
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f'Missing inference columns: {sorted(missing)}')
    frame = frame.copy()
    for column in required:
        if column == 'trip_id':
            if frame[column].isna().any() or frame[column].astype(str).str.strip().eq('').any():
                raise ValueError('Inference trip IDs cannot be empty')
            continue
        frame[column] = pd.to_numeric(frame[column], errors='raise')
        values = frame[column].to_numpy(dtype=float)
        if np.isinf(values).any() or (column == 'timestamp_ms' and np.isnan(values).any()):
            raise ValueError(f'Non-finite inference values: {column}')
    if selected == 'gru':
        keys = ['trip_id', 'timestamp_ms']
        if 'device_id' in frame:
            keys.insert(1, 'device_id')
            if frame.device_id.isna().any():
                raise ValueError('Inference device IDs cannot be missing')
        if frame.duplicated(keys).any():
            raise ValueError('Duplicate inference epochs')
    if frame.empty:
        return np.array([], dtype=float)
    if selected == 'fixed_rules':
        return rule_scores(frame)
    if selected == 'gru':
        return bundle['models'][selected].predict_proba(frame)[:, 1]
    return bundle['models'][selected].predict_proba(frame[bundle['features']])[:, 1]


def write_plot(test: pd.DataFrame, report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trip = sorted(test["trip_id"].unique())[0]
    part = test[test["trip_id"] == trip].sort_values("timestamp_ms")
    seconds = (part["timestamp_ms"] - part["timestamp_ms"].iloc[0]) / 1000
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    axes[0].plot(seconds, part[ERROR_COLUMN], label="Reference horizontal error")
    axes[0].axhline(report["error_limit_m"], color="tab:red", linestyle="--", label="Task threshold")
    axes[0].set_ylabel("Error (m)")
    axes[0].legend()
    axes[1].plot(seconds, part["predicted_risk"], color="tab:orange", label="Predicted risk score")
    axes[1].set_ylabel("Risk score")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel("Time from trip start (s)")
    axes[1].legend()
    fig.suptitle(f"{report['data_kind'].upper()} | {trip} | {report['selected_model']}")
    fig.savefig(path, dpi=160)
    plt.close(fig)
