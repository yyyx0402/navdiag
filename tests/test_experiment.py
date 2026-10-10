import json

import numpy as np

from navdiag.data import generate_demo
from navdiag.evidence import summarize_trip
from navdiag.learning import evaluate, train_and_evaluate


def test_confusion_matrix_explains_false_alarms_and_missed_failures():
    metrics = evaluate(np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.2, 0.9]), 0.5)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5


def test_single_class_test_set_has_no_ranking_metrics():
    metrics = evaluate(np.array([0, 0]), np.array([0.1, 0.2]), 0.5)
    assert metrics["roc_auc"] is None
    assert metrics["average_precision"] is None


def test_full_experiment_preserves_provenance_and_chooses_on_validation(tmp_path):
    path = generate_demo(tmp_path / "demo.csv")
    out = tmp_path / "artifacts"
    report = train_and_evaluate(path, out)
    assert report["data_kind"] == "synthetic"
    assert len(report["dataset_sha256"]) == 64
    winner = report["selected_model"]
    best_validation_f1 = max(model["validation"]["f1"] for model in report["models"].values())
    assert report["models"][winner]["validation"]["f1"] == best_validation_f1
    assert (out / "risk_plot.png").stat().st_size > 0
    assert json.loads((out / "report.json").read_text(encoding="utf-8"))["selected_model"] == winner
    evidence = summarize_trip(out)
    assert "horizontal_error_m" not in evidence
    assert "is_unreliable" not in evidence
    assert evidence["data_kind"] == "synthetic"
    assert 0 <= evidence["flagged_fraction"] <= 1
