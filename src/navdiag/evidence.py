"""Read-only observation summaries for diagnostics; reference labels stay offline."""

import json
from pathlib import Path

import pandas as pd


def summarize_trip(artifact_dir: Path, trip_id: str | None = None) -> dict:
    report = json.loads((artifact_dir / "report.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(artifact_dir / "predictions.csv", dtype={"trip_id": "string"})
    available = sorted(predictions["trip_id"].unique().tolist())
    selected = available[0] if trip_id is None else trip_id
    part = predictions[predictions["trip_id"] == selected]
    if part.empty:
        raise ValueError(f"Unknown trip_id {selected!r}; available: {available}")
    high_risk = part[part["predicted_unreliable"]]
    # Reference errors are intentionally excluded: online diagnosis cannot see truth.
    payload = {
        "tool": "summarize_trip", "data_kind": report["data_kind"], "trip_id": selected,
        "model": report["selected_model"], "epochs": len(part),
        "flagged_epochs": len(high_risk), "flagged_fraction": len(high_risk) / len(part),
        "observed_cn0_mean_dbhz": float(part["cn0_mean_dbhz"].mean()),
        "observed_hdop_mean": float(part["hdop"].mean()),
        "observed_residual_rms_mean_m": float(part["residual_rms_m"].mean()),
        "evidence_source": "predictions.csv (observations and predictions only)",
        "interpretation": "Low signal quality or poor geometry may accompany risk; these summaries do not establish an NLOS cause.",
        "warning": report["warning"],
    }
    return json.loads(pd.Series([payload]).to_json(orient='values'))[0]
