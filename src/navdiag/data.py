"""Start here: features are inputs; reference errors only create labels."""

from pathlib import Path

import numpy as np
import pandas as pd

# Keep this explicit: ground truth, trip IDs, and timestamps cannot become inputs.
FEATURES = [
    "satellite_count",
    "cn0_mean_dbhz",
    "cn0_min_dbhz",
    "hdop",
    "residual_rms_m",
    "speed_mps",
]
ERROR_COLUMN = "horizontal_error_m"
REQUIRED = ["trip_id", "timestamp_ms", "data_kind", *FEATURES, ERROR_COLUMN]


def generate_demo(path: Path, seed: int = 42, trips: int = 15, epochs: int = 180) -> Path:
    """Generate toy observations, NOT a physical GNSS simulator or real benchmark.

    Nearby epochs share an environment state. This lets us teach why randomly
    splitting rows can leak information between training and testing.
    """
    if trips < 5 or epochs < 10:
        raise ValueError("Use at least 5 trips and 10 epochs per trip.")
    rng = np.random.default_rng(seed)
    rows = []
    for trip in range(trips):
        state = rng.uniform(0.1, 0.9)
        receiver_bias = rng.normal(0, 0.7)
        for epoch in range(epochs):
            if epoch % 30 == 0:
                state = rng.uniform(0, 1)
            state = float(np.clip(state + rng.normal(0, 0.025), 0, 1))
            cn0 = float(np.clip(43 - 17 * state + rng.normal(0, 2), 15, 50))
            hdop = float(np.clip(1 + 5 * state + rng.normal(0, 0.5), 0.5, 12))
            residual = max(0.1, 1 + 15 * state + rng.normal(0, 2))
            # Deliberately learnable toy label; good scores here prove no real-world accuracy.
            error = max(0, 0.6 * residual + hdop + receiver_bias + rng.normal(0, 2))
            rows.append({
                "trip_id": f"demo_trip_{trip:02d}",
                "timestamp_ms": 1700000000000 + trip * 1000000 + epoch * 1000,
                "data_kind": "synthetic",
                "satellite_count": int(np.clip(round(22 - 10 * state + rng.normal(0, 2)), 5, 30)),
                "cn0_mean_dbhz": cn0,
                "cn0_min_dbhz": max(10, cn0 - rng.uniform(3, 10)),
                "hdop": hdop,
                "residual_rms_m": residual,
                "speed_mps": max(0, 9 + rng.normal(0, 4)),
                ERROR_COLUMN: error,
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def load_dataset(path: Path, allow_missing_features: bool = False) -> pd.DataFrame:
    """Validate an epoch-level table before fitting any model."""
    frame = pd.read_csv(path, dtype={"trip_id": "string"})
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Dataset is empty.")
    if frame["trip_id"].isna().any() or frame["trip_id"].str.strip().eq("").any():
        raise ValueError("Every row needs a non-empty trip_id.")
    kinds = set(frame["data_kind"].dropna())
    if frame["data_kind"].isna().any() or len(kinds) != 1 or not kinds <= {"synthetic", "real"}:
        raise ValueError("Use exactly one data_kind: synthetic or real.")
    for name in ["timestamp_ms", *FEATURES, ERROR_COLUMN]:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        values = frame[name].to_numpy(dtype=float)
        allowed_missing = allow_missing_features and name in FEATURES
        if np.isinf(values).any() or (not allowed_missing and np.isnan(values).any()):
            raise ValueError(f"Non-finite or missing values in {name}.")
    if (frame[ERROR_COLUMN] < 0).any():
        raise ValueError("Horizontal error cannot be negative.")
    row_key = ["trip_id", "timestamp_ms"]
    if "device_id" in frame:
        if frame["device_id"].isna().any():
            raise ValueError("device_id cannot be missing.")
        row_key.insert(1, "device_id")
    if frame.duplicated(row_key).any():
        raise ValueError("Duplicate trip_id/timestamp_ms; aggregate satellite rows first.")
    return frame.sort_values(row_key).reset_index(drop=True)


def split_by_trip(frame: pd.DataFrame, seed: int = 42, manifest: dict | None = None) -> dict[str, pd.DataFrame]:
    """Whole trips stay in one split, including all phones from the same drive."""
    trips = np.array(sorted(frame["trip_id"].unique()))
    if len(trips) < 3:
        raise ValueError("Need at least 3 independent trips for train/validation/test.")
    if manifest is not None:
        if not isinstance(manifest, dict) or set(manifest) != {'train', 'validation', 'test'}:
            raise ValueError('Split manifest must define train, validation and test')
        if any(not isinstance(manifest[name], list) or not manifest[name] for name in manifest):
            raise ValueError('Each split must list at least one trip')
        all_ids = [trip for name in ['train', 'validation', 'test'] for trip in manifest[name]]
        if any(not isinstance(trip, str) for trip in all_ids):
            raise ValueError('Trip IDs in a split manifest must be strings')
        if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(trips):
            raise ValueError('Split IDs must cover every trip exactly once, without overlap')
        return {name: frame[frame.trip_id.isin(ids)].copy() for name, ids in manifest.items()}
    np.random.default_rng(seed).shuffle(trips)
    held_out = max(1, len(trips) // 5)
    groups = {
        "test": trips[:held_out],
        "validation": trips[held_out:2 * held_out],
        "train": trips[2 * held_out:],
    }
    return {name: frame[frame["trip_id"].isin(ids)].copy() for name, ids in groups.items()}
