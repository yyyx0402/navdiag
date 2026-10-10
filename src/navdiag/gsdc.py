"""Adapt GSDC 2021/2022/2023 derived CSVs to observations and labelled epochs."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from navdiag.positioning import align_reference, ecef_to_enu, ecef_to_lla, lla_to_ecef, solve_wls

GPS_EPOCH_UNIX_MS = 315964800000
GPS_MINUS_UTC_MS = 18000  # Applicable to the 2020-2024 GSDC collections, not future leap seconds.


def _column(frame: pd.DataFrame, names: list[str], default: float | None = None) -> np.ndarray:
    for name in names:
        if name in frame:
            return pd.to_numeric(frame[name], errors="coerce").to_numpy()
    if default is not None:
        return np.full(len(frame), default)
    raise ValueError(f"Required field missing; expected one of {names}")


def read_observations(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "millisSinceGpsEpoch" in frame:
        times = _column(frame, ["millisSinceGpsEpoch"]) + GPS_EPOCH_UNIX_MS - GPS_MINUS_UTC_MS
    else:
        times = _column(frame, ["utcTimeMillis", "UnixTimeMillis"])
    raw = _column(frame, ["rawPrM", "RawPseudorangeMeters"])
    corrected = (raw + _column(frame, ["satClkBiasM", "SvClockBiasMeters"])
                 - _column(frame, ["isrbM", "IsrbMeters"])
                 - _column(frame, ["ionoDelayM", "IonosphericDelayMeters"])
                 - _column(frame, ["tropoDelayM", "TroposphericDelayMeters"]))
    observations = pd.DataFrame({
        "timestamp_ms": times,
        "system": _column(frame, ["constellationType", "ConstellationType"]),
        "sv_id": _column(frame, ["svid", "Svid"]),
        "x_m": _column(frame, ["xSatPosM", "SvPositionXEcefMeters"]),
        "y_m": _column(frame, ["ySatPosM", "SvPositionYEcefMeters"]),
        "z_m": _column(frame, ["zSatPosM", "SvPositionZEcefMeters"]),
        "pseudorange_m": corrected,
        "sigma_m": _column(frame, ["rawPrUncM", "RawPseudorangeUncertaintyMeters"]),
        "cn0_dbhz": _column(frame, ["Cn0DbHz", "cn0DbHz"], np.nan),
    })
    essential = ["timestamp_ms", "system", "sv_id", "x_m", "y_m", "z_m", "pseudorange_m", "sigma_m"]
    observations = observations[np.isfinite(observations[essential]).all(axis=1)]
    observations = observations[(observations.pseudorange_m.between(1e7, 5e7)) & (observations.sigma_m > 0)]
    # One independent code observation per satellite: do not count frequencies as extra satellites.
    observations = observations.sort_values("sigma_m").drop_duplicates(["timestamp_ms", "system", "sv_id"])
    return observations.sort_values(["timestamp_ms", "system", "sv_id"]).reset_index(drop=True)


def read_reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    if "millisSinceGpsEpoch" in frame:
        times = _column(frame, ["millisSinceGpsEpoch"]) + GPS_EPOCH_UNIX_MS - GPS_MINUS_UTC_MS
    else:
        times = _column(frame, ["UnixTimeMillis", "utcTimeMillis"])
    lla = np.column_stack([
        _column(frame, ["latDeg", "LatitudeDegrees"]),
        _column(frame, ["lngDeg", "LongitudeDegrees"]),
        _column(frame, ["heightAboveWgs84EllipsoidM", "AltitudeMeters"]),
    ])
    valid = np.isfinite(times) & np.isfinite(lla).all(axis=1)
    return times[valid], lla_to_ecef(lla[valid])


def preprocess_device(observation_path: Path, reference_path: Path, trip_id: str,
                      device_id: str, max_reference_gap_ms: float = 2000) -> tuple[pd.DataFrame, dict]:
    observations = read_observations(observation_path)
    rows, failures = [], Counter()
    previous, previous_time = None, None
    for timestamp, group in observations.groupby("timestamp_ms", sort=True):
        satellites = group[["x_m", "y_m", "z_m"]].to_numpy()
        pr, systems = group.pseudorange_m.to_numpy(), group.system.to_numpy()
        try:
            equal = solve_wls(satellites, pr, systems, initial_xyz=previous)
            weighted = solve_wls(satellites, pr, systems, group.sigma_m.to_numpy(), equal.xyz)
            try:
                robust = solve_wls(satellites, pr, systems, group.sigma_m.to_numpy(), weighted.xyz, robust=True)
            except ValueError:
                robust = weighted
                failures["robust_fallback"] += 1
        except (ValueError, np.linalg.LinAlgError) as error:
            failures[str(error)] += 1
            continue
        speed = 0.0
        if previous is not None and 0 < timestamp - previous_time < 5000:
            speed = min(70.0, np.linalg.norm(weighted.xyz - previous) / ((timestamp - previous_time) / 1000))
        cn0 = group.cn0_dbhz.dropna()
        row = {"trip_id": trip_id, "device_id": device_id, "timestamp_ms": int(timestamp), "data_kind": "real",
               "satellite_count": len(group), "cn0_mean_dbhz": float(cn0.mean()) if len(cn0) else np.nan,
               "cn0_min_dbhz": float(cn0.min()) if len(cn0) else np.nan,
               "hdop": weighted.hdop, "residual_rms_m": float(np.sqrt(np.mean(weighted.residuals_m ** 2))),
               "speed_mps": speed}
        for name, solution in [("equal", equal), ("wls", weighted), ("robust", robust)]:
            row.update({f"{name}_{axis}_m": float(value) for axis, value in zip("xyz", solution.xyz)})
        rows.append(row)
        previous, previous_time = weighted.xyz, timestamp
    if not rows:
        raise ValueError(f"No valid positioning epochs in {observation_path}; failures: {dict(failures)}")
    result = pd.DataFrame(rows)
    reference_times, reference_xyz = read_reference(reference_path)
    truth, matched = align_reference(result.timestamp_ms.to_numpy(), reference_times, reference_xyz, max_reference_gap_ms)
    failures["reference_not_matched"] = int((~matched).sum())
    result = result.loc[matched].copy().reset_index(drop=True)
    truth = truth[matched]
    if result.empty:
        raise ValueError(f"No matched reference epochs in {observation_path}")
    # Origin comes from the first GNSS solution, never from reference truth.
    origin = ecef_to_lla(result[["wls_x_m", "wls_y_m", "wls_z_m"]].iloc[0].to_numpy())
    for axis, value in zip(["origin_lat_deg", "origin_lon_deg", "origin_alt_m"], origin):
        result[axis] = float(value)
    result[["gt_e_m", "gt_n_m", "gt_u_m"]] = ecef_to_enu(truth, origin)
    for name in ["equal", "wls", "robust"]:
        xyz = result[[f"{name}_{axis}_m" for axis in "xyz"]].to_numpy()
        result[[f"{name}_e_m", f"{name}_n_m", f"{name}_u_m"]] = ecef_to_enu(xyz, origin)
        result[f"{name}_error_m"] = np.linalg.norm(ecef_to_enu(xyz, origin)[:, :2] - result[["gt_e_m", "gt_n_m"]].to_numpy(), axis=1)
    result["horizontal_error_m"] = result["wls_error_m"]
    details = {
        "trip_id": trip_id, "device_id": device_id, "epochs": len(result), "dropped": dict(failures),
        "unavailable_features": [name for name in ["cn0_mean_dbhz", "cn0_min_dbhz"] if result[name].isna().all()],
        "source_files": [{"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in [observation_path, reference_path]],
        "time_policy": "2021 GPS epoch milliseconds converted with 18s GPS-UTC; 2022/2023 provided Unix milliseconds; no extrapolation",
    }
    return result, details


def preprocess_root(root: Path, output: Path, max_reference_gap_ms: float = 2000) -> dict:
    files = sorted(set(root.rglob("device_gnss.csv")) | set(root.rglob("*_derived.csv")))
    if not files:
        raise ValueError("No device_gnss.csv or *_derived.csv files found.")
    frames, devices, failed = [], [], []
    for path in files:
        reference = path.parent / "ground_truth.csv"
        if not reference.exists():
            failed.append({"file": str(path.relative_to(root)), "reason": "ground_truth.csv absent"})
            continue
        try:
            frame, details = preprocess_device(path, reference, path.parent.parent.name, path.parent.name, max_reference_gap_ms)
            frames.append(frame)
            devices.append(details)
        except ValueError as error:
            failed.append({"file": str(path.relative_to(root)), "reason": str(error)})
    if not frames:
        raise ValueError(f"All input devices failed: {failed}")
    output.mkdir(parents=True, exist_ok=True)
    all_epochs = pd.concat(frames, ignore_index=True)
    all_epochs.to_csv(output / "epochs.csv", index=False)
    report = {"data_kind": "real", "epochs": len(all_epochs), "trips": all_epochs.trip_id.nunique(),
              "devices": devices, "failed_files": failed, "max_reference_gap_ms": max_reference_gap_ms,
              "warning": "Check collection independence and source licensing. Published tiny parser samples are not a representative benchmark."}
    (output / "preprocess_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report
