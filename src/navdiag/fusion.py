"""Causal planar position/velocity Kalman filter, optionally with ENU acceleration."""
import numpy as np
import pandas as pd


def filter_positions(times_ms: np.ndarray, positions_en: np.ndarray, variances_m2: np.ndarray,
                     acceleration_en: np.ndarray | None = None, acceleration_noise: float = 2.0,
                     gate_nis: float = 100, reset_gap_s: float = 10) -> np.ndarray:
    """Reference positions are not accepted by this interface.

    Optional acceleration must already be gravity-compensated, bias-corrected,
    time-aligned, and rotated into ENU. This is a planar baseline, not a strapdown INS.
    """
    times_ms, positions_en, variances_m2 = map(np.asarray, [times_ms, positions_en, variances_m2])
    if positions_en.shape != (len(times_ms), 2) or variances_m2.shape != (len(times_ms),):
        raise ValueError('Position, time and observation variance arrays are not aligned')
    if not np.isfinite(times_ms).all() or np.any(np.diff(times_ms) <= 0):
        raise ValueError('Filter timestamps must be finite and strictly increasing')
    if not np.isfinite(variances_m2).all() or np.any(variances_m2 <= 0):
        raise ValueError('Observation variances must be positive and finite')
    if acceleration_en is not None:
        acceleration_en = np.asarray(acceleration_en)
        if acceleration_en.shape != positions_en.shape or not np.isfinite(acceleration_en).all():
            raise ValueError('ENU accelerations must be finite and aligned')
    observation = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
    state, covariance = None, None
    output = np.full_like(positions_en, np.nan, dtype=float)
    for i, fix in enumerate(positions_en):
        valid = np.isfinite(fix).all()
        dt = 0 if i == 0 else (times_ms[i] - times_ms[i - 1]) / 1000
        if state is None or dt > reset_gap_s:
            if not valid:
                state = None
                continue
            state = np.array([*fix, 0.0, 0.0])
            covariance = np.diag([variances_m2[i], variances_m2[i], 100, 100])
            output[i] = fix
            continue
        transition = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        control = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]], float)
        state = transition @ state
        if acceleration_en is not None:
            state += control @ acceleration_en[i]
        covariance = transition @ covariance @ transition.T + acceleration_noise ** 2 * control @ control.T
        if valid:
            noise = np.eye(2) * variances_m2[i]
            innovation = fix - observation @ state
            innovation_covariance = observation @ covariance @ observation.T + noise
            nis = float(innovation @ np.linalg.solve(innovation_covariance, innovation))
            if nis <= gate_nis:
                gain = np.linalg.solve(innovation_covariance, observation @ covariance).T
                state += gain @ innovation
                stable = np.eye(4) - gain @ observation
                covariance = stable @ covariance @ stable.T + gain @ noise @ gain.T
        output[i] = state[:2]
    return output


def _apply(frame: pd.DataFrame, variances: np.ndarray, acceleration_noise: float) -> np.ndarray:
    result = np.empty((len(frame), 2))
    positions = frame.reset_index(drop=True)
    keys = ['trip_id'] + (['device_id'] if 'device_id' in frame else [])
    for _, indices in positions.groupby(keys, sort=False).groups.items():
        ordered = positions.loc[indices].sort_values('timestamp_ms').index.to_numpy()
        part = positions.loc[ordered]
        acceleration = None
        if {'imu_accel_e_mps2', 'imu_accel_n_mps2'} <= set(part):
            acceleration = part[['imu_accel_e_mps2', 'imu_accel_n_mps2']].to_numpy()
        result[ordered] = filter_positions(part.timestamp_ms.to_numpy(), part[['wls_e_m', 'wls_n_m']].to_numpy(),
                                          variances[ordered], acceleration, acceleration_noise)
    return result


def position_metrics(errors: np.ndarray) -> dict:
    values = np.asarray(errors, float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {'epochs': 0, 'median_m': None, 'p95_m': None, 'rmse_m': None,
                'within_error_limits_m': {str(limit): None for limit in [1, 2, 5, 10]}}
    return {'epochs': len(values), 'median_m': float(np.median(values)),
            'p95_m': float(np.quantile(values, 0.95)), 'rmse_m': float(np.sqrt(np.mean(values ** 2))),
            'within_error_limits_m': {str(limit): float(np.mean(values <= limit)) for limit in [1, 2, 5, 10]}}


def evaluate_positioning(validation: pd.DataFrame, test: pd.DataFrame, bundle: dict) -> dict:
    from navdiag.learning import predict_frame
    truth_val = validation[['gt_e_m', 'gt_n_m']].to_numpy()
    best_settings, best_rmse = None, float('inf')
    for sigma in [2, 5, 10, 20]:
        for acceleration_noise in [1, 3]:
            filtered = _apply(validation, np.full(len(validation), sigma ** 2), acceleration_noise)
            score = np.sqrt(np.mean(np.sum((filtered - truth_val) ** 2, axis=1)))
            if score < best_rmse:
                best_rmse, best_settings = score, (sigma, acceleration_noise)
    sigma, acceleration_noise = best_settings
    val_risk, test_risk = predict_frame(bundle, validation), predict_frame(bundle, test)
    edges = np.linspace(0, 1, 6)
    val_errors = validation.horizontal_error_m.to_numpy()
    global_variance = float(np.mean(val_errors ** 2) / 2)
    variances = []
    for low, high in zip(edges[:-1], edges[1:]):
        members = (val_risk >= low) & (val_risk <= high if high == 1 else val_risk < high)
        estimate = np.mean(val_errors[members] ** 2) / 2 if members.sum() >= 10 else global_variance
        variances.append(float(np.clip(estimate, 1, 2500)))
    variances = np.maximum.accumulate(variances)
    bins = np.clip(np.searchsorted(edges, test_risk, side='right') - 1, 0, len(variances) - 1)
    fixed = _apply(test, np.full(len(test), sigma ** 2), acceleration_noise)
    adaptive = _apply(test, variances[bins], acceleration_noise)
    truth = test[['gt_e_m', 'gt_n_m']].to_numpy()
    test[['fixed_filter_e_m', 'fixed_filter_n_m']] = fixed
    test[['adaptive_filter_e_m', 'adaptive_filter_n_m']] = adaptive
    test['fixed_filter_error_m'] = np.linalg.norm(fixed - truth, axis=1)
    test['adaptive_filter_error_m'] = np.linalg.norm(adaptive - truth, axis=1)
    results = {name: position_metrics(test[column].to_numpy()) for name, column in [
        ('wls', 'horizontal_error_m'), ('fixed_filter', 'fixed_filter_error_m'), ('adaptive_filter', 'adaptive_filter_error_m')]}
    for name in ['equal', 'robust']:
        if f'{name}_error_m' in test:
            results[name] = position_metrics(test[f'{name}_error_m'].to_numpy())
    return {'methods': results, 'settings_selected_on': 'validation', 'fixed_sigma_m': sigma,
            'acceleration_noise_mps2': acceleration_noise, 'risk_bin_edges': edges.tolist(),
            'risk_bin_variances_m2': variances.tolist(),
            'uses_imu': {'imu_accel_e_mps2', 'imu_accel_n_mps2'} <= set(test),
            'warning': 'Empirical planar-filter baseline. Report failures as well as improvements; this is not a production INS.'}
