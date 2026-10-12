"""Observation-only causal position/velocity baseline inspired by GNSS Doppler methods."""
import numpy as np


def filter_doppler_positions(times_ms, positions_en, velocities_en, velocity_covariances,
                             position_variance_m2, acceleration_noise=2., reset_gap_s=10.,
                             position_gate_nis=100., velocity_gate_nis=25., covariance_floor=0.04):
    times = np.asarray(times_ms, float)
    positions, velocities, covariances = map(lambda x: np.asarray(x, float),
                                             [positions_en, velocities_en, velocity_covariances])
    if positions.shape != (len(times), 2) or velocities.shape != positions.shape or covariances.shape != (len(times), 2, 2):
        raise ValueError('Doppler inputs must align with position epochs')
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or not np.isfinite(positions).all():
        raise ValueError('Doppler filter requires finite fixes and strictly increasing times')
    if not all(np.isfinite(v) and v > 0 for v in [position_variance_m2, acceleration_noise, reset_gap_s,
                                                 position_gate_nis, velocity_gate_nis, covariance_floor]):
        raise ValueError('Doppler noise, gates and reset settings must be positive')
    hp = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
    hv = np.array([[0, 0, 1, 0], [0, 0, 0, 1]], float)
    result = np.empty_like(positions)
    diagnostics = {'epochs': len(times), 'segments': 0, 'velocity_updates': 0,
                   'missing_velocity_epochs': 0, 'invalid_covariance_epochs': 0,
                   'position_gate_rejections': 0, 'velocity_gate_rejections': 0}
    state, covariance = None, None

    def update(h, measurement, noise, gate):
        nonlocal state, covariance
        innovation = measurement - h @ state
        s = h @ covariance @ h.T + noise
        nis = float(innovation @ np.linalg.solve(s, innovation))
        if nis > gate:
            return False
        gain = np.linalg.solve(s, h @ covariance).T
        state = state + gain @ innovation
        stable = np.eye(4) - gain @ h
        covariance = stable @ covariance @ stable.T + gain @ noise @ gain.T
        return True

    for i, fix in enumerate(positions):
        dt = (times[i] - times[i - 1]) / 1000 if i else 0
        if state is None or dt > reset_gap_s:
            state = np.array([*fix, 0., 0.])
            covariance = np.diag([position_variance_m2] * 2 + [100.] * 2)
            diagnostics['segments'] += 1
        else:
            f = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
            g = np.array([[dt**2 / 2, 0], [0, dt**2 / 2], [dt, 0], [0, dt]])
            state = f @ state
            covariance = f @ covariance @ f.T + acceleration_noise**2 * g @ g.T
            if not update(hp, fix, np.eye(2) * position_variance_m2, position_gate_nis):
                diagnostics['position_gate_rejections'] += 1
        cv = covariances[i]
        if not np.isfinite(velocities[i]).all() or not np.isfinite(cv).all():
            diagnostics['missing_velocity_epochs'] += 1
        elif not np.allclose(cv, cv.T) or np.linalg.eigvalsh(cv).min() <= 0:
            diagnostics['invalid_covariance_epochs'] += 1
        else:
            values, vectors = np.linalg.eigh(cv)
            noise = (vectors * np.maximum(values, covariance_floor)) @ vectors.T
            if update(hv, velocities[i], noise, velocity_gate_nis):
                diagnostics['velocity_updates'] += 1
            else:
                diagnostics['velocity_gate_rejections'] += 1
        result[i] = state[:2]
    return result, diagnostics
