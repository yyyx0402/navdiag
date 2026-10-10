"""Synthetic code observations + actual WLS, for an offline end-to-end demo."""
from pathlib import Path

import numpy as np
import pandas as pd

from navdiag.positioning import (
    earth_rotation_correction, ecef_to_enu, enu_rotation, enu_to_ecef, lla_to_ecef, solve_wls,
)


def generate_navigation_demo(path: Path, seed: int = 42, trips: int = 15, epochs: int = 120) -> Path:
    if trips < 3 or epochs < 20:
        raise ValueError('Navigation demo needs at least 3 trips and 20 epochs')
    rng = np.random.default_rng(seed)
    rows = []
    for trip in range(trips):
        origin = np.array([37 + trip * 0.001, -122, 30])
        rotation = enu_rotation(*origin[:2])
        receiver_origin = lla_to_ecef(origin)
        count = 12
        azimuth = np.linspace(0, 2 * np.pi, count, endpoint=False) + rng.uniform(0, 0.5)
        elevation = rng.uniform(0.25, 1.3, count)
        lines_enu = np.column_stack([np.sin(azimuth) * np.cos(elevation),
                                    np.cos(azimuth) * np.cos(elevation), np.sin(elevation)])
        satellites = receiver_origin + lines_enu @ rotation * 2.2e7
        severity, previous_position = 0.5, None
        for epoch in range(epochs):
            if epoch % 25 == 0:
                severity = rng.uniform(0, 1)
            true_enu = np.array([3 * epoch, 30 * np.sin(epoch / 20), 0.0])
            true_xyz = enu_to_ecef(true_enu, origin)
            cn0 = np.clip(44 - 13 * severity + rng.normal(0, 2, count), 15, 50)
            uncertainty = np.clip(1.5 + 6 * severity + (40 - cn0) / 8, 0.5, 20)
            measurement_noise = rng.normal(0, uncertainty)
            # Per-satellite positive biases emulate the effect of degraded code observations.
            corrupt = rng.choice(count, 3, replace=False)
            measurement_noise[corrupt] += severity ** 2 * rng.uniform(15, 65, 3)
            ranges = np.linalg.norm(earth_rotation_correction(satellites, true_xyz) - true_xyz, axis=1)
            pseudoranges = ranges + 50000 + measurement_noise
            equal = solve_wls(satellites, pseudoranges, initial_xyz=previous_position)
            wls = solve_wls(satellites, pseudoranges, sigma_m=uncertainty, initial_xyz=equal.xyz)
            try:
                robust = solve_wls(satellites, pseudoranges, sigma_m=uncertainty, initial_xyz=wls.xyz, robust=True)
            except ValueError:
                robust = wls
            row = {
                'trip_id': f'nav_demo_{trip:02d}', 'device_id': 'synthetic_receiver',
                'timestamp_ms': 1700000000000 + trip * 1000000 + epoch * 1000, 'data_kind': 'synthetic',
                'satellite_count': count, 'cn0_mean_dbhz': float(cn0.mean()), 'cn0_min_dbhz': float(cn0.min()),
                'hdop': wls.hdop, 'residual_rms_m': float(np.sqrt(np.mean(wls.residuals_m ** 2))),
                'speed_mps': 0 if previous_position is None else min(70, float(np.linalg.norm(wls.xyz - previous_position))),
                'origin_lat_deg': origin[0], 'origin_lon_deg': origin[1], 'origin_alt_m': origin[2],
                'gt_e_m': true_enu[0], 'gt_n_m': true_enu[1], 'gt_u_m': true_enu[2],
                'imu_accel_e_mps2': float(rng.normal(0, 0.03)),
                'imu_accel_n_mps2': float(-30 / 400 * np.sin(epoch / 20) + rng.normal(0, 0.03)),
            }
            for name, solution in [('equal', equal), ('wls', wls), ('robust', robust)]:
                enu = ecef_to_enu(solution.xyz, origin)
                row.update({f'{name}_{axis}_m': float(value) for axis, value in zip('xyz', solution.xyz)})
                row.update({f'{name}_{axis}_m': float(value) for axis, value in zip(['e', 'n', 'u'], enu)})
                row[f'{name}_error_m'] = float(np.linalg.norm(enu[:2] - true_enu[:2]))
            row['horizontal_error_m'] = row['wls_error_m']
            rows.append(row)
            previous_position = wls.xyz
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
