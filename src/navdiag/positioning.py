"""Coordinate transforms and GNSS least squares, with no reference-position inputs."""
from dataclasses import dataclass

import numpy as np

WGS84_A = 6378137.0
WGS84_E2 = 6.6943799901413165e-3
C = 299792458.0
OMEGA_E = 7.2921151467e-5


def lla_to_ecef(lla: np.ndarray) -> np.ndarray:
    """Latitude/longitude in degrees, ellipsoidal height in metres -> ECEF metres."""
    lla = np.asarray(lla, dtype=float)
    lat, lon, height = np.moveaxis(lla, -1, 0)
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    radius = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    return np.stack([(radius + height) * np.cos(lat) * np.cos(lon),
                     (radius + height) * np.cos(lat) * np.sin(lon),
                     (radius * (1 - WGS84_E2) + height) * np.sin(lat)], axis=-1)


def ecef_to_lla(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=float)
    x, y, z = np.moveaxis(xyz, -1, 0)
    p = np.hypot(x, y)
    if np.any(np.linalg.norm(xyz, axis=-1) < 1e6):
        raise ValueError("ECEF point is too close to Earth's centre.")
    lat = np.arctan2(z, p * (1 - WGS84_E2))
    for _ in range(10):
        radius = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
        lat = np.arctan2(z + WGS84_E2 * radius * np.sin(lat), p)
    radius = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    # Projection onto the ellipsoid normal also handles positions near the poles.
    height = p * np.cos(lat) + z * np.sin(lat) - radius * (1 - WGS84_E2 * np.sin(lat) ** 2)
    return np.stack([np.rad2deg(lat), np.rad2deg(np.arctan2(y, x)), height], axis=-1)


def enu_rotation(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat, lon = np.deg2rad([lat_deg, lon_deg])
    return np.array([[-np.sin(lon), np.cos(lon), 0],
                     [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
                     [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]])


def ecef_to_enu(xyz: np.ndarray, origin_lla: np.ndarray) -> np.ndarray:
    return (np.asarray(xyz) - lla_to_ecef(origin_lla)) @ enu_rotation(*origin_lla[:2]).T


def enu_to_ecef(enu: np.ndarray, origin_lla: np.ndarray) -> np.ndarray:
    return np.asarray(enu) @ enu_rotation(*origin_lla[:2]) + lla_to_ecef(origin_lla)


def earth_rotation_correction(satellites: np.ndarray, receiver: np.ndarray) -> np.ndarray:
    """Rotate transmit-time ECEF satellite positions to the receive-time frame."""
    travel_time = np.linalg.norm(satellites - receiver, axis=1) / C
    angle = OMEGA_E * travel_time
    x, y, z = satellites.T
    return np.column_stack([np.cos(angle) * x + np.sin(angle) * y,
                            -np.sin(angle) * x + np.cos(angle) * y, z])


@dataclass
class PositionSolution:
    xyz: np.ndarray
    clocks_m: dict[str, float]
    residuals_m: np.ndarray
    hdop: float
    iterations: int


def solve_wls(satellites: np.ndarray, pseudoranges: np.ndarray, systems: np.ndarray | None = None,
              sigma_m: np.ndarray | None = None, initial_xyz: np.ndarray | None = None,
              robust: bool = False, rotate_earth: bool = True) -> PositionSolution:
    """Estimate receiver ECEF and one clock bias per constellation.

    Pseudoranges must already include satellite clock, inter-signal bias and
    atmospheric corrections. A reference/ground-truth location is never required.
    """
    satellites, pseudoranges = np.asarray(satellites, float), np.asarray(pseudoranges, float)
    if satellites.shape != (len(pseudoranges), 3) or not np.isfinite(satellites).all() or not np.isfinite(pseudoranges).all():
        raise ValueError("Satellite/pseudorange arrays must be finite and aligned.")
    systems = np.full(len(pseudoranges), "gps") if systems is None else np.asarray(systems).astype(str)
    if len(systems) != len(pseudoranges):
        raise ValueError("Constellation IDs are not aligned with observations.")
    names = sorted(set(systems))
    clocks = np.column_stack([systems == name for name in names]).astype(float)
    dimensions = 3 + len(names)
    if len(pseudoranges) < dimensions:
        raise ValueError("Not enough satellites for position and constellation clocks.")
    sigmas = np.ones(len(pseudoranges)) if sigma_m is None else np.asarray(sigma_m, float)
    if sigmas.shape != pseudoranges.shape or not np.isfinite(sigmas).all() or np.any(sigmas <= 0):
        raise ValueError("Observation sigma values must be positive and finite.")
    state = np.zeros(dimensions)
    if initial_xyz is not None:
        state[:3] = np.asarray(initial_xyz, float)
    converged = False
    for iteration in range(30):
        positions = earth_rotation_correction(satellites, state[:3]) if rotate_earth else satellites
        delta = state[:3] - positions
        distance = np.linalg.norm(delta, axis=1)
        design = np.column_stack([delta / distance[:, None], clocks])
        residuals = pseudoranges - distance - clocks @ state[3:]
        weights = 1 / np.clip(sigmas, 0.5, 100) ** 2
        # Start with ordinary WLS; robust weights are meaningful only near a solution.
        if robust and iteration >= 4:
            scale = max(2.0, 1.4826 * np.median(np.abs(residuals - np.median(residuals))))
            weights *= np.minimum(1, 1.5 * scale / np.maximum(np.abs(residuals), 1e-6))
        weighted = design * np.sqrt(weights[:, None])
        step, _, rank, _ = np.linalg.lstsq(weighted, residuals * np.sqrt(weights), rcond=None)
        if rank < dimensions:
            raise ValueError("Satellite geometry is rank deficient.")
        state += step
        if np.linalg.norm(step) < 1e-3 and (not robust or iteration >= 5):
            converged = True
            break
    if not converged or not 6e6 < np.linalg.norm(state[:3]) < 7e6:
        raise ValueError("WLS did not converge to a plausible Earth-surface receiver position.")
    positions = earth_rotation_correction(satellites, state[:3]) if rotate_earth else satellites
    delta = state[:3] - positions
    distance = np.linalg.norm(delta, axis=1)
    design = np.column_stack([delta / distance[:, None], clocks])
    residuals = pseudoranges - distance - clocks @ state[3:]
    covariance = np.linalg.inv(design.T @ design)[:3, :3]
    rotation = enu_rotation(*ecef_to_lla(state[:3])[:2])
    enu_covariance = rotation @ covariance @ rotation.T
    hdop = float(np.sqrt(max(0, enu_covariance[0, 0] + enu_covariance[1, 1])))
    return PositionSolution(state[:3], dict(zip(names, map(float, state[3:]))), residuals, hdop, iteration + 1)


def align_reference(times_ms: np.ndarray, reference_times_ms: np.ndarray, reference_xyz: np.ndarray,
                    max_gap_ms: float = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate ECEF truth only within support; never extrapolate across gaps."""
    times_ms, reference_times_ms = np.asarray(times_ms, float), np.asarray(reference_times_ms, float)
    reference_xyz = np.asarray(reference_xyz, float)
    order = np.argsort(reference_times_ms)
    ref_times, ref_xyz = reference_times_ms[order], reference_xyz[order]
    unique = np.r_[True, np.diff(ref_times) > 0]
    ref_times, ref_xyz = ref_times[unique], ref_xyz[unique]
    if len(ref_times) < 2 or not np.isfinite(ref_xyz).all():
        raise ValueError("Need at least two finite reference fixes.")
    right = np.searchsorted(ref_times, times_ms, side="left")
    right_clipped = np.clip(right, 1, len(ref_times) - 1)
    left = right_clipped - 1
    exact = np.isin(times_ms, ref_times)
    valid = ((times_ms >= ref_times[0]) & (times_ms <= ref_times[-1])
             & ((ref_times[right_clipped] - ref_times[left] <= max_gap_ms) | exact))
    aligned = np.column_stack([np.interp(times_ms, ref_times, ref_xyz[:, axis]) for axis in range(3)])
    aligned[~valid] = np.nan
    return aligned, valid
