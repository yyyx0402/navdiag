"""Independent static-building visibility references, never classifier inputs.

UrbanNav CSV masks contain latitude, longitude, height and 361 boundary
elevations indexed by GNSS azimuth 0..360 degrees (north, clockwise).
These are map-derived reference labels, not manually verified reception truth.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def read_urbannav_truth(path: Path) -> pd.DataFrame:
    rows = [line.split() for line in path.read_text().splitlines()
            if line and line[0].isdigit()]
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2 or values.shape[1] != 20 or not np.isfinite(values).all():
        raise ValueError('UrbanNav truth must have 20 finite numeric fields')
    # The three May 2021 recordings have an 18-second GPS/UTC offset.
    utc = values[:, 1] * 604800 + values[:, 2] + 315964800 - 18
    if np.max(np.abs(utc - values[:, 0])) > .02 or np.any(np.diff(utc) <= 0):
        raise ValueError('Truth UTC and GPS week/seconds disagree, or times repeat')
    if not np.all((utc >= 1609459200) & (utc < 1640995200)):
        raise ValueError('This adapter is restricted to the 2021 UrbanNav release')
    lat = np.sign(values[:, 3]) * (np.abs(values[:, 3]) + values[:, 4] / 60 + values[:, 5] / 3600)
    lon = np.sign(values[:, 6]) * (np.abs(values[:, 6]) + values[:, 7] / 60 + values[:, 8] / 3600)
    if np.any(np.abs(lat) > 90) or np.any(np.abs(lon) > 180):
        raise ValueError('Invalid DMS truth coordinates')
    return pd.DataFrame({'utc': utc, 'latitude_deg': lat, 'longitude_deg': lon,
                         'height_m': values[:, 9], 'quality': values[:, 19]})


def match_truth(times: np.ndarray, truth: pd.DataFrame, tolerance_s: float = .05) -> tuple[np.ndarray, np.ndarray]:
    """Nearest timestamps without extrapolation or silent row-index alignment."""
    ref = truth.utc.to_numpy()
    right = np.clip(np.searchsorted(ref, times), 0, len(ref) - 1)
    left = np.maximum(right - 1, 0)
    ids = np.where(abs(ref[left] - times) <= abs(ref[right] - times), left, right)
    distance = abs(ref[ids] - times)
    valid = distance <= tolerance_s
    if len(np.unique(ids[valid])) != int(valid.sum()):
        raise ValueError('Multiple observation epochs match one truth epoch')
    return ids, valid


def local_xy(lat: np.ndarray, lon: np.ndarray, origin: tuple[float, float]) -> np.ndarray:
    """Local WGS84 grid distances; kilometre-scale areas only."""
    phi = np.radians(origin[0])
    a, e2 = 6378137., 6.69437999014e-3
    n = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    m = a * (1 - e2) / (1 - e2 * np.sin(phi) ** 2) ** 1.5
    return np.column_stack([np.radians(lon - origin[1]) * n * np.cos(phi),
                            np.radians(lat - origin[0]) * m])


def geometry_at_truth(satellite_ecef: np.ndarray, latitude: float, longitude: float,
                      height: float) -> tuple[np.ndarray, np.ndarray]:
    """Satellite azimuth/elevation at independent truth, for reference labels only."""
    phi, lam = np.radians([latitude, longitude])
    a, e2 = 6378137., 6.69437999014e-3
    n = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    receiver = np.array([(n + height) * np.cos(phi) * np.cos(lam),
                         (n + height) * np.cos(phi) * np.sin(lam),
                         (n * (1 - e2) + height) * np.sin(phi)])
    delta = np.asarray(satellite_ecef) - receiver
    rotation = np.array([[-np.sin(lam), np.cos(lam), 0],
                         [-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)],
                         [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)]])
    east, north, up = (delta @ rotation.T).T
    azimuth = np.degrees(np.arctan2(east, north)) % 360
    elevation = np.degrees(np.arctan2(up, np.hypot(east, north)))
    return azimuth, elevation


class SkyMaskGrid:
    def __init__(self, coordinates: np.ndarray, profiles: np.ndarray):
        self.coordinates = np.asarray(coordinates, dtype=float)
        self.profiles = np.asarray(profiles, dtype=np.float32)
        if (self.coordinates.ndim != 2 or self.coordinates.shape[1] != 3
                or self.profiles.shape != (len(self.coordinates), 361)
                or not np.isfinite(self.coordinates).all()
                or not np.isfinite(self.profiles).all()):
            raise ValueError('Mask requires three coordinates and 361 finite angles')
        if np.any((self.profiles < 0) | (self.profiles > 90)):
            raise ValueError('Mask contains invalid elevations or inside-building sentinels')
        self.origin = tuple(self.coordinates[:, :2].mean(axis=0))
        self.xy = local_xy(self.coordinates[:, 0], self.coordinates[:, 1], self.origin)
        self.tree = cKDTree(self.xy)
        self.audit = {'grid_points': len(self.xy), 'origin': list(self.origin),
                      'height_range_m': [float(self.coordinates[:, 2].min()), float(self.coordinates[:, 2].max())],
                      'closure_mismatch_rows': int((abs(self.profiles[:, 0] - self.profiles[:, 360]) > .1).sum()),
                      'closure_max_difference_deg': float(abs(self.profiles[:, 0] - self.profiles[:, 360]).max())}

    @classmethod
    def read(cls, path: Path):
        coordinates, profiles, excluded = [], [], 0
        dtypes = {i: ('float64' if i < 3 else 'float32') for i in range(364)}
        for part in pd.read_csv(path, header=None, skiprows=1, dtype=dtypes, chunksize=20000):
            if part.shape[1] != 364:
                raise ValueError('UrbanNav mask must have 364 comma-separated entries')
            xyz = part.iloc[:, :3].to_numpy()
            angles = part.iloc[:, 3:].to_numpy(dtype=np.float32)
            valid = np.isfinite(xyz).all(axis=1) & np.isfinite(angles).all(axis=1)
            valid &= ((angles >= 0) & (angles <= 90)).all(axis=1)
            excluded += int((~valid).sum())
            coordinates.append(xyz[valid])
            profiles.append(angles[valid])
        grid = cls(np.concatenate(coordinates), np.concatenate(profiles))
        grid.audit['excluded_invalid_or_inside_building_grid_rows'] = excluded
        return grid

    def reference(self, lat: float, lon: float, height: float, quality: float,
                  azimuth: float, elevation: float, *, margin_deg: float = 5.,
                  max_grid_distance_m: float = 2., neighbor_radius_m: float = 3.,
                  max_height_difference_m: float = 3., max_quality: int = 2) -> dict:
        result = {'signal_label': 'unknown', 'reference_reason': 'invalid_geometry',
                  'mask_distance_m': None, 'mask_min_deg': None, 'mask_max_deg': None}
        if not np.isfinite([lat, lon, height, quality, azimuth, elevation]).all() or elevation <= 0:
            return result
        if quality < 1 or quality > max_quality:
            return {**result, 'reference_reason': 'truth_quality'}
        distance, ids = self.tree.query(local_xy(np.array([lat]), np.array([lon]), self.origin)[0],
                                        k=min(4, len(self.xy)))
        distance, ids = np.atleast_1d(distance), np.atleast_1d(ids)
        result['mask_distance_m'] = float(distance[0])
        if distance[0] > max_grid_distance_m:
            return {**result, 'reference_reason': 'outside_mask_coverage'}
        ids = ids[distance <= neighbor_radius_m]
        if np.max(abs(self.coordinates[ids, 2] - height)) > max_height_difference_m:
            return {**result, 'reference_reason': 'height_mismatch'}
        angle = int(np.floor(azimuth % 360))
        slots = {(angle + offset) % 360 for offset in [-1, 0, 1, 2]}
        # Some public files disagree at 0/360 despite the documented closure.
        # Keep both values in the uncertainty envelope, never silently overwrite.
        if 0 in slots:
            slots.add(360)
        values = self.profiles[np.ix_(ids, sorted(slots))]
        low, high = float(values.min()), float(values.max())
        result.update(mask_min_deg=low, mask_max_deg=high)
        label = 'nlos' if elevation < low - margin_deg else 'los' if elevation > high + margin_deg else 'unknown'
        return {**result, 'signal_label': label,
                'reference_reason': 'clear_building_reference' if label != 'unknown' else 'mask_boundary'}
