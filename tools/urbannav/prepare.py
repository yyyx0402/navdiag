# SPDX-License-Identifier: GPL-3.0-or-later
"""Standalone UrbanNav adapter using the pinned external GPL KLT extractor.

The five model features use receiver radio/SPP data. Independent SPAN truth
and building masks are used only for reference labels and audit metadata.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyrtklib as prl

from navdiag.nlos import cn0_variability, validate_observations
from navdiag.visibility import SkyMaskGrid, geometry_at_truth, match_truth, read_urbannav_truth

ROOT = Path(__file__).resolve().parents[2]
EXTRACTOR_SHA = '7f9368986fbc9f96a0e7004194135e3f94dde851721ecc27c533859c4d905472'
DAYS = {'TST': '137', 'WP': '141', 'MK': '138'}
POLICY = {'margin_deg': 5., 'max_grid_distance_m': 2., 'neighbor_radius_m': 3.,
          'max_height_difference_m': 3., 'max_quality': 2}


def extractor_module():
    path = ROOT / '.cache/klt_reference/extract_features.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != EXTRACTOR_SHA:
        raise ValueError('External feature extractor differs from the pinned GPL source')
    spec = importlib.util.spec_from_file_location('external_klt_extractor', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare(raw_root: Path, output: Path) -> dict:
    extractor = extractor_module()
    frames, audits = [], {}
    source_files = []
    for area, day in DAYS.items():
        directory = raw_root / area
        rover = list(directory.glob('*.ublox.f9p.splitter.obs'))
        if len(rover) != 1:
            raise ValueError(f'{area} requires exactly one splitter receiver observation file')
        navfiles = [raw_root / 'nav' / f'brdc{day}0.21n',
                    raw_root / 'nav' / f'HKSC00HKG_R_2021{day}0000_01D_CN.rnx']
        truth_path, mask_path = directory / 'groundtruth.txt', directory / 'sky_mask.csv'
        for path in [rover[0], *navfiles, truth_path, mask_path]:
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            source_files.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': digest,
                                 'bytes': path.stat().st_size})
        print(area, 'reading raw observations and static building grid', flush=True)
        obs, nav = extractor.read_rinex(rover[0], navfiles)
        if obs.n == 0 or nav.n == 0:
            raise ValueError('Empty RINEX observations or ephemeris')
        truth = read_urbannav_truth(truth_path)
        spans = extractor.epoch_slices(obs)
        times = np.array([extractor.epoch_utc(obs.data[i].time) for i, _ in spans])
        truth_ids, time_valid = match_truth(times, truth)
        mask = SkyMaskGrid.read(mask_path)
        table = pd.DataFrame(extractor.raw_epoch_rows(obs, nav, extractor.make_prcopt()))
        label_rows = []
        for epoch, (i, m) in enumerate(spans):
            o = extractor.rover_epoch_obs(obs, i, m)
            rs, dts, variance, health = (prl.Arr1Ddouble(m * 6), prl.Arr1Ddouble(m * 2),
                                        prl.Arr1Ddouble(m), prl.Arr1Dint(m))
            prl.satposs(o.data[0].time, o.data[0], o.n, nav, prl.EPHOPT_BRDC, rs, dts, variance, health)
            truth_row = truth.iloc[truth_ids[epoch]]
            positions = np.array([[rs[k * 6 + j] for j in range(3)] for k in range(o.n)])
            azimuth, elevation = geometry_at_truth(positions, truth_row.latitude_deg,
                                                   truth_row.longitude_deg, truth_row.height_m)
            for k in range(o.n):
                name = extractor.sat_name(o.data[k].sat)
                if name[0] not in {'G', 'C'}:
                    continue
                valid_orbit = np.linalg.norm(positions[k]) > 1e7 and health[k] == 0
                reference = {'signal_label': 'unknown', 'reference_reason': 'missing_ephemeris',
                             'mask_distance_m': None, 'mask_min_deg': None, 'mask_max_deg': None}
                if not time_valid[epoch]:
                    reference['reference_reason'] = 'truth_time_unmatched'
                elif valid_orbit:
                    reference = mask.reference(truth_row.latitude_deg, truth_row.longitude_deg,
                                               truth_row.height_m, truth_row.quality,
                                               azimuth[k], elevation[k], **POLICY)
                label_rows.append({'epoch': epoch, 'sat': name, **reference,
                                   'reference_azimuth_deg': float(azimuth[k]) if valid_orbit else None,
                                   'reference_elevation_deg': float(elevation[k]) if valid_orbit else None,
                                   'reference_truth_quality': int(truth_row.quality) if time_valid[epoch] else None})
        table = extractor.add_derived_features(table)
        frame = table.drop(columns=['nlos']).merge(pd.DataFrame(label_rows), on=['epoch', 'sat'],
                                                 how='left', validate='one_to_one')
        if frame.signal_label.isna().any():
            raise ValueError('A radio feature row has no independently computed visibility reference')
        frame = frame.rename(columns={'sat': 'satellite_id', 'sys': 'system', 'cn0': 'cn0_dbhz',
                                      'elev_deg': 'elevation_deg', 'pr_resid_norm': 'residual_normalized',
                                      'pr_rate_consistency': 'pr_rate_consistency_mps', 'cn0_std': 'cn0_std_dbhz'})
        frame['trip_id'] = f'UrbanNav_{area}_202105{int(day) - 120:02d}'
        frame['device_id'] = 'ublox_f9p_splitter'
        frame['scene_id'], frame['data_kind'] = area, 'real'
        frame['signal_label_source'] = 'static_3d_sky_mask'
        frame['obstruction_type'] = np.where(frame.signal_label == 'nlos', 'static_building_reference', 'unknown')
        frame['timestamp_ms'] = np.rint(frame.utc * 1000).astype('int64')
        frame['cn0_std_dbhz'] = cn0_variability(frame)
        frame = validate_observations(frame)
        audits[area] = {**mask.audit, 'observation_epochs': len(times), 'truth_epochs': len(truth),
                        'matched_truth_epochs': int(time_valid.sum()),
                        'time_delta_max_s': float(abs(times[time_valid] - truth.utc.to_numpy()[truth_ids[time_valid]]).max()),
                        'rows': len(frame), 'epochs': frame.timestamp_ms.nunique(),
                        'label_counts': frame.signal_label.value_counts().to_dict(),
                        'reference_reasons': frame.reference_reason.value_counts().to_dict(),
                        'missing_feature_fraction': {key: float(frame[key].isna().mean()) for key in
                            ['cn0_dbhz', 'elevation_deg', 'residual_normalized', 'pr_rate_consistency_mps', 'cn0_std_dbhz']}}
        frames.append(frame)
        print(area, audits[area]['label_counts'], flush=True)
        del mask, table, label_rows, obs, nav
        gc.collect()
    frame = pd.concat(frames, ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    csv = output / 'satellites.csv'
    frame.to_csv(csv, index=False)
    provenance = {'source_url': 'https://github.com/IPNL-POLYU/UrbanNavDataset',
                  'source_commit': 'Public 2021 release; downloaded file SHA-256 pins define the inputs',
                  'source_sha256': hashlib.sha256(json.dumps(source_files, sort_keys=True).encode()).hexdigest(),
                  'prepared_sha256': hashlib.sha256(csv.read_bytes()).hexdigest(),
                  'original_dataset': 'https://github.com/IPNL-POLYU/UrbanNavDataset',
                  'license': 'UrbanNav README has no explicit data reuse license; SatRef personal noncommercial internal use; raw/derived tables not redistributed',
                  'features_origin': 'Pinned GPL KLT RTKLIB SPP extractor, same GPS/BeiDou five-feature definitions; independent SPAN positions and building masks used only for reference labels',
                  'label_warning': 'Static-building map-derived references, not independently verified satellite reception truth. Dynamic obstacles and LOS multipath are not labelled; near-boundary/poor-truth/uncovered rows are unknown.',
                  'reference_policy': POLICY, 'source_files': source_files, 'areas': audits,
                  'rows': len(frame), 'epochs': len(frame.drop_duplicates(['trip_id', 'timestamp_ms'])),
                  'label_sources': ['static_3d_sky_mask'],
                  'limitations': ['One drive per district; three target districts in one city, no cross-city conclusion.',
                                  'The map has a fixed 5m reference height; height/lever-arm uncertainty remains despite guards.',
                                  'Mask 0/360 closure discrepancies are preserved in an uncertainty envelope.',
                                  'Boundary and spatial margins are engineering guards, not calibrated label error bounds.',
                                  'F9P splitter antenna and map labels differ from KLT camera-labelled captures.',
                                  'Quality and boundary exclusions change the evaluation population; coverage must accompany scores.']}
    (output / 'preprocess_report.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding='utf-8')
    return provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT / 'data/raw/urbannav')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/processed/urbannav')
    args = parser.parse_args()
    prepare(args.raw, args.output)
