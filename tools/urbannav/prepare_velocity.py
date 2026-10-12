# SPDX-License-Identifier: GPL-3.0-or-later
"""Extract RTKLIB Doppler velocity for the already fixed, observation-only SPP population."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyrtklib as prl

from prepare import DAYS, ROOT, extractor_module
from navdiag.positioning import enu_rotation


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(data, output):
    output.mkdir(parents=True, exist_ok=True)
    plan_path = ROOT / 'config/urbannav_positioning.json'
    policy = json.loads(plan_path.read_text(encoding='utf-8'))
    base_path = data / 'observations.csv'
    base = pd.read_csv(base_path)
    plan = {'base_observations_sha256': sha(base_path), 'base_preprocessing_plan_sha256': sha(data / 'preprocessing_plan.json'),
            'policy_sha256': sha(plan_path), 'extractor_sha256': sha(ROOT / '.cache/klt_reference/extract_features.py'),
            'implementation_sha256': sha(Path(__file__)), 'package_version': 'pyrtklib 0.2.7',
            'velocity_policy': 'RTKLIB qv must be finite positive definite, Doppler velocity finite and speed <=70 m/s. No reference used.'}
    frozen = output / 'preprocessing_plan.json'
    if frozen.exists() and json.loads(frozen.read_text(encoding='utf-8')) != plan:
        raise ValueError('Doppler extraction inputs changed; choose a new output directory')
    frozen.write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    extractor = extractor_module()
    frames, audit = [], []
    for area, day in DAYS.items():
        trip = f'UrbanNav_{area}_202105{int(day)-120:02d}'
        part = base[base.trip_id == trip]
        lookup = part.set_index('timestamp_ms')
        directory = ROOT / 'data/raw/urbannav' / area
        obs, nav = extractor.read_rinex(next(directory.glob('*.ublox.f9p.splitter.obs')),
            [ROOT / 'data/raw/urbannav/nav' / f'brdc{day}0.21n',
             ROOT / 'data/raw/urbannav/nav' / f'HKSC00HKG_R_2021{day}0000_01D_CN.rnx'])
        options = extractor.make_prcopt()
        options.elmin = np.radians(policy['elevation_mask_deg'])
        rotation = enu_rotation(part.origin_lat_deg.iloc[0], part.origin_lon_deg.iloc[0])
        rows = []
        for i, m in extractor.epoch_slices(obs):
            timestamp = int(round(extractor.epoch_utc(obs.data[i].time) * 1000))
            if timestamp not in lookup.index:
                continue
            epoch = extractor.rover_epoch_obs(obs, i, m)
            sol = prl.sol_t()
            azel = prl.Arr1Ddouble(epoch.n * 2)
            ssat = prl.Arr1Dssat_t(prl.MAXSAT)
            ok = prl.pntpos(epoch.data.ptr, epoch.n, nav, options, sol, azel, ssat.ptr, prl.Arr1Dchar(128))
            xyz = np.array([sol.rr[k] for k in range(3)])
            if not ok or not np.allclose(xyz, lookup.loc[timestamp, ['wls_x_m', 'wls_y_m', 'wls_z_m']].to_numpy(float), rtol=0, atol=1e-6):
                raise ValueError('Recomputed SPP differs from frozen baseline')
            velocity = np.array([sol.rr[k] for k in range(3, 6)])
            q = [sol.qv[k] for k in range(6)]
            covariance = np.array([[q[0], q[3], q[5]], [q[3], q[1], q[4]], [q[5], q[4], q[2]]])
            valid = bool(np.isfinite(velocity).all() and np.isfinite(covariance).all()
                         and np.linalg.eigvalsh(covariance).min() > 0 and np.linalg.norm(velocity) <= 70)
            enu, cov_enu = rotation @ velocity, rotation @ covariance @ rotation.T
            rows.append({'trip_id': trip, 'device_id': 'ublox_f9p_splitter', 'timestamp_ms': timestamp,
                         'doppler_valid': valid, 'doppler_e_mps': float(enu[0]) if valid else np.nan,
                         'doppler_n_mps': float(enu[1]) if valid else np.nan,
                         'doppler_ee_m2ps2': float(cov_enu[0, 0]) if valid else np.nan,
                         'doppler_nn_m2ps2': float(cov_enu[1, 1]) if valid else np.nan,
                         'doppler_en_m2ps2': float(cov_enu[0, 1]) if valid else np.nan})
        frame = pd.DataFrame(rows)
        if len(frame) != len(part):
            raise ValueError('Velocity extraction must preserve every baseline observation epoch')
        frames.append(frame)
        record = {'trip_id': trip, 'spp_epochs': len(frame), 'valid_doppler_epochs': int(frame.doppler_valid.sum())}
        audit.append(record)
        print(json.dumps(record), flush=True)
    csv = output / 'velocity.csv'
    pd.concat(frames, ignore_index=True).to_csv(csv, index=False)
    (output / 'report.json').write_text(json.dumps({'plan': plan, 'coverage': audit, 'velocity_sha256': sha(csv)}, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/processed/urbannav_positioning_v3')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/processed/urbannav_doppler')
    args = parser.parse_args()
    prepare(args.data, args.output)
