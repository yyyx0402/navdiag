# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional standalone adapter around the external GPL KLT feature extractor.

No upstream implementation is included in the navdiag Python package. Run after
scripts/fetch_klt_extended.py. Raw images are discarded from compact labels;
models never receive images, ground-truth positions or collection IDs as inputs.
"""
import gc
import hashlib
import importlib.util
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import pyrtklib as prl

from navdiag.nlos import SOURCE_COMMIT, SOURCE_SHA256, cn0_variability, validate_observations

ROOT = Path(__file__).resolve().parents[2]
COMMIT = 'b13aa856f723a472bd35d45171017580642054cf'
EXTRACTOR_SHA = '7f9368986fbc9f96a0e7004194135e3f94dde851721ecc27c533859c4d905472'
RUN_EPOCHS = {'0610_KLT1_203': 203, '0610_KLT2_209': 209, '0610_KLT3_404': 404,
              '1109_KLT1_421': 421, '1109_KLT2_294': 294, '1109_KLT2_301': 301,
              '1109_KLT3_666': 666, '1116_KLT1_184': 184, '1116_KLT2_169': 169}


def check_timestamp_alignment(observation_utc: np.ndarray, label_gps: np.ndarray) -> dict:
    """Fail instead of silently truncating or matching labels by array position."""
    label_utc = np.asarray(label_gps) - 18.0  # Both acquisition years have GPS-UTC=18s.
    if not len(label_utc) or not np.all(np.diff(label_utc) > 0):
        raise ValueError('Label timestamps must be unique and increasing')
    if len(observation_utc) < 2 or not np.all(np.diff(observation_utc) > 0):
        raise ValueError('Observed epoch times must be unique and increasing')
    # Receiver epochs in June 2021 carry a ~5 ms fractional time offset.
    # Match timestamps one-to-one within RTKLIB's 50 ms epoch tolerance.
    positions = np.clip(np.searchsorted(observation_utc, label_utc), 1, len(observation_utc) - 1)
    positions = np.where(np.abs(observation_utc[positions] - label_utc) < np.abs(observation_utc[positions - 1] - label_utc),
                         positions, positions - 1)
    matched = observation_utc[positions]
    if len(set(positions)) != len(label_utc) or not np.allclose(matched, label_utc, rtol=0, atol=0.05):
        raise ValueError('Every label epoch needs a unique RINEX match within 50 milliseconds')
    if not np.allclose(np.diff(matched), 1., rtol=0, atol=0.001):
        raise ValueError('This extractor adapter requires contiguous 1 Hz labelled epochs')
    return {'epochs': len(label_utc), 'max_time_difference_ms': float(np.max(np.abs(matched - label_utc)) * 1000),
            'first_utc': float(matched[0]), 'last_utc': float(matched[-1]),
            'matched_timestamp_ms': np.rint(matched * 1000).astype('int64').tolist()}


def main():
    raw_root = ROOT / 'data/raw/klt_extended'
    source_manifest = json.loads((raw_root / 'manifest.json').read_text())
    source_manifest += json.loads((raw_root / 'nav_manifest.json').read_text())
    for item in source_manifest:
        path = ROOT / item['path']
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != item['sha256']:
            raise ValueError(f'Raw source hash changed: {path.name}')
    reference = ROOT / '.cache/klt_reference'
    reference.mkdir(parents=True, exist_ok=True)
    script = reference / 'extract_features.py'
    if not script.exists():
        script.write_bytes(urlopen(f'https://raw.githubusercontent.com/fossettlab/gnss-nlos-notebook/{COMMIT}/extract_features.py').read())
    if hashlib.sha256(script.read_bytes()).hexdigest() != EXTRACTOR_SHA:
        raise ValueError('External extractor differs from the reviewed pinned source')
    spec = importlib.util.spec_from_file_location('klt_external_extractor', script)
    extractor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extractor)
    published_path = ROOT / 'data/raw/klt_features/nlos_features.csv.gz'
    if hashlib.sha256(published_path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('Published four-run label table differs from pinned source')
    published = pd.read_csv(published_path)
    compact = ROOT / '.cache/klt_compact'
    metadata, audits, configurations, expected_labels = {}, {}, {}, []
    obs_times = {}
    for run, epochs in RUN_EPOCHS.items():
        cfg_path = raw_root / 'config' / f'{run}.json'
        cfg = json.loads(cfg_path.read_text())
        day = cfg['files'][0].split('/')[2]
        config_sha = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
        configurations[run] = {'sha256': config_sha, 'config': cfg}
        if day not in obs_times:
            nav_files = extractor.discover_nav_files(raw_root / 'data/GNSS' / day, 'hksc')
            # GPS and BeiDou navigation only; meteorological files are not inputs.
            nav_files = [p for p in nav_files if p.suffix.lower().endswith(('n', 'f'))]
            obs, _nav = extractor.read_rinex(raw_root / cfg['files'][0], nav_files)
            obs_times[day] = np.asarray([extractor.epoch_utc(obs.data[i].time) for i, _ in extractor.epoch_slices(obs)])
            mirror = compact / 'data/GNSS' / day
            (mirror / 'sta').mkdir(parents=True, exist_ok=True)
            for source in [raw_root / cfg['files'][0], *nav_files]:
                dest = mirror / ('sta' if source in nav_files else '') / source.name
                if not dest.exists():
                    os.link(source, dest)  # Same-disk local hard link; no second raw download.
        part = published[published.run == run]
        if len(part):
            labels = []
            for utc, group in part.groupby('utc', sort=True):
                satids = [int(prl.satid2no(s)) - 1 for s in group.sat]
                if min(satids) < 0:
                    raise ValueError('Published satellite ID cannot be converted with this RTKLIB build')
                los = [s for s, nlos in zip(satids, group.nlos) if nlos == 0]
                labels.append([float(utc) + 18., satids, los, None, {}])
            label_source = 'pinned published numeric table'
        else:
            full = extractor.safe_load_labels(raw_root / 'label' / run / 'nlos.pkl')
            labels = [[float(row[0]), [int(s) for s in row[1]], [int(s) for s in row[2]], None, {}] for row in full]
            del full
            gc.collect()
            label_source = 'original KLT fisheye labels'
        if len(labels) != epochs:
            raise ValueError(f'Unexpected label epoch count: {run}')
        audit = check_timestamp_alignment(obs_times[day], np.asarray([l[0] for l in labels]))
        audits[run] = {**audit, 'label_source': label_source}
        destination = compact / 'label' / run
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / 'nlos.pkl').open('wb') as stream:
            pickle.dump(labels, stream, protocol=4)
        shutil.copyfile(raw_root / cfg['gt'], destination / 'gt.csv')
        metadata[run] = {'day': day, 'obs': Path(cfg['files'][0]).name, 'sta_prefix': 'hksc',
                         'start': audit['first_utc'], 'end': audit['last_utc']}
        for matched_ms, (gps, all_sat, los, _, _) in zip(audit['matched_timestamp_ms'], labels):
            if not set(los) <= set(all_sat):
                raise ValueError('LOS label IDs must be a subset of observed IDs')
            for sat in all_sat:
                buf = prl.Arr1Dchar(4)
                prl.satno2id(sat + 1, buf)
                if buf.ptr.startswith(('G', 'C')):
                    expected_labels.append((run, matched_ms, buf.ptr, int(sat not in los)))
    # Upstream CLI expects a data-only RUNS module absent from its public repository.
    (reference / 'fetch_klt_data.py').write_text('RUNS = ' + repr(metadata) + '\n', encoding='utf-8')
    output = ROOT / 'data/processed/klt_extended'
    output.mkdir(parents=True, exist_ok=True)
    feature_path = output / 'extracted_features.csv.gz'
    subprocess.run([sys.executable, str(script), '--raw', str(compact), '--runs', *RUN_EPOCHS,
                    '--out', str(feature_path)], check=True, cwd=ROOT)
    table = pd.read_csv(feature_path)
    observed_labels = [(str(r.run), round(r.utc * 1000), r.sat, int(r.nlos)) for r in table.itertuples()]
    if len(observed_labels) != len(expected_labels) or set(observed_labels) != set(expected_labels):
        raise ValueError('Extracted observation labels do not exactly match the timestamp/satellite reference')
    frame = table.rename(columns={'run': 'trip_id', 'sat': 'satellite_id', 'sys': 'system', 'cn0': 'cn0_dbhz',
                                 'elev_deg': 'elevation_deg', 'pr_resid_norm': 'residual_normalized',
                                 'pr_rate_consistency': 'pr_rate_consistency_mps', 'cn0_std': 'cn0_std_dbhz'})
    frame['timestamp_ms'] = np.rint(frame.utc * 1000).astype('int64')
    frame['device_id'] = 'ublox_f9p'
    frame['data_kind'] = 'real'
    frame['scene_id'] = frame.trip_id.str.extract(r'_(KLT[123])_', expand=False)
    frame['signal_label'] = np.where(frame.nlos == 1, 'nlos', 'los')
    frame['signal_label_source'] = 'fisheye_sky_mask'
    frame['obstruction_type'] = 'unknown'
    frame['collection_date'] = frame.trip_id.map({r: f"{m['day'][:4]}-{m['day'][4:6]}-{m['day'][6:]}" for r, m in metadata.items()})
    frame['cn0_std_dbhz'] = cn0_variability(frame)
    frame = validate_observations(frame)
    frame.to_csv(output / 'satellites.csv', index=False)
    report = {'source_url': 'https://github.com/ebhrz/KLTDataset',
              'source_commit': (raw_root / 'config_commit.txt').read_text().strip(),
              'source_sha256': hashlib.sha256(json.dumps(source_manifest, sort_keys=True).encode()).hexdigest(),
              'prepared_sha256': hashlib.sha256((output / 'satellites.csv').read_bytes()).hexdigest(),
              'original_dataset': 'https://github.com/ebhrz/KLTDataset', 'license': 'GPL-3.0',
              'features_origin': 'All nine runs re-extracted with the same pinned Fossett RTKLIB SPP extractor; label times strictly verified; trailing C/N0 std recomputed locally',
              'extractor': {'source_url': 'https://github.com/fossettlab/gnss-nlos-notebook', 'commit': COMMIT,
                            'sha256': EXTRACTOR_SHA, 'license': 'GPL-3.0-or-later'},
              'published_label_source_commit': SOURCE_COMMIT,
              'source_files': source_manifest, 'configurations': configurations, 'alignment': audits,
              'excluded_unreliable_runs': ['1116_KLT3_649'],
              'rows': len(frame), 'epochs': len(frame.drop_duplicates(['trip_id', 'timestamp_ms'])),
              'dates': sorted(frame.collection_date.unique().tolist()),
              'per_run': {r: {'rows': len(g), 'epochs': g.timestamp_ms.nunique(), 'nlos_count': int(g.nlos.sum()),
                              'area': str(g.scene_id.iloc[0])} for r, g in frame.groupby('trip_id')}}
    (output / 'preprocess_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Prepared', report['epochs'], 'epochs /', report['rows'], 'observations', flush=True)


if __name__ == '__main__':
    main()
