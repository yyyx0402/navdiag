"""Describe measured C/N0 domain shifts without refitting or tuning models.

Satellite observations are correlated. KS/Wasserstein values below describe
empirical distributions; no IID p-values or significance claims are reported.
"""
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]


def compare(a, b):
    return {'ks_distance': float(ks_2samp(a, b).statistic),
            'wasserstein_dbhz': float(wasserstein_distance(a, b)),
            'median_difference_dbhz': float(np.median(a) - np.median(b))}


def audit(frame):
    frame = frame[frame.cn0_dbhz.notna() & frame.cn0_dbhz.gt(0)].copy()
    frame['elevation_bin'] = pd.cut(frame.elevation_deg, [0, 15, 30, 60, 90], include_lowest=False)
    areas = {}
    for area, part in frame.groupby('scene_id'):
        areas[area] = {'observed_rows': len(part), 'cn0_quantiles_dbhz':
                       {str(q): float(part.cn0_dbhz.quantile(q)) for q in [.1, .25, .5, .75, .9]},
                       'fraction_below_30': float(part.cn0_dbhz.lt(30).mean()),
                       'by_reference_label': {label: {'rows': len(g), 'median_dbhz': float(g.cn0_dbhz.median()),
                                                     'fraction_below_30': float(g.cn0_dbhz.lt(30).mean())}
                                              for label, g in part.groupby('signal_label')}}
    pairs = {}
    keys = ['signal_label', 'system', 'elevation_bin']
    for aa, bb in combinations(sorted(areas), 2):
        a, b = frame[frame.scene_id == aa], frame[frame.scene_id == bb]
        strata_a = {key: g.cn0_dbhz.to_numpy() for key, g in a[a.signal_label.ne('unknown')].groupby(keys, observed=True)}
        strata_b = {key: g.cn0_dbhz.to_numpy() for key, g in b[b.signal_label.ne('unknown')].groupby(keys, observed=True)}
        shared = [key for key in strata_a.keys() & strata_b.keys() if min(len(strata_a[key]), len(strata_b[key])) >= 30]
        weights = np.array([min(len(strata_a[key]), len(strata_b[key])) for key in shared], dtype=float)
        weights /= weights.sum()
        matched = [compare(strata_a[key], strata_b[key]) for key in shared]
        pairs[f'{aa}_vs_{bb}'] = {'all_observed': compare(a.cn0_dbhz, b.cn0_dbhz),
                                  'matched_label_system_elevation': {
                                      'shared_strata': len(shared),
                                      'mean_ks_distance': float(sum(w * d['ks_distance'] for w, d in zip(weights, matched))),
                                      'mean_wasserstein_dbhz': float(sum(w * d['wasserstein_dbhz'] for w, d in zip(weights, matched)))}}
    return {'areas': areas, 'pairs': pairs}, frame


if __name__ == '__main__':
    reports, frames = {}, {}
    for name, relative in [('klt', 'data/processed/klt_extended/satellites.csv'),
                           ('urbannav', 'data/processed/urbannav/satellites.csv')]:
        reports[name], frames[name] = audit(pd.read_csv(ROOT / relative))
    reports['method'] = 'Observed positive C/N0 only; no fitting. Empirical distances, no p-values. Matched strata: label, GPS/BeiDou, SPP elevation in 0-15/15-30/30-60/60-90 degrees; >=30 rows in both areas; minimum-count weights.'
    reports['limitations'] = ['One or few correlated drives per region; empirical distances do not establish population significance.',
                              'SPP elevation can be biased; matching four bins is incomplete control for geometry.',
                              'Class-conditional UrbanNav distributions use accepted map references only; unknown exclusions may bias them.',
                              'KLT and UrbanNav differ in antenna, dates and reference method; between-dataset differences are not city effects.',
                              'All six regions are in Hong Kong; no inter-city C/N0 distribution is measured.']
    (ROOT / 'docs/experiments/nlos_domain_audit.json').write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding='utf-8')
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7))
    for ax, dataset, label, title in [(axes[0, 0], 'klt', None, 'KLT: all observed C/N0'),
                                    (axes[0, 1], 'urbannav', None, 'UrbanNav: all observed C/N0'),
                                    (axes[1, 0], 'urbannav', 'los', 'UrbanNav: accepted LOS references'),
                                    (axes[1, 1], 'urbannav', 'nlos', 'UrbanNav: accepted NLOS references')]:
        f = frames[dataset]
        if label is not None:
            f = f[f.signal_label == label]
        for area, part in f.groupby('scene_id'):
            x = np.sort(part.cn0_dbhz.to_numpy())
            ax.step(x, np.arange(1, len(x) + 1) / len(x), where='post', label=area)
        ax.axvline(30, color='#777', linestyle=':', linewidth=1)
        ax.set(title=title, xlim=(15, 55), ylim=(0, 1), xlabel='C/N0 (dB-Hz)', ylabel='Empirical CDF')
        ax.legend(frameon=False)
        ax.grid(alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
    fig.text(.5, .012, 'One city; correlated observations. Conditional panels exclude unknown references. No significance claim.',
             ha='center', fontsize=9, color='#555')
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(ROOT / 'docs/images/nlos-cn0-domain-audit.png', dpi=180)
    fig.savefig(ROOT / 'docs/images/nlos-cn0-domain-audit.svg')
    for dataset, report in reports.items():
        if isinstance(report, dict) and 'areas' in report:
            print(dataset, {area: {'median': m['cn0_quantiles_dbhz']['0.5'],
                                    'los': m['by_reference_label'].get('los', {}).get('median_dbhz'),
                                    'nlos': m['by_reference_label'].get('nlos', {}).get('median_dbhz')}
                            for area, m in report['areas'].items()})
            print(report['pairs'])
