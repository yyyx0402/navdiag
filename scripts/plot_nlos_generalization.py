"""Render scientific comparison figures from recorded results, without fitting."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'logistic_regression': 'LR', 'random_forest': 'RF', 'mlp': 'MLP', 'cnn': 'CNN',
         'lstm': 'LSTM', 'gru': 'GRU', 'transformer': 'Transformer', 'cn0_rule': 'C/N0 rule'}


if __name__ == '__main__':
    r = json.loads((ROOT / 'docs/experiments/urbannav_generalization_report.json').read_text(encoding='utf-8'))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
    x = np.arange(len(NAMES))
    for i, (key, label, color) in enumerate([
        ('external_transfer', 'Frozen KLT -> UrbanNav', '#087f8c'),
        ('district_holdout', 'UrbanNav held-out district', '#e4a44b')]):
        y = [r[key]['models'][n]['test_macro']['f1'] for n in NAMES]
        bars = axes[0].bar(x + (i - .5) * .36, y, width=.36, label=label, color=color)
        axes[0].bar_label(bars, labels=[f'{v:.3f}' for v in y], fontsize=7, rotation=90, padding=3)
    axes[0].set(xticks=x, xticklabels=[*list(NAMES.values())[:-1], 'C/N0'], ylim=(0, 1),
                ylabel='Equal-district mean test F1', title='Fixed models and validation-only thresholds')
    axes[0].tick_params(axis='x', labelsize=9)
    axes[0].legend(frameon=False, fontsize=8)
    xx = np.arange(3)
    for i, (name, key, label, color) in enumerate([
        ('random_forest', 'external_transfer', 'RF: frozen transfer', '#087f8c'),
        ('transformer', 'external_transfer', 'Transformer: frozen transfer', '#65b8be'),
        ('random_forest', 'district_holdout', 'RF: district holdout', '#bd7920'),
        ('transformer', 'district_holdout', 'Transformer: district holdout', '#edbc77')]):
        model = r[key]['models'][name]
        y = [model['per_area'][area]['f1'] if key == 'external_transfer' else
             model['per_area'][area]['test_mean']['f1'] for area in ['TST', 'WP', 'MK']]
        bars = axes[1].bar(xx + (i - 1.5) * .2, y, width=.2, label=label, color=color)
        axes[1].bar_label(bars, labels=[f'{v:.3f}' for v in y], fontsize=7, rotation=90, padding=3)
    axes[1].set(xticks=xx, xticklabels=['TST', 'WP', 'MK'], ylim=(0, 1), ylabel='Test F1',
                title='Regional differences remain visible')
    axes[1].legend(frameon=True, facecolor='white', edgecolor='white', fontsize=8, loc='lower left')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.text(.5, .012, 'Map-derived building references; coverage TST 58.6%, WP 59.6%, MK 19.9%. Three drives, one city.',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, .04, 1, 1))
    for suffix in ['png', 'svg']:
        fig.savefig(ROOT / f'docs/images/urbannav-generalization.{suffix}', dpi=180)
