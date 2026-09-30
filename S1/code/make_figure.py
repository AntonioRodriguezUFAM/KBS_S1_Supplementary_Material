#!/usr/bin/env python3
"""make_figure.py — regenerates Figure 1 of the manuscript from stageA_combined_48cells.csv.
Usage: python3 make_figure.py            (writes fig_stageA_combined.png next to this script)
Intervals in Table 3 are block bootstraps over the six mission-level means; this script prints them too."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
C = pd.read_csv(__import__('os').path.join(__import__('os').path.dirname(__file__), '..', 'results', 'stageA_combined_48cells.csv')); modes = ['reset','retain','ctxnoage','ctx']
PM = C.groupby(['source','mission'])[modes+['wema_err_power']].mean()
PM['K1'] = PM.retain-PM.reset; PM['K2'] = PM.ctxnoage-PM.retain; PM['age'] = PM.ctx-PM.ctxnoage
rng = np.random.default_rng(7)
for c in ['K1','K2','age']:
    v = PM[c].values; boot = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(10000)])
    print(f'{c}: mean {v.mean():+.4f} W, 95% [{np.percentile(boot,2.5):+.4f}, {np.percentile(boot,97.5):+.4f}], '
          f'missions improved {(v<0).sum()}/6, cells improved {(C[c]<0).sum()}/{len(C)}')
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
lab = ['R-Reset','R-Retain','R-Context (no age)','R-Context','W-EMA (ref.)']; x = np.arange(len(PM)); w = 0.15
for i, (m, l) in enumerate(zip(modes+['wema_err_power'], lab)): ax[0].bar(x+(i-2)*w, PM[m].values, w, label=l)
ax[0].set_xticks(x); ax[0].set_xticklabels([f'{s[:7]}\n{m}' for s, m in PM.index], fontsize=8)
ax[0].set_ylabel('early-visit power MAE [W]'); ax[0].set_title('(a) six missions, mean of 8 returning cells'); ax[0].legend(fontsize=8)
ax[1].axhline(0, color='k', lw=0.8)
for i, c in enumerate(['K1','K2','age']):
    ax[1].scatter(np.full(len(C), i)+rng.uniform(-0.12, 0.12, len(C)), C[c], s=14, alpha=0.5, label='cell' if i == 0 else None)
    ax[1].scatter(np.full(6, i)+np.linspace(-0.2, 0.2, 6), PM[c].values, marker='D', s=40, color='k', label='mission mean' if i == 0 else None)
ax[1].set_xticks([0,1,2]); ax[1].set_xticklabels(['K1: Retain − Reset','K2: Context − Retain','Age: Context − Context(no age)'])
ax[1].set_ylabel('paired difference [W]'); ax[1].set_title('(b) 48 returning cells; 6 mission means'); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(__import__('os').path.join(__import__('os').path.dirname(__file__), '..', 'results', 'fig_stageA_combined.png'), dpi=160); print('wrote fig_stageA_combined.png')
