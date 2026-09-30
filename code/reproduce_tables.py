#!/usr/bin/env python3
"""reproduce_tables.py — prints Table 3 (per-mission means), Table 4 (contrasts with bootstrap intervals,
leave-one-mission-out ranges, sign tests) and Table 5 (sensitivity grid) of the manuscript from the result files."""
import os, math, numpy as np, pandas as pd
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
C = pd.read_csv(os.path.join(R, 'stageA_combined_48cells.csv')); modes = ['reset', 'retain', 'ctxnoage', 'ctx']
PM = C.groupby(['source', 'mission'])[modes + ['wema_err_power']].mean()
print('== Table 3: early-visit power MAE (W) per mission, mean over 8 returning cells\n', PM.round(4).to_string(), '\n')
PM['K1'] = PM.retain - PM.reset; PM['K2'] = PM.ctxnoage - PM.retain; PM['age'] = PM.ctx - PM.ctxnoage
rng = np.random.default_rng(7); print('== Table 4: contrasts (mission-level; bootstrap over six missions is descriptive)')
for c, name in [('K1', 'Retain - Reset'), ('K2', 'Context(no age) - Retain'), ('age', 'Context - Context(no age)')]:
    v = PM[c].values; boot = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(10000)])
    loo = [np.delete(v, i).mean() for i in range(len(v))]; neg = int((v < 0).sum()); n = len(v)
    p = 2 * min(sum(math.comb(n, k) for k in range(0, neg + 1)), sum(math.comb(n, k) for k in range(neg, n + 1))) / 2 ** n
    print(f'  {name:26s} mean {v.mean():+.4f} W | boot95 [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}] | LOO [{min(loo):+.4f}, {max(loo):+.4f}] | missions {neg}/6 | cells {(C[c] < 0).sum()}/48 | sign p={min(1, p):.3f}')
    arch = PM.xs('archive_replay_6Sep', level='source')[c].mean() if 'archive_replay_6Sep' in PM.index.get_level_values(0) else float('nan')
    live = PM[c].values[[i for i, s in enumerate(PM.index.get_level_values(0)) if 'live' in s]].mean()
    print(f'  {"":26s} archive {arch:+.4f} | instrumented {live:+.4f}')
print('\n== Table 5: predeclared sensitivity grid (four missions)\n', pd.read_csv(os.path.join(R, 'stageA_K2_sensitivity_grid.csv')).round(4).to_string(index=False))
