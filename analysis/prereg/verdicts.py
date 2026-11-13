"""Every pre-registered verdict in one table, and whether the HOLDS ones survive a Holm correction → data/results/verdicts.json.

    uv run --with pandas --with numpy --with scikit-learn python -m analysis.prereg.verdicts

Holm can only withdraw a HOLDS, so a p is needed only where the rule held. For a rule that needs k passing cells
(e.g. ≥ 2 of 3 budgets), its p is the k-th smallest cell p within the cell family that passed. NOT SHOWN verdicts enter
the correction with p = 1, which is conservative for the others.
"""

import json

import numpy as np
import pandas as pd

from analysis.prereg import stage3
from analysis.prereg.analyze import RNG
from analysis.prereg.load import ROOT, episodes

DRAWS = 10_000


def load(name: str) -> dict:
    return json.loads((ROOT / f'data/results/{name}').read_text())


def sign_flip(d: np.ndarray, tracks: np.ndarray) -> float:
    """Two-sided paired permutation p of the macro over tracks of d."""

    groups = [np.flatnonzero(tracks == t) for t in np.unique(tracks)]
    point = np.mean([d[g].mean() for g in groups])
    flips = np.mean([(d[g] * RNG.choice([-1, 1], (DRAWS, len(g)))).mean(1) for g in groups], 0)
    return float((np.sum(np.abs(flips) >= abs(point) - 1e-12) + 1) / (DRAWS + 1))


def cascade_p(result: dict, key: str, need: int) -> float:
    """The rule needs `need` passing budgets of one family for some model: the smallest such `need`-th p."""

    best = 1.0
    for big in result['by_big'].values():
        for curve in big['curves'].values():
            for family in ('big×', 'small×'):
                ps = sorted(r[f'{key}_p'] for label, r in curve['matched'].items()
                            if label.startswith(family) and key in r and r[key][1] > 0)
                if len(ps) >= need:
                    best = min(best, ps[need - 1])
    return best


def hr4_p(hr: dict) -> float:
    """HR4 holds at a budget where the pair beats the model-only router; p of the best such budget."""

    frame = episodes()
    test = frame[(frame.split == 'test') & (frame.repeat == 0)]
    best = 1.0
    for row in hr['HR4']['budgets'].values():
        if not (row['diff'][0] >= 0.02 and row['diff'][1] > 0):
            continue
        a = test[(test.model == row['pair'][0]) & (test.condition == row['pair'][1])].set_index('task_id')
        b = test[(test.model == row['model_only'][0]) & (test.condition == row['model_only'][1])].set_index('task_id')
        common = a.index.intersection(b.index)
        d = a.loc[common, 'passed'].astype(float).to_numpy() - b.loc[common, 'passed'].astype(float).to_numpy()
        best = min(best, sign_flip(d, a.loc[common, 'track'].to_numpy()))
    return best


def l2_p() -> float:
    """Bootstrap p of L2's gap (mean skill_inline lift of the new models minus the gateway models')."""

    data = stage3.frame()
    models = stage3.phase_l(data)
    matrix, tracks = stage3.lift_matrix(data, 'skill_inline', models + stage3.GATEWAY)
    groups = [matrix[tracks == t].to_numpy() for t in stage3.SOLO_TRACKS]
    draws = np.mean([g[RNG.integers(0, len(g), (DRAWS, len(g)))].mean(1) for g in groups], 0)
    gap = draws[:, :len(models)].mean(1) - draws[:, len(models):].mean(1)
    return float(min(1.0, 2 * min((gap >= 0).mean(), (gap <= 0).mean()) + 1 / DRAWS))


def holm(ps: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    order = sorted(ps, key=ps.get)
    survive, m = {}, len(order)
    for i, key in enumerate(order):
        if ps[key] > alpha / (m - i):
            for rest in order[i:]:
                survive[rest] = False
            break
        survive[key] = True
    return survive


def main() -> None:
    a, hr, s3, cp = load('prereg-stage1.json'), load('prereg-stage2.json'), load('prereg-stage3.json'), load('prereg-stage4.json')
    verdicts = {f'A{i}': a[f'A{i}']['verdict'] for i in range(1, 6)}
    verdicts |= {k: hr[k].get('verdict', hr[k].get('verdict_hr3')) for k in ('HR1', 'HR2', 'HR3', 'HR4', 'HR5')}
    verdicts |= {k: s3[k]['verdict'] for k in ('L1', 'L2', 'L3')}
    verdicts['PD3'] = 'HOLDS' if s3['PD3'].get('prediction_met') else 'NOT SHOWN'
    verdicts |= {k: s3[k]['verdict'] for k in ('PD1', 'PD2', 'PD4', 'PD5')}
    verdicts |= {k: cp[k]['verdict'] for k in ('CP1', 'CP2', 'CP3', 'CP4', 'CP5')}
    # a superiority verdict stands only if its rule passes ≤ 5% of label-shuffled copies (PREREG-HR, PREREG-CP);
    # CP3 is non-inferiority, which equal systems satisfy, so the shuffle does not test it (PREREG-CP log)
    shuffled = {'HR2': hr['calibration']['hr2_false_pass'], 'HR3': hr['calibration']['hr3_false_pass'],
                'CP4': cp['calibration']['CP4'], 'CP5': cp['calibration']['CP5']}
    for key, rate in shuffled.items():
        if rate > 0.05:
            verdicts[key] = 'NOT SHOWN'
    p = {k: 1.0 for k in verdicts}
    holds = {k for k, v in verdicts.items() if v == 'HOLDS'}
    found = {'A3': lambda: cascade_p(a['A3'], 'diff', 2), 'HR3': lambda: cascade_p(hr['HR3'], 'diff_switch', 2),
             'HR4': lambda: hr4_p(hr), 'L2': l2_p, 'CP2': lambda: cp['CP2']['diff'][3], 'CP3': lambda: cp['CP3']['diff'][3]}
    for key in holds:
        p[key] = found[key]()
    survive = holm(p)
    table = pd.DataFrame({'verdict': verdicts, 'p': p, 'holm_survives': {k: survive.get(k, False) for k in verdicts}})
    (ROOT / 'data/results/verdicts.json').write_text(table.to_json(orient='index', indent=1))
    print(table.to_string())
    print(f'{len(holds)} HOLDS of {len(verdicts)}; surviving Holm over all {len(verdicts)}: {sum(survive.get(k, False) for k in holds)}')


if __name__ == '__main__':
    main()
