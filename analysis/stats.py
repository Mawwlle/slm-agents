"""Paired comparisons used throughout: a macro over tracks, a task bootstrap within track, a sign-flip permutation."""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20261108)
DRAWS = 10_000


def macro(values: pd.Series, tracks: pd.Series) -> float:
    """Mean over tracks of the per-track mean, so a large track does not drown a small one."""

    return float(values.groupby(tracks.to_numpy()).mean().mean())


def paired(a: pd.Series, b: pd.Series, tracks: pd.Series, alpha: float = 0.05) -> dict:
    """a − b on their common tasks (index = task id): point, CI and two-sided permutation p."""

    common = a.index.intersection(b.index)
    d = a.loc[common].astype(float).to_numpy() - b.loc[common].astype(float).to_numpy()
    t = tracks.reindex(common).to_numpy()
    groups = [np.flatnonzero(t == k) for k in np.unique(t)]
    point = float(np.mean([d[g].mean() for g in groups]))
    boot = np.mean([d[g][RNG.integers(0, len(g), (DRAWS, len(g)))].mean(1) for g in groups], 0)
    flips = np.mean([(d[g] * RNG.choice([-1, 1], (DRAWS, len(g)))).mean(1) for g in groups], 0)
    p = (np.sum(np.abs(flips) >= abs(point) - 1e-12) + 1) / (DRAWS + 1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {'diff': point, 'lo': float(lo), 'hi': float(hi), 'p': float(p), 'n': len(common)}


def holm(p: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Which hypotheses stay rejected under Holm's step-down correction."""

    order = sorted(p, key=lambda k: p[k])
    keep = {}
    for i, key in enumerate(order):
        if p[key] > alpha / (len(order) - i):
            keep |= {k: False for k in order[i:]}
            break
        keep[key] = True
    return keep
