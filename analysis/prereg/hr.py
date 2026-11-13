"""Stage 2 tests (PREREG-HR.md) → data/results/prereg-stage2.json.

    uv run --with pandas --with numpy --with scikit-learn python -m analysis.prereg.hr [copies]
"""

import json
import sys

import numpy as np
import pandas as pd

from analysis.prereg.analyze import RNG, a1_detector, a3_policies, macro_diff, paired
from analysis.prereg.load import BIG, ROOT, SIZE, TRACKS, episodes

HARNESSES = ['base', 'skill', 'watchdog', 'memory', 'skill_inline', 'strict', 'skill_inline+memory',
             'strict+skill_inline+memory']
COMPONENTS = ['skill', 'skill_inline', 'memory', 'watchdog', 'strict']
SMALL = [m for m, s in SIZE.items() if s < 12 and m != 'qwen3.5-0.8b-wsp']
BUDGETS = [1e13, 2e13, 5e13]


def table(frame: pd.DataFrame, condition: str, models: list[str]) -> pd.DataFrame:
    """task × model matrix of paired lifts (condition − base), NaN where a run is missing."""

    columns = {}
    for m in models:
        d = paired(frame, m, condition, 'base')
        if d is not None:
            columns[m] = d['d']
    return pd.DataFrame(columns)


def track_of(frame: pd.DataFrame) -> pd.Series:
    return frame.drop_duplicates('task_id').set_index('task_id').track


def macro_draws(matrix: pd.DataFrame, tracks: pd.Series, draws: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-model macro lift (point) and its bootstrap draws, with one task resample shared by all models."""

    point, sample = [], np.zeros((draws, matrix.shape[1]))
    by_track = [matrix[tracks.reindex(matrix.index) == t].to_numpy(float) for t in TRACKS]
    by_track = [x for x in by_track if len(x)]
    for x in by_track:
        point.append(np.nanmean(x, 0))
        idx = RNG.integers(0, len(x), (draws, len(x)))
        sample += np.stack([np.nanmean(x[i], 0) for i in idx])
    return np.mean(point, 0), sample / len(by_track)


def hr1(frame: pd.DataFrame, draws: int = 2000) -> dict:
    tracks = track_of(frame)
    out: dict = {'components': {}}
    for c in COMPONENTS:
        matrix = table(frame, c, SMALL + BIG)
        small = [m for m in SMALL if m in matrix]
        big = [m for m in BIG if m in matrix]
        if not small:
            continue
        point, sample = macro_draws(matrix, tracks, draws)
        cols = list(matrix.columns)
        s_idx, b_idx = [cols.index(m) for m in small], [cols.index(m) for m in big]
        row = {'per_model': {m: float(point[cols.index(m)]) for m in cols},
               'small': [float(point[s_idx].mean()), *np.quantile(sample[:, s_idx].mean(1), [0.025, 0.975]).tolist()]}
        if big:
            row['big'] = [float(point[b_idx].mean()), *np.quantile(sample[:, b_idx].mean(1), [0.025, 0.975]).tolist()]
            d = sample[:, s_idx].mean(1) - sample[:, b_idx].mean(1)
            row['d'] = [row['small'][0] - row['big'][0], *np.quantile(d, [0.025, 0.975]).tolist()]
        out['components'][c] = row
    ds = [r['d'] for r in out['components'].values() if 'd' in r]
    out['verdict'] = 'HOLDS' if any(d[1] > 0 for d in ds) and any(d[2] < 0 for d in ds) else 'NOT SHOWN'
    return out


def q_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Macro Q and mean FLOPs per (model, harness, part) with part ∈ {fit (screen+train), test}."""

    part = frame[frame.repeat == 0].assign(part=np.where(frame[frame.repeat == 0].split == 'test', 'test', 'fit'))
    q = part.groupby(['model', 'condition', 'part', 'track']).passed.mean().groupby(['model', 'condition', 'part']).mean()
    c = part.groupby(['model', 'condition', 'part']).flops.mean()
    return pd.DataFrame({'q': q, 'flops': c}).reset_index()


def choose(frame: pd.DataFrame) -> dict:
    """H* (best single harness on fit) and h*(m) (best per model on fit, not dearer than H* for m)."""

    Q = q_table(frame)
    fit = Q[Q.part == 'fit']
    models = sorted(fit.model.unique())
    common = [h for h in HARNESSES if all(((fit.model == m) & (fit.condition == h)).any() for m in models)]
    h_all = max(common, key=lambda h: fit[fit.condition == h].q.mean())
    own = {}
    for m in models:
        rows = fit[(fit.model == m) & fit.condition.isin(HARNESSES)].set_index('condition')
        cap = rows.loc[h_all, 'flops']
        own[m] = rows[rows.flops <= cap * 1.0001].q.idxmax()
    return {'single': h_all, 'own': own, 'common': common}


def hr2(frame: pd.DataFrame, chosen: dict, draws: int = 2000) -> dict:
    test = frame[frame.split == 'test']
    tracks = track_of(frame)
    per, cols = {}, {}
    for m, h in chosen['own'].items():
        d = paired(test, m, h, chosen['single'])
        cols[m] = d['d']
        per[m] = {'harness': h, 'diff': list(macro_diff(d, alpha=0.05)) if h != chosen['single'] else [0.0, 0.0, 0.0]}
    point, sample = macro_draws(pd.DataFrame(cols), tracks, draws)
    mean = [float(point.mean()), *np.quantile(sample.mean(1), [0.025, 0.975]).tolist()]
    moved = [v for v in per.values() if v['harness'] != chosen['single']]
    wins = sum(v['diff'][1] > 0 for v in moved)
    verdict = 'HOLDS' if moved and wins >= 2 / 3 * len(moved) and mean[1] > 0 else (
        'REVERSED' if mean[2] < 0 else 'NOT SHOWN')
    return {'single': chosen['single'], 'per_model': per, 'mean': mean, 'wins': wins, 'moved': len(moved), 'verdict': verdict}


def hr3(frame: pd.DataFrame, chosen: dict, a1: dict, draws: int = 2000) -> dict:
    # a model whose own harness is base has no switch: its retry would be the same run
    a3 = a3_policies(frame, a1, switch={m: h for m, h in chosen['own'].items() if h != 'base'}, draws=draws)
    best = worst = 0
    for d in a3.get('by_big', {}).values():
        for c in d['curves'].values():
            for family in ('big×', 'small×'):
                rows = [r for label, r in c['matched'].items() if label.startswith(family) and 'diff_switch' in r]
                best = max(best, sum(r['diff_switch'][1] > 0 for r in rows))
                worst = max(worst, sum(r['diff_switch'][2] < 0 for r in rows))
    a3['verdict_hr3'] = 'HOLDS' if best >= 2 else ('REVERSED' if worst >= 2 else 'NOT SHOWN')
    return a3


def hr4(frame: pd.DataFrame) -> dict:
    Q = q_table(frame)
    fit, test = Q[Q.part == 'fit'], Q[Q.part == 'test']
    pairs = fit[fit.condition.isin(HARNESSES)]
    out: dict = {'budgets': {}}
    for budget in BUDGETS:
        ok = pairs[pairs.flops <= budget]
        if ok.empty:
            continue
        pair = ok.loc[ok.q.idxmax(), ['model', 'condition']].tolist()
        base = ok[ok.condition == 'base']
        if base.empty:
            continue
        single = base.loc[base.q.idxmax(), ['model', 'condition']].tolist()
        out['budgets'][f'{budget:.0e}'] = {
            'pair': pair, 'model_only': single, 'diff': pair_diff(frame, pair, single),
            'cost_pair': float(test[(test.model == pair[0]) & (test.condition == pair[1])].flops.iloc[0]),
            'cost_model': float(test[(test.model == single[0]) & (test.condition == 'base')].flops.iloc[0])}
    rows = out['budgets'].values()
    out['frontier'] = {'pairs': fit_frontier(pairs, test), 'models': fit_frontier(pairs[pairs.condition == 'base'], test)}
    out['verdict'] = 'HOLDS' if any(r['diff'][0] >= 0.02 and r['diff'][1] > 0 for r in rows) else 'NOT SHOWN'
    return out


def pair_diff(frame: pd.DataFrame, a: list, b: list) -> list:
    """Test macro Q of pair a minus pair b, paired by task, with a 95% task-bootstrap CI."""

    t = frame[(frame.split == 'test') & (frame.repeat == 0)]
    x = t[(t.model == a[0]) & (t.condition == a[1])].set_index('task_id')
    y = t[(t.model == b[0]) & (t.condition == b[1])].set_index('task_id')
    common = x.index.intersection(y.index)
    d = pd.DataFrame({'d': x.loc[common, 'passed'].astype(float).to_numpy() - y.loc[common, 'passed'].astype(float).to_numpy(),
                      'track': x.loc[common, 'track'].to_numpy()})
    return list(macro_diff(d))


def rule_harness(model: str, cost: dict, budget: float) -> str:
    """HR5: memory whenever it fits the budget; the skill only for the big models."""

    if model in BIG:
        return 'skill_inline+memory' if cost.get('skill_inline+memory', np.inf) <= budget else 'skill'
    return 'memory' if cost.get('memory', np.inf) <= budget else 'base'


def hr5(frame: pd.DataFrame) -> dict:
    Q = q_table(frame)
    fit = Q[(Q.part == 'fit') & Q.condition.isin(HARNESSES)]
    cost = {m: dict(zip(g.condition, g.flops)) for m, g in fit.groupby('model')}
    out: dict = {'budgets': {}}
    for budget in np.geomspace(4e12, 1.6e14, 16):
        ok = fit[fit.flops <= budget]
        rules = ok[[h == rule_harness(m, cost[m], budget) for m, h in zip(ok.model, ok.condition)]]
        if ok.empty or rules.empty:
            continue
        search = ok.loc[ok.q.idxmax(), ['model', 'condition']].tolist()
        rule = rules.loc[rules.q.idxmax(), ['model', 'condition']].tolist()
        d = [0.0, 0.0, 0.0] if rule == search else pair_diff(frame, rule, search)
        out['budgets'][f'{budget:.2e}'] = {'rule': rule, 'search': search, 'diff': d}
    diffs = [r['diff'] for r in out['budgets'].values()]
    out['mean_diff'] = float(np.mean([d[0] for d in diffs]))
    out['verdict'] = 'HOLDS' if all(d[1] > -0.02 for d in diffs) else 'NOT SHOWN'
    return out


def fit_frontier(fit: pd.DataFrame, test: pd.DataFrame) -> list:
    """Pairs on the fit-split Pareto frontier, with their test cost and quality."""

    points = fit.sort_values('flops')
    frontier, best = [], -1.0
    for r in points.itertuples():
        if r.q > best:
            best = r.q
            t = test[(test.model == r.model) & (test.condition == r.condition)]
            if len(t):
                frontier.append({'model': r.model, 'harness': r.condition, 'flops': float(t.flops.iloc[0]), 'q': float(t.q.iloc[0])})
    return frontier


def calibration(frame: pd.DataFrame, chosen: dict, a1: dict, copies: int = 50) -> dict:
    """The HR2 and HR3 rules on copies where the compared runs are exchangeable.

    HR2: per (model, task), the harness labels are shuffled among that task's runs.
    HR3: per (model, task), the switch run (h*(m)) and the plain retry (base repeat 1) trade places with p = 1/2."""

    hr2_passes = hr3_passes = 0
    part = frame[(frame.repeat == 0) & frame.condition.isin(HARNESSES)].copy()
    for _ in range(copies):
        shuffled = part.copy()
        shuffled['passed'] = shuffled.groupby(['model', 'task_id']).passed.transform(lambda s: RNG.permutation(s.to_numpy()))
        hr2_passes += hr2(shuffled, choose(shuffled), draws=300)['verdict'] == 'HOLDS'
        hr3_passes += hr3(swap_retry(frame, chosen['own']), chosen, a1, draws=300)['verdict_hr3'] == 'HOLDS'
    return {'hr2_false_pass': hr2_passes / copies, 'hr3_false_pass': hr3_passes / copies, 'copies': copies}


def swap_retry(frame: pd.DataFrame, own: dict) -> pd.DataFrame:
    out = frame.copy()
    for model, harness in own.items():
        if harness == 'base':
            continue
        switch = (out.model == model) & (out.condition == harness) & (out.repeat == 0)
        retry = (out.model == model) & (out.condition == 'base') & (out.repeat == 1)
        flip = set(t for t in out[switch].task_id if RNG.random() < 0.5) & set(out[retry].task_id)
        a, b = switch & out.task_id.isin(flip), retry & out.task_id.isin(flip)
        out.loc[a, ['condition', 'repeat']] = ['base', 1]
        out.loc[b, ['condition', 'repeat']] = [harness, 0]
    return out


def main() -> None:
    frame = episodes()
    print(f'{len(frame)} episodes; conditions: {sorted(frame.condition.unique())}')
    chosen = choose(frame)
    results = {'chosen': chosen, 'Q': q_table(frame).to_dict('records'), 'HR1': hr1(frame)}
    results['HR2'] = hr2(frame, chosen)
    a1 = a1_detector(frame)
    results['HR3'] = hr3(frame, chosen, a1)
    results['HR4'] = hr4(frame)
    results['HR5'] = hr5(frame)
    results['calibration'] = calibration(frame, chosen, a1, copies=int(sys.argv[1]) if len(sys.argv) > 1 else 50)
    (ROOT / 'data/results/prereg-stage2.json').write_text(json.dumps(results, indent=1, default=float))
    print('chosen', chosen['single'], chosen['own'])
    for key in ('HR1', 'HR2', 'HR4', 'HR5'):
        print(key, results[key]['verdict'])
    print('HR3', results['HR3'].get('verdict_hr3'), 'calibration', results['calibration'])


if __name__ == '__main__':
    main()
