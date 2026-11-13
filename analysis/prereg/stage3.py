"""Phase L (PREREG-L.md) and stage 3 (PREREG-PD.md) tests → data/results/prereg-stage3.json.

    uv run --with pandas --with numpy --with scikit-learn python -m analysis.prereg.stage3

Files are named, not globbed: data/runs also holds smokes, parser-sensitivity copies (`-L-v4`) and two delegation
runs per executor that share the condition name (`-pd` plans with qwen3.8, `-pdg` with Gemma).
"""

import json

import numpy as np
import pandas as pd

from analysis.prereg.analyze import RNG, macro_diff
from analysis.prereg.load import ROOT, RUNS, SIZE

PHASE_L = ['lfm2.5-230m', 'lfm2.5-350m', 'lfm2.5-1.2b', 'lfm2.5-2.6b', 'lfm2.5-8b-a1b', 'gemma-4-e2b', 'gemma-4-e4b']
GATEWAY = ['qwen3.8', 'gemma-4-26b-a4b', 'qwen3-vl-30b']
ACTIVE = {'lfm2.5-230m': 0.23, 'lfm2.5-350m': 0.35, 'lfm2.5-1.2b': 1.2, 'lfm2.5-2.6b': 2.69, 'lfm2.5-8b-a1b': 1.5,
          'gemma-4-e2b': 2.3, 'gemma-4-e4b': 4.5, 'qwen3.5-2b': 2.0, 'qwen3.5-4b': 4.2, 'qwen3.8': 27.0}
SOLO_FILES = ['', '-memory', '-hr']  # stage 1–2 runs of a model, next to its name
DECOMPOSABLE = ['multiturn', 'shell']
SOLO_TRACKS = ['code', 'tools', 'shell']
GEMMA_PLANS = 'delegate@gemma'
BUDGETS_L3 = np.geomspace(4e12, 1.6e14, 16)
DRAWS = 10_000


def read(path, **override) -> list[dict]:
    rows = []
    for line in open(path):
        e = json.loads(line)
        rows.append({'model': e['model'], 'condition': override.get('condition', e['condition']), 'repeat': e['repeat'],
                     'task_id': e['task_id'], 'track': e['track'], 'split': e['split'], 'passed': bool(e['passed']),
                     'flops': e['flops'], 'planner_flops': e.get('planner_flops', 0.0)})
    return rows


def frame() -> pd.DataFrame:
    rows = []
    for m in SIZE:
        rows += [r for suffix in SOLO_FILES if (RUNS / m / f'{m}{suffix}.jsonl').exists() for r in read(RUNS / m / f'{m}{suffix}.jsonl')]
    for m in PHASE_L:
        if (RUNS / m / f'{m}-L.jsonl').exists():
            rows += read(RUNS / m / f'{m}-L.jsonl')
    for m in list(SIZE) + PHASE_L:
        for suffix, override in (('-mt', {}), ('-pd', {}), ('-pdg', {'condition': GEMMA_PLANS})):
            if (RUNS / m / f'{m}{suffix}.jsonl').exists():
                rows += read(RUNS / m / f'{m}{suffix}.jsonl', **override)
    out = pd.DataFrame(rows)
    return out[out.repeat == 0].drop_duplicates(['model', 'condition', 'task_id'], keep='last').reset_index(drop=True)


def diff(data: pd.DataFrame, a: tuple, b: tuple, tracks: list[str], split: str | None = 'test') -> list:
    """Macro over `tracks` of pass(a) − pass(b), paired by task, with a 95% within-track task bootstrap."""

    part = data[data.track.isin(tracks) & ((data.split == split) if split else True)]
    x = part[(part.model == a[0]) & (part.condition == a[1])].set_index('task_id')
    y = part[(part.model == b[0]) & (part.condition == b[1])].set_index('task_id')
    common = x.index.intersection(y.index)
    d = pd.DataFrame({'d': x.loc[common, 'passed'].astype(float) - y.loc[common, 'passed'].astype(float),
                      'track': x.loc[common, 'track']})
    return list(macro_diff(d))


def q(data: pd.DataFrame, model: str, condition: str, tracks: list[str], part: str = 'test') -> float:
    """Macro pass rate on `part`: test, or fit (screen + train)."""

    split = (data.split == 'test') if part == 'test' else (data.split != 'test')
    rows = data[(data.model == model) & (data.condition == condition) & data.track.isin(tracks) & split]
    return float(rows.groupby('track').passed.mean().mean())


# phase L -------------------------------------------------------------------------------------

def phase_l(data: pd.DataFrame) -> list[str]:
    """The phase-L models whose run is in: verdicts before all seven are in are previews."""

    return [m for m in PHASE_L if (data[data.model == m].condition == 'skill_inline').any()]


def lifts(data: pd.DataFrame, condition: str, models: list[str]) -> dict:
    return {m: diff(data, (m, condition), (m, 'base'), SOLO_TRACKS, split=None) for m in models}


def lift_matrix(data: pd.DataFrame, condition: str, models: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    part = data[data.track.isin(SOLO_TRACKS)]
    cols = {}
    for m in models:
        x = part[(part.model == m) & (part.condition == condition)].set_index('task_id').passed.astype(float)
        y = part[(part.model == m) & (part.condition == 'base')].set_index('task_id').passed.astype(float)
        cols[m] = x - y
    matrix = pd.DataFrame(cols).dropna()
    return matrix, part.drop_duplicates('task_id').set_index('task_id').track.reindex(matrix.index)


def l1(data: pd.DataFrame) -> dict:
    per = lifts(data, 'memory', phase_l(data))
    wins = sum(v[1] > 0 for v in per.values())
    return {'per_model': per, 'wins': wins, 'of': len(per), 'verdict': 'HOLDS' if wins >= 5 else 'NOT SHOWN'}


def l2(data: pd.DataFrame) -> dict:
    models = phase_l(data)
    per = lifts(data, 'skill_inline', models)
    wins = sum(v[1] > 0 for v in per.values())
    matrix, tracks = lift_matrix(data, 'skill_inline', models + GATEWAY)
    groups = [matrix[tracks == t].to_numpy() for t in SOLO_TRACKS]
    point = np.mean([g.mean(0) for g in groups], 0)
    draws = np.mean([g[RNG.integers(0, len(g), (DRAWS, len(g)))].mean(1) for g in groups], 0)
    small, big = slice(0, len(models)), slice(len(models), None)
    d = draws[:, small].mean(1) - draws[:, big].mean(1)
    gap = [float(point[small].mean() - point[big].mean()), *np.quantile(d, [0.025, 0.975]).tolist()]
    return {'per_model': per, 'wins': wins, 'of': len(models), 'mean_new': float(point[small].mean()), 'mean_gateway': float(point[big].mean()),
            'gap': gap, 'verdict': 'HOLDS' if wins <= 2 and gap[2] < 0 else 'NOT SHOWN'}


def costs(data: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    part = data[data.model.isin(models) & data.track.isin(SOLO_TRACKS)].assign(
        part=lambda f: np.where(f.split == 'test', 'test', 'fit'))
    qs = part.groupby(['model', 'condition', 'part', 'track']).passed.mean().groupby(['model', 'condition', 'part']).mean()
    return pd.DataFrame({'q': qs, 'flops': part.groupby(['model', 'condition', 'part']).flops.mean()}).reset_index()


def ladder(model: str, cost: dict) -> list[str]:
    """PREREG-L's L3 rule, best rung first."""

    if model in GATEWAY:
        return ['skill_inline+memory', 'skill', 'base']
    return ['memory', 'skill' if cost.get('skill', np.inf) < cost.get('base', np.inf) else 'base']


def l3(data: pd.DataFrame) -> dict:
    models = phase_l(data) + GATEWAY
    table = costs(data, models)
    fit = table[(table.part == 'fit') & table.condition.isin(['base', 'memory', 'skill', 'skill_inline', 'skill_inline+memory'])]
    cost = {m: dict(zip(g.condition, g.flops)) for m, g in fit.groupby('model')}
    out = {'budgets': {}}
    for budget in BUDGETS_L3:
        rule = []
        for m in models:
            rung = next((h for h in ladder(m, cost[m]) if cost[m].get(h, np.inf) <= budget), None)
            if rung:
                rule.append(fit[(fit.model == m) & (fit.condition == rung)])
        ok = fit[fit.flops <= budget]
        if not rule or ok.empty:
            continue
        rule = pd.concat(rule)
        pick = rule.loc[rule.q.idxmax(), ['model', 'condition']].tolist()
        search = ok.loc[ok.q.idxmax(), ['model', 'condition']].tolist()
        d = [0.0, 0.0, 0.0] if pick == search else diff(data, tuple(pick), tuple(search), SOLO_TRACKS)
        out['budgets'][f'{budget:.2e}'] = {'rule': pick, 'search': search, 'diff': d}
    diffs = [r['diff'] for r in out['budgets'].values()]
    out['mean_diff'] = float(np.mean([d[0] for d in diffs]))
    out['same_pair'] = float(np.mean([r['rule'] == r['search'] for r in out['budgets'].values()]))
    out['verdict'] = 'HOLDS' if all(d[1] > -0.05 for d in diffs) and out['mean_diff'] >= -0.02 else 'NOT SHOWN'
    return out


def l4(data: pd.DataFrame) -> dict:
    """FLOPs per task of skill and skill_inline against base, paired by task, with a 95% bootstrap CI."""

    out = {}
    part = data[data.track.isin(SOLO_TRACKS)]
    for m in phase_l(data):
        base = part[(part.model == m) & (part.condition == 'base')].set_index('task_id').flops
        out[m] = {'base': float(base.mean())}
        for h in ('skill', 'skill_inline'):
            x = part[(part.model == m) & (part.condition == h)].set_index('task_id').flops
            d = (x - base).dropna().to_numpy()
            draws = d[RNG.integers(0, len(d), (DRAWS, len(d)))].mean(1)
            out[m][h] = [float(d.mean()), *np.quantile(draws, [0.025, 0.975]).tolist()]
    return out


# stage 3 -------------------------------------------------------------------------------------

def executors(data: pd.DataFrame) -> list[str]:
    return sorted(set(data[data.condition == 'delegate'].model) - {'qwen3.8'}, key=lambda m: ACTIVE[m])


def pd1(data: pd.DataFrame) -> dict:
    small = [m for m in executors(data) if ACTIVE[m] <= 4]
    best = max(small, key=lambda m: q(data, m, 'delegate', DECOMPOSABLE, 'fit'))
    d = diff(data, (best, 'delegate'), ('qwen3.8', 'delegate'), DECOMPOSABLE)
    return {'best_on_fit': best, 'diff': d, 'verdict': 'HOLDS' if d[1] > -0.05 else 'NOT SHOWN'}


def pd3(data: pd.DataFrame) -> dict:
    per = {m: diff(data, (m, 'delegate'), ('qwen3.8', 'delegate'), DECOMPOSABLE) for m in executors(data)}
    passing = [m for m in executors(data) if per[m][1] > -0.05]
    return {'per_executor': per, 'smallest': passing[0] if passing else None,
            'active_b': ACTIVE[passing[0]] if passing else None,
            'prediction_met': bool(passing) and ACTIVE[passing[0]] <= 1.5}


def pd4(data: pd.DataFrame) -> dict:
    """|Q(qwen3.8 plans) − Q(Gemma plans)| − |Q(4B executes) − Q(1.2B executes)|, each averaged over the other factor."""

    cells = {(e, c): data[(data.model == e) & (data.condition == c) & data.track.isin(DECOMPOSABLE) & (data.split == 'test')]
             .set_index('task_id') for e in ('qwen3.5-4b', 'lfm2.5-1.2b') for c in ('delegate', GEMMA_PLANS)}
    common = sorted(set.intersection(*(set(c.index) for c in cells.values())))
    tracks = cells[('qwen3.5-4b', 'delegate')].loc[common, 'track']
    passed = {k: v.loc[common, 'passed'].astype(float).to_numpy() for k, v in cells.items()}

    def stat(idx_by_track):
        mean = {k: np.mean([v[i].mean(-1) for i in idx_by_track], 0) for k, v in passed.items()}
        planner = (mean[('qwen3.5-4b', 'delegate')] + mean[('lfm2.5-1.2b', 'delegate')]
                   - mean[('qwen3.5-4b', GEMMA_PLANS)] - mean[('lfm2.5-1.2b', GEMMA_PLANS)]) / 2
        executor = (mean[('qwen3.5-4b', 'delegate')] + mean[('qwen3.5-4b', GEMMA_PLANS)]
                    - mean[('lfm2.5-1.2b', 'delegate')] - mean[('lfm2.5-1.2b', GEMMA_PLANS)]) / 2
        return np.abs(planner) - np.abs(executor), planner, executor

    positions = [np.flatnonzero(tracks.to_numpy() == t) for t in DECOMPOSABLE]
    point = [float(x) for x in stat(positions)]
    draws = stat([p[RNG.integers(0, len(p), (DRAWS, len(p)))] for p in positions])[0]
    ci = [point[0], *np.quantile(draws, [0.025, 0.975]).tolist()]
    return {'diff': ci, 'planner_effect': point[1], 'executor_effect': point[2], 'verdict': 'HOLDS' if ci[1] > 0 else 'NOT SHOWN'}


def pd5(data: pd.DataFrame) -> dict:
    per = {m: diff(data, (m, 'delegate'), (m, 'delegate+noex'), DECOMPOSABLE) for m in ('lfm2.5-1.2b', 'qwen3.5-4b')}
    loop = {m: diff(data, (m, 'delegate'), (m, 'delegate_open'), DECOMPOSABLE) for m in ('lfm2.5-1.2b', 'qwen3.5-4b')}
    return {'examples': per, 'closed_vs_open': loop, 'verdict': 'HOLDS' if all(v[1] > 0 for v in per.values()) else 'NOT SHOWN'}


def pairs(data: pd.DataFrame, part: str) -> pd.DataFrame:
    """Macro Q over the decomposable tracks and mean FLOPs, per (model, condition) present on both tracks."""

    sel = data[data.track.isin(DECOMPOSABLE) & ((data.split == 'test') if part == 'test' else (data.split != 'test'))]
    g = sel.groupby(['model', 'condition'])
    out = pd.DataFrame({'q': sel.groupby(['model', 'condition', 'track']).passed.mean().groupby(['model', 'condition']).mean(),
                        'flops': g.flops.mean(), 'tracks': g.track.nunique()}).reset_index()
    return out[out.tracks == len(DECOMPOSABLE)]


def pd2(data: pd.DataFrame) -> dict:
    fit, test = pairs(data, 'fit'), pairs(data, 'test')
    delegated = fit[(fit.condition == 'delegate') & (fit.model != 'qwen3.8')]  # P→S: P→B is not a small executor
    single = fit[fit.condition.isin(['base', 'memory'])]
    lo, hi = fit.flops.min(), fit.flops.max()
    out = {'budgets': {}}
    for budget in np.geomspace(lo, hi, 16):
        a, b = delegated[delegated.flops <= budget], single[single.flops <= budget]
        if a.empty or b.empty:
            continue
        pa = a.loc[a.q.idxmax(), ['model', 'condition']].tolist()
        pb = b.loc[b.q.idxmax(), ['model', 'condition']].tolist()
        out['budgets'][f'{budget:.2e}'] = {'delegated': pa, 'single': pb, 'diff': diff(data, tuple(pa), tuple(pb), DECOMPOSABLE)}
    out['frontier'] = {'test': test.sort_values('flops').to_dict('records')}
    out['verdict'] = 'HOLDS' if any(r['diff'][0] >= 0.02 and r['diff'][1] > 0 for r in out['budgets'].values()) else 'NOT SHOWN'
    return out


def main() -> None:
    data = frame()
    print(f'{len(data)} episodes, {data.model.nunique()} models')
    results = {'L1': l1(data), 'L2': l2(data), 'L3': l3(data), 'L4': l4(data),
               'PD1': pd1(data), 'PD2': pd2(data), 'PD3': pd3(data), 'PD4': pd4(data), 'PD5': pd5(data)}
    (ROOT / 'data/results/prereg-stage3.json').write_text(json.dumps(results, indent=1, default=float))
    for key, value in results.items():
        print(key, value.get('verdict', ''), {k: v for k, v in value.items() if k in ('wins', 'gap', 'mean_diff', 'same_pair', 'best_on_fit', 'diff', 'smallest', 'active_b')})


if __name__ == '__main__':
    main()
