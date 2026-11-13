"""All pre-registered tests (PREREG.md) → data/results/prereg-stage1.json.

    uv run --with pandas --with numpy --with scikit-learn python -m analysis.prereg.analyze
"""

import json
import math
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from analysis.prereg.load import BIG, ROOT, SIZE, SMALL, TRACKS, episodes, step_features

warnings.filterwarnings('ignore')
RNG = np.random.default_rng(20261108)
B = 10_000
LEVERS = ['skill', 'watchdog', 'memory']
BINS = [('≤0.6B', 0, 0.8), ('0.8–2.3B', 0.8, 2.5), ('3–5B', 2.5, 6), ('8–10B', 6, 12), ('≥25B', 12, 99)]


def wilson(k: float, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - spread) / (1 + z * z / n), (centre + spread) / (1 + z * z / n)


def boot_mean(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    values = np.asarray(values, float)
    if len(values) == 0:
        return float('nan'), float('nan'), float('nan')
    idx = RNG.integers(0, len(values), (B, len(values)))
    means = values[idx].mean(1)
    return float(values.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def macro_diff(frame: pd.DataFrame, alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean over tracks of the paired per-task difference column `d`, bootstrapped within track."""

    groups = [g['d'].to_numpy(float) for _, g in frame.groupby('track')]
    if not groups:
        return float('nan'), float('nan'), float('nan')
    point = np.mean([g.mean() for g in groups])
    draws = np.zeros(B)
    for g in groups:
        draws += g[RNG.integers(0, len(g), (B, len(g)))].mean(1)
    draws /= len(groups)
    return float(point), float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def pivot(frame: pd.DataFrame, condition: str, repeat: int = 0) -> pd.DataFrame:
    part = frame[(frame.condition == condition) & (frame.repeat == repeat)]
    return part.set_index(['model', 'task_id'])


# ---------------------------------------------------------------------------------------------

def quality(frame: pd.DataFrame) -> dict:
    out = {}
    for (model, condition, repeat), g in frame.groupby(['model', 'condition', 'repeat']):
        tracks = {}
        for track, t in g.groupby('track'):
            k, n = int(t.passed.sum()), len(t)
            lo, hi = wilson(k, n)
            tracks[track] = {'q': k / n, 'lo': lo, 'hi': hi, 'n': n,
                             'flops': float(t.flops.mean()), 'seconds': float(t.seconds.mean()),
                             'tokens': float(t.tokens.mean())}
        macro = float(np.mean([v['q'] for v in tracks.values()]))
        # tracks are independent samples: normal-approximation CI of their mean
        se = math.sqrt(sum(v['q'] * (1 - v['q']) / v['n'] for v in tracks.values())) / len(tracks)
        failures = Counter(g.loc[~g.passed, 'failure'])
        out[f'{model}|{condition}|{repeat}'] = {
            'model': model, 'condition': condition, 'repeat': int(repeat), 'size': SIZE.get(model),
            'macro': macro, 'macro_lo': max(0.0, macro - 1.96 * se), 'macro_hi': min(1.0, macro + 1.96 * se), 'tracks': tracks, 'n': len(g), 'flops': float(g.flops.mean()),
            'failures': dict(failures), 'skill_read': float(g.skill_read.mean()) if condition == 'skill' else None}
    return out


def paired(frame: pd.DataFrame, model: str, a: str, b: str, ra: int = 0, rb: int = 0, column: str = 'passed'):
    x = frame[(frame.model == model) & (frame.condition == a) & (frame.repeat == ra)].set_index('task_id')
    y = frame[(frame.model == model) & (frame.condition == b) & (frame.repeat == rb)].set_index('task_id')
    common = x.index.intersection(y.index)
    if len(common) == 0:
        return None
    return pd.DataFrame({'d': x.loc[common, column].astype(float).to_numpy() - y.loc[common, column].astype(float).to_numpy(),
                         'track': x.loc[common, 'track'].to_numpy()}, index=common)


def a4_skill_curve(frame: pd.DataFrame) -> dict:
    per_model = {}
    for model in SIZE:
        for lever in LEVERS:
            d = paired(frame, model, lever, 'base')
            if d is None:
                continue
            point, lo, hi = macro_diff(d)
            per_model.setdefault(model, {})[lever] = {'lift': point, 'lo': lo, 'hi': hi, 'n': len(d)}
        noise = paired(frame, model, 'base', 'base', 1, 0)
        if noise is not None:
            per_model.setdefault(model, {})['noise'] = dict(zip(('lift', 'lo', 'hi'), macro_diff(noise)))
    bins = {}
    for label, lo_size, hi_size in BINS:
        models = [m for m, s in SIZE.items() if lo_size <= s < hi_size and m != 'qwen3.5-0.8b-wsp']
        parts = [paired(frame, m, 'skill', 'base') for m in models]
        parts = [p for p in parts if p is not None]
        if parts:
            d = pd.concat(parts)
            bins[label] = dict(zip(('lift', 'lo', 'hi'), macro_diff(d))) | {'models': models, 'n': len(d), '_d': d}
    verdict = 'NOT SHOWN'
    if len(bins) >= 3:
        labels = list(bins)
        peak = max(labels[1:-1], key=lambda l: bins[l]['lift']) if len(labels) > 2 else None
        if peak:
            ok = True
            for edge in (labels[0], labels[-1]):
                diff = boot_diff(bins[peak]['_d']['d'].to_numpy(), bins[edge]['_d']['d'].to_numpy())
                bins[peak][f'vs_{edge}'] = diff
                ok &= diff[1] > 0
            verdict = 'HOLDS' if ok else 'NOT SHOWN'
            bins['_peak'] = peak
    for v in bins.values():
        if isinstance(v, dict):
            v.pop('_d', None)
    return {'per_model': per_model, 'bins': bins, 'verdict': verdict}


def boot_diff(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    da = a[RNG.integers(0, len(a), (B, len(a)))].mean(1)
    db = b[RNG.integers(0, len(b), (B, len(b)))].mean(1)
    diff = da - db
    return float(a.mean() - b.mean()), float(np.quantile(diff, alpha / 2)), float(np.quantile(diff, 1 - alpha / 2))


def a2_recoverable(frame: pd.DataFrame) -> dict:
    """Among base failures of a small model: the share some lever solves, net of resampling."""

    records = []
    for model in SMALL:
        base = frame[(frame.model == model) & (frame.condition == 'base') & (frame.repeat == 0)].set_index('task_id')
        failed = base.index[~base.passed]
        if len(failed) == 0:
            continue
        rows = pd.DataFrame(index=failed)
        rows['track'] = base.loc[failed, 'track']
        rows['kind'] = base.loc[failed, 'failure']
        levers = list(LEVERS) + (['wsp'] if model == 'qwen3.5-0.8b' else [])
        for lever in levers + ['resample']:
            if lever == 'wsp':
                other = frame[(frame.model == 'qwen3.5-0.8b-wsp') & (frame.condition == 'base') & (frame.repeat == 0)]
            elif lever == 'resample':
                other = frame[(frame.model == model) & (frame.condition == 'base') & (frame.repeat == 1)]
            else:
                other = frame[(frame.model == model) & (frame.condition == lever) & (frame.repeat == 0)]
            other = other.set_index('task_id')['passed']
            rows[lever] = other.reindex(failed).astype(float)
        rows['model'] = model
        records.append(rows.reset_index())
    if not records:
        return {'verdict': 'NOT SHOWN', 'note': 'no data'}
    data = pd.concat(records, ignore_index=True)
    have = [l for l in LEVERS + ['wsp'] if l in data and data[l].notna().any()]
    data = data.dropna(subset=['resample'] + [l for l in LEVERS if l in have and l != 'wsp'])
    lever_cols = [l for l in have if l != 'wsp']
    data['any'] = data[lever_cols + (['wsp'] if 'wsp' in have else [])].max(axis=1, skipna=True)
    k = data[lever_cols + (['wsp'] if 'wsp' in have else [])].notna().sum(axis=1)
    out = {'n_failures': int(len(data)), 'levers': {}, 'by_kind': {}, 'by_model': {}}
    n_rate = data['resample'].mean()
    for lever in lever_cols + (['wsp'] if 'wsp' in have else []) + ['resample']:
        col = data[lever].dropna()
        point, lo, hi = boot_mean(col.to_numpy())
        out['levers'][lever] = {'share': point, 'lo': lo, 'hi': hi, 'net': point - (n_rate if lever != 'resample' else 0), 'n': int(len(col))}
    # any lever, net of k independent resamples at the observed resample rate (per failure)
    noise_any = 1 - (1 - n_rate) ** k
    net_any = data['any'].to_numpy() - noise_any.to_numpy()
    point, lo, hi = boot_mean(net_any)
    raw = boot_mean(data['any'].to_numpy())
    out['any'] = {'share': raw[0], 'share_lo': raw[1], 'share_hi': raw[2], 'net': point, 'lo': lo, 'hi': hi,
                  'noise_any': float(noise_any.mean())}
    for kind, g in data.groupby('kind'):
        out['by_kind'][kind] = {'n': int(len(g)), **{l: float(g[l].mean()) for l in lever_cols + ['resample'] + (['wsp'] if 'wsp' in have and g['wsp'].notna().any() else [])}}
    for model, g in data.groupby('model'):
        out['by_model'][model] = {'n': int(len(g)), 'any': float(g['any'].mean()), 'resample': float(g['resample'].mean())}
    out['verdict'] = 'HOLDS' if point >= 0.40 and lo > 0.30 else 'NOT SHOWN'
    return out


# ---- A1 ----------------------------------------------------------------------------------------

def feature_table(frame: pd.DataFrame, k: int, models: list[str]) -> pd.DataFrame:
    part = frame[frame.model.isin(models) & (frame.condition == 'base') & (frame.repeat == 0)]
    rows = []
    for e in part.itertuples():
        f = step_features(e.steps, k)
        f.update({'task_id': e.task_id, 'model': e.model, 'track': e.track, 'split': e.split,
                  'failed': float(not e.passed), 'log_size': math.log(SIZE[e.model]), 'ended_at_k': len(e.steps) <= k})
        rows.append(f)
    return pd.DataFrame(rows)


def a1_detector(frame: pd.DataFrame) -> dict:
    emb = np.load(ROOT / 'data/embeddings.npz')
    vectors = dict(zip(emb['ids'], emb['vectors']))
    out = {'k': {}}
    for k in (1, 2, 3):
        table = feature_table(frame, k, [m for m in SMALL if m != 'qwen3.5-0.8b-wsp'])
        if table.empty or table.split.nunique() < 2:
            continue
        harness_cols = [c for c in table.columns if c.startswith('f_')]
        track_dummies = pd.get_dummies(table.track, prefix='t').astype(float)
        model_dummies = pd.get_dummies(table.model, prefix='m').astype(float)
        context = pd.concat([track_dummies, model_dummies, table[['log_size']]], axis=1)
        X_h = pd.concat([table[harness_cols], context], axis=1).to_numpy(float)
        y = table.failed.to_numpy()
        train = (table.split != 'test').to_numpy()
        test = ~train
        # the prompt baseline: PCA of the embedding alone (fit on train), then the same context as the detector
        E = np.stack([vectors[t] for t in table.task_id])
        E = PCA(64, random_state=0).fit(E[train]).transform(E)
        X_e = np.hstack([E, context.to_numpy(float)])
        X_c = context.to_numpy(float)

        def fit(X, pca=False):
            steps = [SimpleImputer(strategy='median', add_indicator=True), StandardScaler()]
            if pca:
                steps.append(PCA(min(64, int(train.sum()) - 1, X.shape[1]), random_state=0))
            model = make_pipeline(*steps, LogisticRegression(C=0.5, max_iter=3000))
            model.fit(X[train], y[train])
            return model.predict_proba(X[test])[:, 1]

        p_h, p_e = fit(X_h), fit(X_e)
        p_c, p_x = fit(np.hstack([X_h, X_e])), fit(X_c)
        y_t, groups = y[test], table.task_id.to_numpy()[test]
        result = {'n_test': int(test.sum()), 'fail_rate': float(y_t.mean()),
                  'auc_harness': auc_ci(y_t, p_h, groups), 'auc_prompt': auc_ci(y_t, p_e, groups),
                  'auc_both': auc_ci(y_t, p_c, groups), 'auc_context': auc_ci(y_t, p_x, groups),
                  'diff': auc_diff_ci(y_t, p_h, p_e, groups)}
        # within one model: what a router serving that model sees (pooled AUC also ranks models against each other)
        result['per_model'] = {}
        models_t = table.model.to_numpy()[test]
        for name in np.unique(models_t):
            m = models_t == name
            if m.sum() > 50 and 0.03 < y_t[m].mean() < 0.97:
                result['per_model'][name] = {'n': int(m.sum()), 'fail_rate': float(y_t[m].mean()),
                                             'harness': float(roc_auc_score(y_t[m], p_h[m])),
                                             'prompt': float(roc_auc_score(y_t[m], p_e[m]))}
        within = result['per_model'].values()
        result['within_model_mean'] = {'harness': float(np.mean([v['harness'] for v in within])),
                                       'prompt': float(np.mean([v['prompt'] for v in within])),
                                       'diff': within_diff_ci(y_t, p_h, p_e, groups, models_t, list(result['per_model']))}
        result['per_track'] = {}
        tracks_t = table.track.to_numpy()[test]
        for track in TRACKS:
            m = tracks_t == track
            if m.sum() > 20 and 0 < y_t[m].mean() < 1:
                result['per_track'][track] = {'harness': float(roc_auc_score(y_t[m], p_h[m])),
                                              'prompt': float(roc_auc_score(y_t[m], p_e[m]))}
        running = ~table.ended_at_k.to_numpy()[test]
        if running.sum() > 50 and 0 < y_t[running].mean() < 1:
            result['running_at_k'] = {'n': int(running.sum()), 'harness': float(roc_auc_score(y_t[running], p_h[running])),
                                      'prompt': float(roc_auc_score(y_t[running], p_e[running]))}
        # calibration: the decision rule on label-shuffled data
        passes = 0
        for s in range(50):
            y_s = RNG.permutation(y)
            model = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(),
                                  LogisticRegression(C=0.5, max_iter=2000)).fit(X_h[train], y_s[train])
            a = roc_auc_score(y_s[test], model.predict_proba(X_h[test])[:, 1])
            passes += a >= 0.75
        result['calibration_false_pass'] = passes / 50
        coefficients = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(),
                                     LogisticRegression(C=0.5, max_iter=3000)).fit(X_h[train], y[train])[-1].coef_[0]
        names = harness_cols + list(context.columns)
        result['top_features'] = sorted(zip(names, map(float, coefficients[:len(names)])), key=lambda x: -abs(x[1]))[:10]
        out['k'][k] = result
    best = out['k'].get(3) or out['k'].get(2) or out['k'].get(1)
    if best:
        h, d = best['auc_harness'], best['diff']
        out['verdict'] = 'HOLDS' if h[0] >= 0.75 and d[1] > 0.10 else ('REVERSED' if d[2] < 0 else 'NOT SHOWN')
        w = best['within_model_mean']
        # the router's case (one model): the same thresholds on the within-model mean
        out['verdict_within'] = 'HOLDS' if w['harness'] >= 0.75 and w['diff'][1] > 0.10 else 'NOT SHOWN'
    return out


def within_diff_ci(y, p1, p2, groups, models, names, alpha=0.05, draws=300):
    """Mean over models of AUC(p1) − AUC(p2) within each model, task-bootstrap CI."""

    def stat(pick):
        values = []
        for name in names:
            m = pick[models[pick] == name]
            if 0 < y[m].mean() < 1:
                values.append(roc_auc_score(y[m], p1[m]) - roc_auc_score(y[m], p2[m]))
        return float(np.mean(values))

    unique = np.unique(groups)
    index = {g: np.where(groups == g)[0] for g in unique}
    boot = [stat(np.concatenate([index[g] for g in RNG.choice(unique, len(unique))])) for _ in range(draws)]
    return stat(np.arange(len(y))), float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def auc_ci(y, p, groups, alpha=0.05):
    point = roc_auc_score(y, p)
    unique = np.unique(groups)
    index = {g: np.where(groups == g)[0] for g in unique}
    draws = []
    for _ in range(1000):
        pick = np.concatenate([index[g] for g in RNG.choice(unique, len(unique))])
        if 0 < y[pick].mean() < 1:
            draws.append(roc_auc_score(y[pick], p[pick]))
    return float(point), float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def auc_diff_ci(y, p1, p2, groups, alpha=0.05):
    point = roc_auc_score(y, p1) - roc_auc_score(y, p2)
    unique = np.unique(groups)
    index = {g: np.where(groups == g)[0] for g in unique}
    draws = []
    for _ in range(1000):
        pick = np.concatenate([index[g] for g in RNG.choice(unique, len(unique))])
        if 0 < y[pick].mean() < 1:
            draws.append(roc_auc_score(y[pick], p1[pick]) - roc_auc_score(y[pick], p2[pick]))
    return float(point), float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


# ---- A5 ----------------------------------------------------------------------------------------

def a5_wsp(frame: pd.DataFrame) -> dict:
    out = {}
    for condition in ('base', 'skill', 'watchdog'):
        a = frame[(frame.model == 'qwen3.5-0.8b-wsp') & (frame.condition == condition) & (frame.repeat == 0)].set_index('task_id')
        b = frame[(frame.model == 'qwen3.5-0.8b') & (frame.condition == condition) & (frame.repeat == 0)].set_index('task_id')
        common = a.index.intersection(b.index)
        if len(common) == 0:
            continue
        dq = pd.DataFrame({'d': a.loc[common, 'passed'].astype(float).to_numpy() - b.loc[common, 'passed'].astype(float).to_numpy(),
                           'track': a.loc[common, 'track'].to_numpy()})
        dl = pd.DataFrame({'d': (a.loc[common, 'failure'] == 'loop').astype(float).to_numpy() - (b.loc[common, 'failure'] == 'loop').astype(float).to_numpy(),
                           'track': a.loc[common, 'track'].to_numpy()})
        dt = pd.DataFrame({'d': np.log1p(a.loc[common, 'tokens'].to_numpy(float)) - np.log1p(b.loc[common, 'tokens'].to_numpy(float)),
                           'track': a.loc[common, 'track'].to_numpy()})
        out[condition] = {'dq': macro_diff(dq), 'dloop': macro_diff(dl), 'dlogtokens': macro_diff(dt), 'n': int(len(common)),
                          'loop_wsp': float((a.loc[common, 'failure'] == 'loop').mean()),
                          'loop_base': float((b.loc[common, 'failure'] == 'loop').mean()),
                          'cut_wsp': float(a.loc[common, 'failure'].isin(['loop', 'model_stopped']).mean()),
                          'cut_base': float(b.loc[common, 'failure'].isin(['loop', 'model_stopped']).mean())}
    if 'base' in out:
        dq, dl = out['base']['dq'], out['base']['dloop']
        out['verdict'] = 'HOLDS' if dq[1] > 0 and dl[2] < 0 else ('REVERSED' if dq[2] < 0 else 'NOT SHOWN')
    return out


# ---- A3 (offline cascades on recorded episodes) -----------------------------------------------

def a3_policies(frame: pd.DataFrame, a1: dict, switch: dict[str, str] | None = None, draws: int = 2000) -> dict:
    """Restart cascades on the test split, cost in FLOPs, quality macro over tracks.

    escalate:  run s; if the detector at step k flags it, run the big model from scratch.
    random:    the same, with the flag drawn at random at the same rate.
    levers:    run s; if flagged, rerun s with the lever the train split says is best for the
               predicted failure kind; if that run is flagged again, run the big model.
    switch:    (stage 2) like retry, but the second run uses the model's own harness `switch[model]`.
    Cost of a flagged run counts its first k steps (it is stopped there)."""

    k = 2
    small_models = [m for m in ('qwen3.5-2b', 'qwen3.5-4b', 'qwen3.5-9b', 'granite-4.0-micro', 'qwen3-vl-8b') if m in set(frame.model)]
    big = next((m for m in ('gemma-4-26b-a4b', 'qwen3.8') if m in set(frame.model)), None)
    if not small_models or big is None:
        return {'note': 'not enough data'}
    table_all = []
    runs = [(None, 'base', 0, 'base'), (None, 'base', 1, 'resample')] + [(None, l, 0, l) for l in LEVERS]
    runs += [(m, h, 0, 'switch') for m, h in (switch or {}).items() if m in small_models]
    for only, condition, repeat, label in runs:
        part = frame[frame.model.isin([only] if only else small_models) & (frame.condition == condition) & (frame.repeat == repeat)]
        for e in part.itertuples():
            f = step_features(e.steps, k)
            flops_k = e.flops * (min(k, len(e.steps)) / max(1, len(e.steps)))
            f.update({'task_id': e.task_id, 'model': e.model, 'track': e.track, 'split': e.split,
                      'condition': label, 'failed': float(not e.passed), 'passed': float(e.passed),
                      'flops': e.flops, 'flops_k': flops_k, 'kind': e.failure, 'log_size': math.log(SIZE[e.model])})
            table_all.append(f)
    T = pd.DataFrame(table_all)
    cols = [c for c in T.columns if c.startswith('f_')]
    ctx = pd.concat([pd.get_dummies(T.track, prefix='t').astype(float), T[['log_size']]], axis=1)
    X = pd.concat([T[cols], ctx], axis=1).to_numpy(float)
    train = (T.split != 'test').to_numpy() & (T.condition == 'base').to_numpy()
    detector = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(),
                             LogisticRegression(C=0.5, max_iter=3000)).fit(X[train], T.failed.to_numpy()[train])
    T['risk'] = detector.predict_proba(X)[:, 1]

    # lever map from train: failure kind of base → lever with the best recovery among base failures
    a2_train = {}
    base_train = T[(T.condition == 'base') & (T.split != 'test')]
    for kind, g in base_train[base_train.failed == 1].groupby('kind'):
        best, best_rate = None, -1
        for lever in LEVERS:
            other = T[(T.condition == lever) & (T.split != 'test')].set_index(['model', 'task_id'])['passed']
            rates = other.reindex(list(zip(g.model, g.task_id))).dropna()
            if len(rates) >= 5 and rates.mean() > best_rate:
                best, best_rate = lever, rates.mean()
        a2_train[kind] = {'lever': best, 'rate': float(best_rate)}
    # a kind predictor on train (multinomial) so the policy does not peek at the true kind
    fails = (T.condition == 'base').to_numpy() & (T.failed == 1).to_numpy() & (T.split != 'test').to_numpy()
    kinds = T.kind.to_numpy()
    kind_model = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(),
                               LogisticRegression(C=0.5, max_iter=3000)).fit(X[fails], kinds[fails])
    T['kind_hat'] = kind_model.predict(X)

    out = {'k': k, 'lever_map': a2_train, 'by_big': {}}
    for big in [m for m in ('gemma-4-26b-a4b', 'qwen3.8') if m in set(frame.model)]:
        curves = cascades(frame, T, big, small_models, a2_train, draws)
        out['by_big'][big] = {'curves': curves, 'verdict': a3_verdict(curves)}
    verdicts = [v['verdict'] for v in out['by_big'].values()]
    out['verdict'] = 'HOLDS' if 'HOLDS' in verdicts else ('REVERSED' if 'REVERSED' in verdicts else 'NOT SHOWN')
    return out


def cascades(frame, T, big, small_models, lever_map, draws=2000) -> dict:
    """Policy curves per small model. The threshold of each policy is chosen on the fit split (screen + train) and
    scored on test: choosing it on test let the rule pass 48% of shuffled copies (PREREG-HR calibration)."""

    bigs = frame[(frame.model == big) & (frame.condition == 'base') & (frame.repeat == 0)].set_index('task_id')
    curves = {}
    for model in small_models:
        parts = {}
        for part, sel in (('fit', T.split != 'test'), ('test', T.split == 'test')):
            mine = T[(T.model == model) & sel]
            base = mine[mine.condition == 'base'].set_index('task_id')
            levers = {l: mine[mine.condition == l].set_index('task_id') for l in LEVERS + ['resample', 'switch']}
            tasks = [t for t in base.index if t in bigs.index and all(t in levers[l].index for l in LEVERS if not levers[l].empty)]
            parts[part] = (base.loc[tasks], bigs.loc[tasks], levers, tasks)
        if len(parts['test'][3]) < 50 or len(parts['fit'][3]) < 20:
            continue
        taus = np.quantile(parts['fit'][0].risk, np.linspace(0, 1, 21))
        sims = {part: simulate(*parts[part], taus, lever_map) for part in parts}
        b, g = parts['test'][0], parts['test'][1]
        macro = sims['test']['macro']
        points = {'small': (float(b.flops.mean()), macro(b.passed.to_numpy())),
                  'big': (float(g.flops.mean()), macro(g.passed.to_numpy().astype(float)))}
        fit_b, fit_g = parts['fit'][0], parts['fit'][1]
        fit_points = {'small': (float(fit_b.flops.mean()), None), 'big': (float(fit_g.flops.mean()), None)}
        policies = {name: [(float(c.mean()), macro(q), float(f)) for q, c, f in rs] for name, rs in sims['test']['runs'].items()}
        oracle_q = macro(np.maximum(b.passed.to_numpy(), g.passed.to_numpy().astype(float)))
        curves[model] = {'points': points, 'policies': policies, 'oracle': oracle_q, 'n': len(parts['test'][3]),
                         'matched': matched(sims['fit'], sims['test'], fit_points, len(small_models), draws)}
    return curves


def simulate(b, g, levers, tasks, taus, lever_map) -> dict:
    """Per threshold: (per-task quality, per-task cost, flag rate) of each policy on one split."""

    tracks = b.track.to_numpy()
    g_pass, g_flops = g.passed.to_numpy().astype(float), g.flops.to_numpy()

    def macro(q, pick=None):
        pick = np.arange(len(q)) if pick is None else pick
        return float(np.mean([q[pick][tracks[pick] == t].mean() for t in TRACKS if (tracks[pick] == t).any()]))

    runs = {'escalate': [], 'random': [], 'levers': [], 'retry': []} | ({'switch': []} if not levers['switch'].empty else {})
    for tau in taus:
        flag = (b.risk >= tau).to_numpy()
        runs['escalate'].append((np.where(flag, g_pass, b.passed), np.where(flag, b.flops_k + g_flops, b.flops), flag.mean()))
        rnd = RNG.random(len(b)) < flag.mean()
        runs['random'].append((np.where(rnd, g_pass, b.passed), np.where(rnd, b.flops_k + g_flops, b.flops), rnd.mean()))
        for name in [p for p in ('levers', 'retry', 'switch') if p in runs]:
            q, c = b.passed.to_numpy().astype(float), b.flops.to_numpy().copy()
            for i, task in enumerate(tasks):
                if not flag[i]:
                    continue
                lever = {'retry': 'resample', 'switch': 'switch'}.get(name) or lever_map.get(b.kind_hat.iloc[i], {}).get('lever') or 'watchdog'
                L = levers[lever]
                if L.empty or task not in L.index:
                    q[i], c[i] = g_pass[i], b.flops_k.iloc[i] + g_flops[i]
                    continue
                second = L.loc[task]
                if second.risk >= tau:
                    q[i], c[i] = g_pass[i], b.flops_k.iloc[i] + second.flops_k + g_flops[i]
                else:
                    q[i], c[i] = second.passed, b.flops_k.iloc[i] + second.flops
            runs[name].append((q, c, flag.mean()))
    return {'runs': runs, 'macro': macro, 'tracks': tracks}


def matched(fit: dict, test: dict, fit_points: dict, n_models: int, draws: int = 2000) -> dict:
    """Best policy point within a budget, chosen on fit; a − b on test with a task-bootstrap CI and a paired
    permutation p (per task the two policies' outcomes swap with p = 1/2).

    Budgets: 25/50/100% of the big model's cost (pre-registered), and — because a MoE big model can be cheaper
    than a dense small one — the small model's own cost × 1.25 / 1.5 / 2, all on fit. CI: 99%, Bonferroni over models."""

    alpha = 0.01 / max(1, n_models)
    budgets = {f'big×{s}': fit_points['big'][0] * s for s in (0.25, 0.5, 1.0)}
    budgets.update({f'small×{s}': fit_points['small'][0] * s for s in (1.25, 1.5, 2.0)})
    macro, tracks = test['macro'], test['tracks']
    out = {}
    for label, budget in budgets.items():
        row, chosen = {'budget': float(budget)}, {}
        for name, rs in fit['runs'].items():
            feasible = [(fit['macro'](q), j) for j, (q, c, _) in enumerate(rs) if c.mean() <= budget]
            chosen[name] = max(feasible)[1] if feasible else None
            if chosen[name] is not None:
                q, c, _ = test['runs'][name][chosen[name]]
                row[name], row[f'{name}_cost'] = macro(q), float(c.mean())
            else:
                row[name] = None
        for key, (a, b) in {'diff': ('levers', 'escalate'), 'diff_retry': ('levers', 'retry'), 'diff_switch': ('switch', 'retry')}.items():
            if chosen.get(a) is not None and chosen.get(b) is not None:
                # what to retry with is compared at one threshold (retry's): at different thresholds the two policies
                # escalate different shares to the big model, and that difference, not the retry, decided the test
                j = chosen[b] if b == 'retry' else chosen[a]
                q_a, q_b = test['runs'][a][j][0], test['runs'][b][chosen[b]][0]
                n = len(q_a)
                boot = [macro(q_a, pick) - macro(q_b, pick) for pick in (RNG.integers(0, n, n) for _ in range(draws))]
                point = macro(q_a) - macro(q_b)
                if b == 'retry':
                    row[f'{key}_cost'] = float(test['runs'][a][j][1].mean())
                swaps = RNG.random((draws, n)) < 0.5
                null = [macro(np.where(s, q_b, q_a)) - macro(np.where(s, q_a, q_b)) for s in swaps]
                p = float((np.sum(np.abs(null) >= abs(point) - 1e-12) + 1) / (draws + 1))
                row[key] = [point, float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))]
                row[f'{key}_p'] = p
        out[label] = row
    return out


def a3_verdict(curves: dict) -> str:
    """HOLDS: levers beat escalation (CI above 0) at ≥ 2 of the 3 budgets of one family for some model."""

    best, worst = 0, 0
    for c in curves.values():
        for family in ('big×', 'small×'):
            rows = [r for label, r in c['matched'].items() if label.startswith(family) and 'diff' in r]
            best = max(best, sum(r['diff'][1] > 0 for r in rows))
            worst = max(worst, sum(r['diff'][2] < 0 for r in rows))
    return 'HOLDS' if best >= 2 else ('REVERSED' if worst >= 2 else 'NOT SHOWN')


def main() -> None:
    frame = episodes()
    print(f'{len(frame)} episodes, models: {sorted(frame.model.unique())}')
    results = {'n_episodes': int(len(frame)), 'models': sorted(frame.model.unique()),
               'quality': quality(frame)}
    results['A4'] = a4_skill_curve(frame)
    results['A2'] = a2_recoverable(frame)
    results['A1'] = a1_detector(frame)
    results['A5'] = a5_wsp(frame)
    results['A3'] = a3_policies(frame, results['A1'])
    (ROOT / 'data/results/prereg-stage1.json').write_text(json.dumps(results, indent=1, default=float))
    for key in ('A1', 'A2', 'A3', 'A4', 'A5'):
        print(key, results[key]['verdict'])


if __name__ == '__main__':
    main()
