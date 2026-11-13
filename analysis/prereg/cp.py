"""Stage 4 tests (PREREG-CP.md): a cheap planner with a router against a tiny stack → data/results/prereg-stage4.json.

    uv run --with pandas --with numpy --with scikit-learn python -m analysis.prereg.cp [copies]

Every task is scored by a rule fitted on the other four folds (5 folds by task, stratified by track); intervals are
95% paired task bootstraps within track; the calibration reruns the selection on copies whose pass labels are
shuffled among all systems' episodes of each task.
"""

import hashlib
import json
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.prereg.load import ROOT, RUNS, step_features
from ear.delegate import decision

TRACKS = ['multiturn', 'multiturn_miss_param', 'multiturn_long_context', 'shell']
SOLO = ['qwen3.5-4b', 'qwen3.5-9b', 'lfm2.5-2.6b', 'lfm2.5-350m']
PLANNERS = {'cp9': 'qwen3.5-9b', 'cp4': 'qwen3.5-4b', 'cpl': 'lfm2.5-2.6b'}
# R(M; P) of CP4. R(4B; 4B) is out: it needs D(4B→4B), which the pre-registered run list does not include (PREREG-CP log)
ROUTED = [('qwen3.5-4b', 'qwen3.5-9b'), ('lfm2.5-2.6b', 'qwen3.5-9b'), ('lfm2.5-2.6b', 'qwen3.5-4b')]
OBSERVED = {'timeout', 'loop', 'broken_tool_call', 'model_stopped', 'agent_failed'}
K, FOLDS, DRAWS = 3, 5, 10_000
RNG = np.random.default_rng(20261111)


def solo_files(m: str) -> list[tuple[str, str]]:
    """(file, condition) holding the `base` episodes of m on the four tracks."""

    shell = (f'{m}/{m}.jsonl', 'base') if m.startswith('qwen') else (f'{m}/{m}-L.jsonl', 'base')
    return [(f'{m}/{m}-mt.jsonl', 'base'), shell, (f'{m}/{m}-cpsolo.jsonl', 'base')]


def load() -> pd.DataFrame:
    rows = []

    def add(path, system, condition):
        if not (RUNS / path).exists():
            print(f'missing: {path}', file=sys.stderr)
            return
        for line in open(RUNS / path):
            e = json.loads(line)
            if e['track'] not in TRACKS or e['condition'] != condition or e['repeat'] != 0:
                continue
            steps = e['steps']
            prefix = steps[:K]
            rows.append({'system': system, 'task_id': e['task_id'], 'track': e['track'], 'passed': float(e['passed']),
                         'flops': e['flops'], 'failure': e['failure'],
                         'flops_k': e['flops'] * sum(s['prompt_tokens'] + s['completion_tokens'] for s in prefix)
                         / max(1, sum(s['prompt_tokens'] + s['completion_tokens'] for s in steps)),
                         'violations': sum(s.get('role') == 'planner' and not (decision(s['text']) or {}).keys() & {'step', 'finish'} for s in steps),
                         'planner_turns': sum(s.get('role') == 'planner' for s in steps), **step_features(steps, K)})

    for m in SOLO:
        for path, condition in solo_files(m):
            add(path, f'S({m})', condition)
    for suffix, planner in PLANNERS.items():
        for path in RUNS.glob(f'*/*-{suffix}.jsonl'):
            add(path.relative_to(RUNS), f'D({planner}→{path.parent.name})', 'delegate')
    frame = pd.DataFrame(rows).drop_duplicates(['system', 'task_id'], keep='last')
    frame['fold'] = frame.task_id.map(fold)
    return frame


def fold(task_id: str) -> int:
    return int(hashlib.sha256(f'cp:{task_id}'.encode()).hexdigest(), 16) % FOLDS


def table(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return frame.pivot(index='task_id', columns='system', values=column)


def macro(q: np.ndarray, tracks: np.ndarray) -> float:
    return float(np.mean([q[tracks == t].mean() for t in TRACKS if (tracks == t).any()]))


def paired(a: np.ndarray, b: np.ndarray, tracks: np.ndarray) -> list:
    """Macro of a − b over tracks, with a 95% bootstrap within track and a paired sign-flip permutation p."""

    groups = [np.flatnonzero(tracks == t) for t in TRACKS if (tracks == t).any()]
    d = a - b
    draws = np.mean([d[g][RNG.integers(0, len(g), (DRAWS, len(g)))].mean(1) for g in groups], 0)
    point = float(np.mean([d[g].mean() for g in groups]))
    flips = np.mean([(d[g] * RNG.choice([-1, 1], (2000, len(g)))).mean(1) for g in groups], 0)
    p = float((np.sum(np.abs(flips) >= abs(point) - 1e-12) + 1) / 2001)
    return [point, *np.quantile(draws, [0.025, 0.975]).tolist(), p]


class Router:
    """R(M; P): M alone; when the signal fires, the task again as D(P→M). Outcomes per task, for one threshold."""

    def __init__(self, frame: pd.DataFrame, m: str, p: str):
        self.solo, self.plan = f'S({m})', f'D({p}→{m})'
        tasks = frame[frame.system == self.solo].set_index('task_id')
        plan = frame[frame.system == self.plan].set_index('task_id')
        self.tasks = tasks.index.intersection(plan.index)
        self.s, self.d = tasks.loc[self.tasks], plan.loc[self.tasks]
        self.observed = self.s.failure.isin(OBSERVED).to_numpy()
        self.risk = np.zeros(len(self.tasks))
        features = [c for c in self.s.columns if c.startswith('f_')]
        X = pd.concat([self.s[features], pd.get_dummies(self.s.track).astype(float)], axis=1).to_numpy(float)
        y = 1 - self.s.passed.to_numpy()
        for f in range(FOLDS):  # each task's risk from a detector fitted on the other folds
            held = (self.s.fold == f).to_numpy()
            model = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(),
                                  LogisticRegression(C=0.5, max_iter=3000)).fit(X[~held], y[~held])
            self.risk[held] = model.predict_proba(X[held])[:, 1]

    def outcome(self, signal: str, tau: float) -> tuple[np.ndarray, np.ndarray]:
        s, d = self.s, self.d
        if signal == 'obs':
            fire = self.observed
            cost = np.where(fire, s.flops + d.flops, s.flops)
        else:
            fire = self.risk >= tau
            cost = np.where(fire, s.flops_k + d.flops, s.flops)
        return np.where(fire, d.passed, s.passed), np.asarray(cost, float)

    def options(self) -> list[tuple[str, float]]:
        return [('obs', 0.0)] + [('det', t) for t in np.quantile(self.risk, np.linspace(0, 1, 21))]


def crossfit(frame: pd.DataFrame, candidates: list, budget_of) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Per held-out fold, the candidate (a callable → per-task q, c over `tasks`) with the best fit Q under the fold's
    budget; returns the assembled held-out q, c."""

    tasks = candidates[0][1]
    tracks = frame.drop_duplicates('task_id').set_index('task_id').track.reindex(tasks).to_numpy()
    folds = np.array([fold(t) for t in tasks])
    q_out, c_out = np.full(len(tasks), np.nan), np.full(len(tasks), np.nan)
    for f in range(FOLDS):
        fit = folds != f
        budget = budget_of(fit)
        best, best_q = None, -1.0
        for _, _, get in candidates:
            q, c = get()
            if c[fit].mean() <= budget and macro(q[fit], tracks[fit]) > best_q:
                best, best_q = (q, c), macro(q[fit], tracks[fit])
        if best is not None:
            q_out[~fit], c_out[~fit] = best[0][~fit], best[1][~fit]
    return q_out, c_out, tracks


def tests(frame: pd.DataFrame) -> dict:
    P, C = table(frame, 'passed'), table(frame, 'flops')
    out = {}
    b = 'D(lfm2.5-2.6b→lfm2.5-350m)'
    tasks = P.index[P[[b, 'S(lfm2.5-2.6b)']].notna().all(axis=1)]
    tracks = frame.drop_duplicates('task_id').set_index('task_id').track.reindex(tasks).to_numpy()
    rows = frame[frame.system == b]
    violation = float(rows.violations.sum() / max(1, rows.planner_turns.sum()))
    d = paired(P.loc[tasks, b].to_numpy(), P.loc[tasks, 'S(lfm2.5-2.6b)'].to_numpy(), tracks)
    out['CP1'] = {'violation_rate': violation, 'diff': d, 'verdict': 'HOLDS' if violation <= 0.10 and d[1] > 0 else 'NOT SHOWN'}

    a, s = 'D(qwen3.5-9b→lfm2.5-350m)', 'S(lfm2.5-350m)'
    tasks2 = P.index[P[[a, s]].notna().all(axis=1)]
    t2 = frame.drop_duplicates('task_id').set_index('task_id').track.reindex(tasks2).to_numpy()
    d = paired(P.loc[tasks2, a].to_numpy(), P.loc[tasks2, s].to_numpy(), t2)
    out['CP2'] = {'diff': d, 'verdict': 'HOLDS' if d[1] > 0 else 'NOT SHOWN'}

    routers = {(m, p): Router(frame, m, p) for m, p in ROUTED}
    r = routers[('qwen3.5-4b', 'qwen3.5-9b')]
    dq, dc = P.loc[r.tasks, r.plan].to_numpy(), C.loc[r.tasks, r.plan].to_numpy()
    cands = [(sig, r.tasks, (lambda sig=sig, tau=tau: r.outcome(sig, tau))) for sig, tau in r.options()]
    q, c, tr = crossfit(frame, cands, lambda fit: 0.6 * dc[fit].mean())
    ok = ~np.isnan(q)
    out['CP3'] = {'diff': paired(q[ok], dq[ok], tr[ok]), 'cost_ratio': float(c[ok].mean() / dc[ok].mean()),
                  'covered': float(ok.mean())}
    out['CP3']['verdict'] = 'HOLDS' if out['CP3']['diff'][1] > -0.03 and out['CP3']['cost_ratio'] <= 0.6 and ok.all() else 'NOT SHOWN'

    common = P.index[P[[f'S({m})' for m in SOLO] + [b]].notna().all(axis=1)]
    for rt in routers.values():
        common = common.intersection(rt.tasks)
    r_cands, s_cands = [], []
    for rt in routers.values():
        idx = rt.tasks.get_indexer(common)
        for sig, tau in rt.options():
            r_cands.append((sig, common, (lambda rt=rt, sig=sig, tau=tau, idx=idx: tuple(x[idx] for x in rt.outcome(sig, tau)))))
    for m in SOLO:
        s_cands.append(('solo', common, (lambda m=m: (P.loc[common, f'S({m})'].to_numpy(), C.loc[common, f'S({m})'].to_numpy()))))
    costs = [get()[1].mean() for _, _, get in r_cands + s_cands]
    out['CP4'] = {'budgets': {}}
    for budget in np.geomspace(min(costs), max(costs), 16):
        q_r, _, tr = crossfit(frame, r_cands, lambda fit, budget=budget: budget)
        q_s, _, _ = crossfit(frame, s_cands, lambda fit, budget=budget: budget)
        ok = ~np.isnan(q_r) & ~np.isnan(q_s)
        if ok.all():
            out['CP4']['budgets'][f'{budget:.2e}'] = paired(q_r, q_s, tr)
    out['CP4']['verdict'] = 'HOLDS' if any(v[0] >= 0.02 and v[1] > 0 for v in out['CP4']['budgets'].values()) else 'NOT SHOWN'

    bq, bc = P.loc[common, b].to_numpy(), C.loc[common, b].to_numpy()
    q, c, tr = crossfit(frame, r_cands, lambda fit: bc[fit].mean())
    ok = ~np.isnan(q)
    out['CP5'] = {'diff': paired(q[ok], bq[ok], tr[ok]) if ok.any() else None, 'cost_A': float(np.nanmean(c)),
                  'cost_B': float(bc.mean()), 'covered': float(ok.mean())}
    out['CP5']['verdict'] = 'HOLDS' if ok.all() and out['CP5']['diff'][1] > 0 else 'NOT SHOWN'
    return out


def calibration(frame: pd.DataFrame, copies: int) -> dict:
    """CP3–CP5 on copies where each task's pass labels are shuffled among all systems' episodes of that task."""

    passes = {k: 0 for k in ('CP3', 'CP4', 'CP5')}
    for _ in range(copies):
        shuffled = frame.copy()
        shuffled['passed'] = shuffled.groupby('task_id').passed.transform(lambda s: RNG.permutation(s.to_numpy()))
        result = tests(shuffled)
        for k in passes:
            passes[k] += result[k]['verdict'] == 'HOLDS'
    return {k: v / copies for k, v in passes.items()} | {'copies': copies}


def main() -> None:
    frame = load()
    print(f'{len(frame)} episodes; systems: {sorted(frame.system.unique())}')
    results = tests(frame)
    results['calibration'] = calibration(frame, int(sys.argv[1]) if len(sys.argv) > 1 else 50)
    (ROOT / 'data/results/prereg-stage4.json').write_text(json.dumps(results, indent=1, default=float))
    for key in ('CP1', 'CP2', 'CP3', 'CP4', 'CP5'):
        print(key, results[key]['verdict'], {k: v for k, v in results[key].items() if k not in ('verdict', 'budgets')})
    print('calibration', results['calibration'])


if __name__ == '__main__':
    main()
