"""C5 — prompted small planners: does a planner hold its role, and what a cheap plan does to weak and strong executors.

    python -m analysis.claims.c5_cheap     → data/results/c5.json

Stage 4 runs on all 650 tasks of the multi-turn tracks and shell (all splits). A planner turn breaks the protocol when its
reply is not one JSON object with `step` or `finish` (ear.delegate.decision).
"""

import json

from analysis.data import ACTIVE, MULTITURN, ROOT, delegated, solo
from analysis.prereg import cp
from analysis.stats import macro, paired
from ear.delegate import decision

TRACKS = MULTITURN + ['shell']
PLANNERS = ['lfm2.5-2.6b', 'qwen3.5-4b', 'qwen3.5-9b']
EXECUTORS = ['lfm2.5-350m', 'lfm2.5-2.6b', 'qwen3.5-4b']


def broken(steps: list[dict]) -> tuple[int, int]:
    turns = [s for s in steps if s.get('role') == 'planner']
    return sum(not (decision(s['text']) or {}).keys() & {'step', 'finish'} for s in turns), len(turns)


def main() -> None:
    plan = delegated(steps=True)
    plan = plan[plan.planner.isin(PLANNERS) & plan.track.isin(TRACKS)]
    alone = solo()
    alone = alone[alone.track.isin(TRACKS) & (alone.condition == 'base') & (alone.repeat == 0) & alone.model.isin(EXECUTORS)]
    tracks = alone.drop_duplicates('task_id').set_index('task_id').track
    out = {'protocol': {}, 'systems': {}}
    for planner in PLANNERS:
        counts = [broken(s) for s in plan[plan.planner == planner].steps]
        out['protocol'][planner] = {'active_b': ACTIVE[planner], 'violation_rate': sum(c[0] for c in counts) / sum(c[1] for c in counts)}
    for executor in EXECUTORS:
        own = alone[alone.model == executor].set_index('task_id')
        out['systems'][f'{executor} alone'] = summary(own, tracks)
        for planner in PLANNERS:
            rows = plan[(plan.model == executor) & (plan.planner == planner)].set_index('task_id')
            if len(rows):
                out['systems'][f'{planner} → {executor}'] = summary(rows, tracks) | {'vs_alone': paired(rows.passed, own.passed, tracks)}
    out['frontier'] = frontier()
    (ROOT / 'data/results/c5.json').write_text(json.dumps(out, indent=1))
    for planner, r in out['protocol'].items():
        print(f"planner {planner:12} ({r['active_b']}B): protocol broken in {r['violation_rate']:.0%} of turns")
    for name, r in out['systems'].items():
        vs = f" vs alone {r['vs_alone']['diff']:+.3f} [{r['vs_alone']['lo']:+.3f}, {r['vs_alone']['hi']:+.3f}]" if 'vs_alone' in r else ''
        print(f"{name:28} Q {r['q']:.3f}  FLOPs {r['flops']:.2e}{vs}")


def frontier() -> dict:
    """Stage-4 cost against quality on the pre-registered frame: every system, and R(4B; 9B) over its thresholds with
    the share of tasks it sends to the plan."""

    frame = cp.load()
    passed, flops = cp.table(frame, 'passed'), cp.table(frame, 'flops')
    tracks = frame.drop_duplicates('task_id').set_index('task_id').track
    systems = {}
    for system in passed.columns:
        rows = passed[system].dropna()
        systems[system] = {'q': cp.macro(rows.to_numpy(), tracks.reindex(rows.index).to_numpy()),
                           'flops': float(flops[system].reindex(rows.index).mean())}
    router = cp.Router(frame, 'qwen3.5-4b', 'qwen3.5-9b')
    sweep = []
    for signal, tau in router.options():
        q, c = router.outcome(signal, tau)
        fire = router.observed if signal == 'obs' else router.risk >= tau
        sweep.append({'signal': signal, 'tau': float(tau), 'q': cp.macro(q, tracks.reindex(router.tasks).to_numpy()),
                      'flops': float(c.mean()), 'planned': float(fire.mean())})
    return {'systems': systems, 'router': sorted(sweep, key=lambda r: r['flops'])}


def summary(rows, tracks) -> dict:
    return {'q': macro(rows.passed.astype(float), tracks.reindex(rows.index)), 'flops': float(rows.flops.mean()),
            'by_track': {t: float(rows[rows.track == t].passed.mean()) for t in TRACKS}}


if __name__ == '__main__':
    main()
