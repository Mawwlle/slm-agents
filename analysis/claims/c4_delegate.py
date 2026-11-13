"""C4 — a 27B planner levels small executors, feedback is what makes it work, and the planner is most of the cost.

    python -m analysis.claims.c4_delegate     → data/results/c4.json

All on the test split, macro over the decomposable tracks (multi-turn and shell), paired by task.
"""

import json

from analysis.data import ACTIVE, ROOT, delegated, solo
from analysis.stats import macro, paired

PLANNER = 'qwen3.8'
TRACKS = ['multiturn', 'shell']


def main() -> None:
    plan = delegated()
    plan = plan[(plan.split == 'test') & plan.track.isin(TRACKS)]
    alone = solo()
    alone = alone[(alone.split == 'test') & alone.track.isin(TRACKS) & (alone.condition == 'base') & (alone.repeat == 0)]
    tracks = alone.drop_duplicates('task_id').set_index('task_id').track

    def planned(model, condition='delegate'):
        return plan[(plan.model == model) & (plan.condition == condition) & (plan.planner == PLANNER)].set_index('task_id')

    def own(model):
        return alone[alone.model == model].set_index('task_id')

    def quality(rows):
        return macro(rows.passed.astype(float), tracks.reindex(rows.index))

    reference = planned(PLANNER).passed  # P→B: qwen3.8 executes its own plan
    executors = sorted(set(plan[(plan.planner == PLANNER) & (plan.model != PLANNER)].model), key=ACTIVE.get)
    out = {'executors': {}, 'p_to_b': quality(planned(PLANNER)), 'big_alone': quality(own(PLANNER)),
           'p_to_b_by_track': {t: float(planned(PLANNER)[planned(PLANNER).track == t].passed.mean()) for t in TRACKS}}
    for m in executors:
        with_plan, by_itself = planned(m), own(m)
        out['executors'][m] = {
            'active_b': ACTIVE[m],
            'alone': {t: float(by_itself[by_itself.track == t].passed.mean()) for t in TRACKS},
            'planned': {t: float(with_plan[with_plan.track == t].passed.mean()) for t in TRACKS},
            'vs_p_to_b': paired(with_plan.passed, reference, tracks),
            'planner_share': float(with_plan.planner_flops.sum() / with_plan.flops.sum()),
        }
    out['ablations'] = {m: {'without_feedback': paired(planned(m).passed, planned(m, 'delegate_open').passed, tracks),
                            'without_examples': paired(planned(m).passed, planned(m, 'delegate+noex').passed, tracks)}
                        for m in ('lfm2.5-1.2b', 'qwen3.5-4b')}
    (ROOT / 'data/results/c4.json').write_text(json.dumps(out, indent=1))
    print(f"P→B {out['p_to_b']:.3f}  qwen3.8 alone {out['big_alone']:.3f}")
    for m, r in out['executors'].items():
        v = r['vs_p_to_b']
        print(f"{m:16} alone mt {r['alone']['multiturn']:.3f} sh {r['alone']['shell']:.3f} | planned mt {r['planned']['multiturn']:.3f} "
              f"sh {r['planned']['shell']:.3f} | vs P→B {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] | planner {r['planner_share']:.0%}")
    for m, r in out['ablations'].items():
        print(m, {k: f"{v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]" for k, v in r.items()})


if __name__ == '__main__':
    main()
