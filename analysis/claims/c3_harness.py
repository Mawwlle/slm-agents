"""C3 — harness fixes are model-specific: which stage-2 findings held on seven new models, and what retrying with a
different harness buys.

    python -m analysis.claims.c3_harness     → data/results/c3.json

These numbers are the pre-registered procedures' own (analysis/prereg, run first); this script only collects them.
"""

import json

from analysis.data import ROOT

RESULTS = ROOT / 'data/results'


def cells(cascade: dict, key: str) -> dict:
    """Per model with qwen3.8 as the big model: the cascade differences at each budget."""

    curves = cascade['by_big']['qwen3.8']['curves']
    return {m: {b: r[key] for b, r in c['matched'].items() if key in r} for m, c in curves.items()}


def main() -> None:
    stage2 = json.loads((RESULTS / 'prereg-stage2.json').read_text())
    phase_l = json.loads((RESULTS / 'prereg-stage3.json').read_text())
    out = {
        'memory_lift': phase_l['L1']['per_model'], 'memory_helps': f"{phase_l['L1']['wins']} of {phase_l['L1']['of']}",
        'instruction_lift': phase_l['L2']['per_model'], 'instruction_gap': phase_l['L2']['gap'],
        'instruction_mean_new': phase_l['L2']['mean_new'], 'instruction_mean_gateway': phase_l['L2']['mean_gateway'],
        'skill_extra_flops': phase_l['L4'],
        'switch_minus_retry': cells(stage2['HR3'], 'diff_switch'),
        'switch_false_pass': stage2['calibration']['hr3_false_pass'],
    }
    (RESULTS / 'c3.json').write_text(json.dumps(out, indent=1))
    print('memory helps', out['memory_helps'], '| instructions: new models', f"{out['instruction_mean_new']:+.3f}",
          'gateway', f"{out['instruction_mean_gateway']:+.3f}", 'gap', [round(x, 3) for x in out['instruction_gap']])
    for m, budgets in out['switch_minus_retry'].items():
        print(f'switch − retry {m:18}', {b: round(v[0], 3) for b, v in budgets.items()})
    print('switch rule on shuffled labels: false pass', out['switch_false_pass'])


if __name__ == '__main__':
    main()
