"""C1 — how often a small model solves what the 27B model solves, and at what cost per task.

    python -m analysis.claims.c1_replace     → data/results/c1.json

Replaceability of m on a track: among the test tasks qwen3.8 solves alone, the share m also solves alone.
Cost ratio: qwen3.8's mean FLOPs per episode over m's, on the same multi-turn and shell test tasks.
"""

import json

from analysis.data import ACTIVE, ROOT, solo

BIG = 'qwen3.8'
TRACKS = ['multiturn', 'shell', 'tools', 'code']


def main() -> None:
    frame = solo()
    base = frame[(frame.condition == 'base') & (frame.repeat == 0) & (frame.split == 'test')]
    big = base[base.model == BIG].set_index(['track', 'task_id'])
    out = {}
    for model in sorted(set(base.model) - {BIG}, key=lambda m: -ACTIVE[m]):
        small = base[base.model == model].set_index(['track', 'task_id'])
        common = big.index.intersection(small.index)
        row = {'active_b': ACTIVE[model], 'replace': {}}
        for track in TRACKS:
            solved = [k for k in common if k[0] == track and big.loc[k, 'passed']]
            if solved:
                row['replace'][track] = float(small.loc[solved, 'passed'].mean())
        agentic = [k for k in common if k[0] in ('multiturn', 'shell')]
        if agentic:
            row['cost_ratio'] = float(big.loc[agentic, 'flops'].mean() / small.loc[agentic, 'flops'].mean())
        out[model] = row
    (ROOT / 'data/results').mkdir(parents=True, exist_ok=True)
    (ROOT / 'data/results/c1.json').write_text(json.dumps(out, indent=1))
    for model, row in out.items():
        cells = ' '.join(f'{t} {v:4.0%}' for t, v in row['replace'].items())
        print(f"{model:18} {row['active_b']:5.2f}B  {cells}  cost ÷{row.get('cost_ratio', float('nan')):.1f}")


if __name__ == '__main__':
    main()
