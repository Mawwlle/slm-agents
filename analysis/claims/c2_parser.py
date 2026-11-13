"""C2 — how much the tool-call parser moves a model's measured quality.

    python -m analysis.claims.c2_parser     → data/results/c2.json

Runs made with earlier parser versions (data/episodes/parser/v1, v2, v3) were set aside when the parser changed and
rerun from scratch; the pass rate of an old run is compared with the final run on the same `base` tasks.
v1 did not read Gemma 4's `call:name{…}` shape, v2 read it, v3 lacked the unambiguous-recovery rules of v4.
Quality is the macro pass rate over tracks on the tasks both runs have; the difference is final − old.
"""

import json

import pandas as pd

from analysis.data import EPISODES, ROOT, files, read
from analysis.stats import macro, paired


def base(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    return frame[(frame.condition == 'base') & (frame.repeat == 0)].drop_duplicates('task_id', keep='last').set_index('task_id')


def broken_share(path) -> float:
    steps = [s for line in open(path) for s in json.loads(line)['steps']]
    return sum(s['broken'] for s in steps) / max(1, len(steps))


def main() -> None:
    out = {}
    for version in sorted(p.name for p in (EPISODES / 'parser').iterdir()):
        for old_path in sorted((EPISODES / 'parser' / version).glob('*.jsonl')):
            model = old_path.stem
            final_path = files(model, '')[0]
            old, final = base(read(old_path)), base(read(final_path))
            common = old.index.intersection(final.index)
            old, final = old.loc[common], final.loc[common]
            out[f'{model} {version}'] = {
                'model': model, 'parser': version,
                'old': macro(old.passed.astype(float), final.track), 'final': macro(final.passed.astype(float), final.track),
                'diff': paired(final.passed, old.passed, final.track),
                'broken_old': broken_share(old_path), 'broken_final': broken_share(final_path)}
    (ROOT / 'data/results/c2.json').write_text(json.dumps(out, indent=1))
    for key, r in out.items():
        print(f"{key:28} n={r['diff']['n']:5}  pass {r['old']:.3f} → {r['final']:.3f}  "
              f"({r['diff']['diff']:+.3f} [{r['diff']['lo']:+.3f}, {r['diff']['hi']:+.3f}])  broken steps {r['broken_old']:.1%} → {r['broken_final']:.1%}")


if __name__ == '__main__':
    main()
