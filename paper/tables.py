"""The appendix tables → paper/tables.tex: every pre-registered verdict, and the parser sensitivity runs.

    python -m paper.tables        (after the analysis scripts)
"""

import json

from analysis.data import ROOT

RESULTS = ROOT / 'data/results'
# a one-line paraphrase of each hypothesis; the rule in full is in prereg/
CLAIMS = {
    'A1': 'Trajectory signals at step $\\le$ 3 predict a small model\'s failure better than the prompt',
    'A2': 'At least 40\\% of small-model failures are recoverable obstacles',
    'A3': 'A lever-matched cascade beats escalation alone at equal cost',
    'A4': 'The lift from a skill is $\\cap$-shaped in model size',
    'A5': 'A weight edit (WSP) lowers loops and raises quality',
    'HR1': 'Some harness component helps small models and not large ones, or the reverse',
    'HR2': 'A harness chosen per model beats the best single harness',
    'HR3': 'Retrying with the model\'s own harness beats a plain retry',
    'HR4': 'Routing (model, harness) pairs dominates routing models alone',
    'HR5': 'A rule that assigns harnesses is non-inferior to a search',
    'L1': 'Worked examples help at least 5 of 7 new small models',
    'L2': 'Instructions help at most 2 of 7, and less than large models',
    'L3': 'The corrected rule router is non-inferior to a search',
    'PD1': 'The best executor $\\le$ 4B is within 5\\,pp of the 27B executor',
    'PD2': 'Delegation beats every single model at some budget',
    'PD3': 'An executor $\\le$ 1.5B is already within 5\\,pp',
    'PD4': 'Swapping the planner matters more than swapping the executor',
    'PD5': 'Worked calls help both executors',
    'CP1': 'A 2.6B planner holds its role and beats its model alone',
    'CP2': 'A 9B planner lifts a 350M executor',
    'CP3': 'Planning after a failure is non-inferior to planning always, at $\\le$ 60\\% of the cost',
    'CP4': 'The router beats the best single model at some budget',
    'CP5': 'The router beats the tiny stack at the tiny stack\'s cost',
}
STAGES = {'A': 'Stage 1', 'H': 'Stage 2', 'L': 'Phase L', 'P': 'Stage 3', 'C': 'Stage 4'}


def verdicts() -> str:
    table = json.loads((RESULTS / 'verdicts.json').read_text())
    rows, stage = [], None
    for key, claim in CLAIMS.items():
        if STAGES[key[0]] != stage:
            stage = STAGES[key[0]]
            rows.append(f'\\multicolumn{{4}}{{l}}{{\\textit{{{stage}}}}} \\\\')
        v = table[key]
        held = v['verdict'] == 'HOLDS'
        p = f"{v['p']:.4f}" if held else '--'
        rows.append(f"{key} & {claim} & {'holds' if held else 'not shown'} & {p} \\\\")
    return ('\\begin{tabular}{@{}l p{0.62\\textwidth} l r@{}}\n\\toprule\nID & Claim & Verdict & $p$ \\\\\n\\midrule\n'
            + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}')


def parser() -> str:
    runs = json.loads((RESULTS / 'c2.json').read_text())
    rows = [f"{r['model']} & {r['parser']} & {r['diff']['n']} & {100 * r['old']:.1f} & {100 * r['final']:.1f} & "
            f"${100 * r['diff']['diff']:+.1f}$ & {100 * r['broken_old']:.1f} & {100 * r['broken_final']:.1f} \\\\" for r in runs.values()]
    return ('\\begin{tabular}{@{}l l r r r r r r@{}}\n\\toprule\n'
            'Model & Old & Tasks & Old \\% & Final \\% & $\\Delta$ pp & Broken old \\% & Broken final \\% \\\\\n\\midrule\n'
            + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}')


def main() -> None:
    body = (f'% written by paper/tables.py — do not edit\n\\newcommand{{\\verdictTable}}{{{verdicts()}}}\n'
            f'\\newcommand{{\\parserTable}}{{{parser()}}}\n')
    (ROOT / 'paper/tables.tex').write_text(body)
    print('paper/tables.tex')


if __name__ == '__main__':
    main()
