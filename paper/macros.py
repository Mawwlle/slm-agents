"""Every number the paper quotes, as LaTeX macros → paper/numbers.tex. The text uses only these.

    python -m paper.macros        (after the analysis scripts)
"""

import json

from analysis.data import ROOT

RESULTS = ROOT / 'data/results'


def load(name: str) -> dict:
    return json.loads((RESULTS / f'{name}.json').read_text())


def pct(x: float) -> str:
    return f'{100 * x:.0f}\\%'


def pp(x: float) -> str:
    """A signed difference in percentage points, in math mode so the minus is a minus."""

    return f'${100 * x:+.1f}$'


def ci(d: dict | list) -> str:
    lo, hi = (d['lo'], d['hi']) if isinstance(d, dict) else (d[1], d[2])
    return f'$[{100 * lo:+.1f}, {100 * hi:+.1f}]$'


def numbers() -> dict[str, str]:
    c1, c2, c3, c4, c5 = (load(f'c{i}') for i in range(1, 6))
    verdicts, stage4 = load('verdicts'), load('prereg-stage4')
    manifest = json.loads((ROOT / 'data/episodes.json').read_text())
    kept = [f for f in manifest.values() if 'set aside' not in f['stage']]
    holds = [k for k, v in verdicts.items() if v['verdict'] == 'HOLDS']
    n = {
        'nEpisodes': f"{sum(f['episodes'] for f in kept):,}".replace(',', '{,}'),
        'nModels': str(len({m for m in c1} | {'qwen3.8'})),
        'nHypotheses': str(len(verdicts)), 'nHolds': str(len(holds)),
        'nHolm': str(sum(verdicts[k]['holm_survives'] for k in holds)),
    }
    for key, model in (('Nine', 'qwen3.5-9b'), ('Four', 'qwen3.5-4b'), ('LfmTwo', 'lfm2.5-2.6b'), ('VlThirty', 'qwen3-vl-30b'),
                       ('LfmOne', 'lfm2.5-1.2b')):
        r = c1[model]
        n[f'repl{key}MT'], n[f'repl{key}Shell'] = pct(r['replace']['multiturn']), pct(r['replace']['shell'])
        n[f'repl{key}Code'], n[f'cost{key}'] = pct(r['replace']['code']), f"{r['cost_ratio']:.1f}"
    g1 = c2['gemma-4-26b-a4b v1']
    n |= {'gemmaOld': pct(g1['old']), 'gemmaNew': pct(g1['final']), 'gemmaDiff': pp(g1['diff']['diff']), 'gemmaCI': ci(g1['diff']),
          'gemmaN': str(g1['diff']['n']), 'qwenBrokenOld': f"{100 * c2['qwen3.8 v3']['broken_old']:.1f}\\%",
          'qwenBrokenNew': f"{100 * c2['qwen3.8 v3']['broken_final']:.1f}\\%"}
    n |= {'memoryHelps': c3['memory_helps'].split(' of ')[0], 'instrNew': pp(c3['instruction_mean_new']),
          'instrGateway': pp(c3['instruction_mean_gateway']), 'instrGap': pp(c3['instruction_gap'][0]), 'instrGapCI': ci(c3['instruction_gap']),
          'switchFalse': pct(c3['switch_false_pass'])}
    four = c3['switch_minus_retry']['qwen3.5-4b']
    n |= {'switchLo': pp(min(v[0] for v in four.values())), 'switchHi': pp(max(v[0] for v in four.values()))}
    e = c4['executors']
    n |= {'pbMT': pct(c4['p_to_b_by_track']['multiturn']), 'bigAloneQ': pct(c4['big_alone']),
          'tinyAloneMT': pct(e['lfm2.5-350m']['alone']['multiturn']), 'tinyPlanMT': pct(e['lfm2.5-350m']['planned']['multiturn']),
          'fourAloneMT': pct(e['qwen3.5-4b']['alone']['multiturn']), 'fourPlanMT': pct(e['qwen3.5-4b']['planned']['multiturn']),
          'bestVsPB': pp(e['lfm2.5-2.6b']['vs_p_to_b']['diff']), 'bestVsPBCI': ci(e['lfm2.5-2.6b']['vs_p_to_b']),
          'plannerShareLo': pct(min(r['planner_share'] for r in e.values())),
          'feedbackSmall': pp(c4['ablations']['lfm2.5-1.2b']['without_feedback']['diff']),
          'feedbackFour': pp(c4['ablations']['qwen3.5-4b']['without_feedback']['diff']),
          'examplesSmall': pp(c4['ablations']['lfm2.5-1.2b']['without_examples']['diff']),
          'examplesFour': pp(c4['ablations']['qwen3.5-4b']['without_examples']['diff'])}
    level = [r['planned']['multiturn'] for r in e.values() if r['active_b'] >= 1.5]
    n |= {'levelLo': f'{100 * min(level):.0f}', 'levelHi': f'{100 * max(level):.0f}\\%'}
    proto, systems = c5['protocol'], c5['systems']
    n |= {'protoLfm': pct(proto['lfm2.5-2.6b']['violation_rate']), 'protoFour': pct(proto['qwen3.5-4b']['violation_rate']),
          'protoNine': pct(proto['qwen3.5-9b']['violation_rate']),
          'tinyStackQ': pct(systems['lfm2.5-2.6b → lfm2.5-350m']['q']),
          'liftTiny': pp(systems['qwen3.5-9b → lfm2.5-350m']['vs_alone']['diff']), 'liftTinyCI': ci(systems['qwen3.5-9b → lfm2.5-350m']['vs_alone']),
          'hurtFour': pp(systems['qwen3.5-9b → qwen3.5-4b']['vs_alone']['diff']), 'hurtFourCI': ci(systems['qwen3.5-9b → qwen3.5-4b']['vs_alone']),
          'hurtLfm': pp(systems['qwen3.5-9b → lfm2.5-2.6b']['vs_alone']['diff']),
          'planCostFour': f"{systems['qwen3.5-9b → qwen3.5-4b']['flops'] / systems['qwen3.5-4b alone']['flops']:.1f}",
          'routerVsPlan': pp(stage4['CP3']['diff'][0]), 'routerVsPlanCI': ci(stage4['CP3']['diff']),
          'routerCost': pct(stage4['CP3']['cost_ratio']), 'cpFourFalse': pct(stage4['calibration']['CP4']),
          'tinyStackVsAlone': pp(stage4['CP1']['diff'][0]), 'tinyStackVsAloneCI': ci(stage4['CP1']['diff'])}
    best = max(stage4['CP4']['budgets'].values(), key=lambda v: v[0])
    n |= {'routerVsSingle': pp(best[0]), 'routerVsSingleCI': ci(best)}
    return n


def main() -> None:
    lines = [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in numbers().items()]
    (ROOT / 'paper/numbers.tex').write_text('% written by paper/macros.py — do not edit\n' + '\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
