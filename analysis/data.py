"""Episodes, by stage, from data/episodes/<model>/<file>.jsonl (fetched by data/fetch.py).

Files are named, never globbed: a model's folder holds runs of several stages, and two delegation runs share a
condition name (`-pd` is planned by qwen3.8, `-pdg` by Gemma-4-26B).
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EPISODES = ROOT / 'data/episodes'

STAGE1 = ['mnn-pocket-0.3b', 'qwen3-0.6b', 'qwen3.5-0.8b', 'qwen3.5-0.8b-wsp', 'qwen3.5-2b', 'granite-4.0-micro',
          'qwen3.5-4b', 'qwen3-vl-8b', 'qwen3.5-9b', 'gemma-4-26b-a4b', 'qwen3.8', 'qwen3-vl-30b']
PHASE_L = ['lfm2.5-230m', 'lfm2.5-350m', 'lfm2.5-1.2b', 'lfm2.5-2.6b', 'lfm2.5-8b-a1b', 'gemma-4-e2b', 'gemma-4-e4b']
# active parameters (B): the FLOPs of an episode are 2 · active · tokens
ACTIVE = {'mnn-pocket-0.3b': 0.36, 'qwen3-0.6b': 0.6, 'qwen3.5-0.8b': 0.8, 'qwen3.5-0.8b-wsp': 0.8, 'qwen3.5-2b': 2.0,
          'granite-4.0-micro': 3.4, 'qwen3.5-4b': 4.2, 'qwen3-vl-8b': 8.2, 'qwen3.5-9b': 9.0, 'gemma-4-26b-a4b': 3.8,
          'qwen3.8': 27.0, 'qwen3-vl-30b': 3.0, 'lfm2.5-230m': 0.23, 'lfm2.5-350m': 0.35, 'lfm2.5-1.2b': 1.2,
          'lfm2.5-2.6b': 2.69, 'lfm2.5-8b-a1b': 1.5, 'gemma-4-e2b': 2.3, 'gemma-4-e4b': 4.5}
SOLO_TRACKS = ['code', 'tools', 'shell']
MULTITURN = ['multiturn', 'multiturn_miss_param', 'multiturn_long_context']


def read(path: Path, steps: bool = False, **fixed) -> list[dict]:
    rows = []
    for line in open(path):
        e = json.loads(line)
        row = {k: e.get(k) for k in ('model', 'condition', 'repeat', 'task_id', 'track', 'split', 'failure', 'planner')}
        row |= {'passed': bool(e['passed']), 'flops': e['flops'], 'planner_flops': e.get('planner_flops', 0.0)} | fixed
        if steps:
            row['steps'] = e['steps']
        rows.append(row)
    return rows


def files(model: str, *suffixes: str) -> list[Path]:
    return [p for s in suffixes if (p := EPISODES / model / f'{model}{s}.jsonl').exists()]


def solo(steps: bool = False) -> pd.DataFrame:
    """Every model alone: stages 1–2 (code, tools, shell), phase L, and the multi-turn tracks."""

    rows = []
    for m in STAGE1:
        rows += [r for p in files(m, '', '-memory', '-hr', '-mt', '-cpsolo') for r in read(p, steps)]
    for m in PHASE_L:
        rows += [r for p in files(m, '-L', '-mt', '-cpsolo') for r in read(p, steps)]
    return dedupe(pd.DataFrame(rows))


def delegated(steps: bool = False) -> pd.DataFrame:
    """Planner-executor runs: stage 3 (`-pd` qwen3.8, `-pdg` Gemma-26B) and stage 4 (`-cp9`, `-cp4`, `-cpl`)."""

    planners = {'-pd': 'qwen3.8', '-pdg': 'gemma-4-26b-a4b', '-cp9': 'qwen3.5-9b', '-cp4': 'qwen3.5-4b', '-cpl': 'lfm2.5-2.6b'}
    rows = []
    for m in STAGE1 + PHASE_L:
        for suffix, planner in planners.items():
            rows += [r for p in files(m, suffix) for r in read(p, steps, planner=planner)]
    return dedupe(pd.DataFrame(rows))


def dedupe(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [k for k in ('model', 'planner', 'condition', 'repeat', 'task_id') if k in frame]
    return frame.drop_duplicates(keys, keep='last').reset_index(drop=True)
