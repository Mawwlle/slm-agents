"""Stage 1–2 episodes from data/episodes/<model>/<name>.jsonl into one DataFrame, plus per-step features."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'data' / 'episodes'

# total parameters (B): the size axis; active parameters come with the episodes' FLOPs
SIZE = {
    'mnn-pocket-0.3b': 0.36, 'qwen3-0.6b': 0.75, 'qwen3.5-0.8b': 0.87, 'qwen3.5-0.8b-wsp': 0.87,
    'qwen3.5-2b': 2.27, 'granite-4.0-micro': 3.4, 'qwen3.5-4b': 4.66, 'qwen3-vl-8b': 8.77,
    'qwen3.5-9b': 9.65, 'gemma-4-26b-a4b': 25.2, 'qwen3.8': 27.0, 'qwen3-vl-30b': 31.1,
}
SMALL = [m for m, s in SIZE.items() if s < 12]
BIG = [m for m, s in SIZE.items() if s >= 12]
TRACKS = ['code', 'tools', 'shell']


STAGE12_FILES = ('', '-memory', '-hr')  # a model's stage 1–2 runs; later stages read their files by name (analysis.stage3)


def episodes() -> pd.DataFrame:
    """The stage 1–2 episodes of the 12 models in SIZE."""

    rows = []
    for path in [RUNS / m / f'{m}{s}.jsonl' for m in SIZE for s in STAGE12_FILES]:
        if not path.exists():
            continue
        for line in open(path):
            e = json.loads(line)
            if e['failure'] == 'agent_failed':
                continue
            steps = e.pop('steps')
            e['n_steps'] = len(steps)
            e['tokens'] = sum(s['completion_tokens'] for s in steps)
            e['steps'] = steps
            rows.append(e)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.drop_duplicates(['task_id', 'model', 'condition', 'repeat'], keep='last')
    frame['size'] = frame['model'].map(SIZE)
    frame['passed'] = frame['passed'].astype(bool)
    return frame.reset_index(drop=True)


def step_features(steps: list[dict], k: int) -> dict:
    """What a router sees after the first k steps (fewer if the episode ended sooner)."""

    seen = steps[:k]
    if not seen:
        return {'f_steps': 0}

    def mean(key):
        values = [s[key] for s in seen if s.get(key) is not None]
        return float(np.mean(values)) if values else np.nan

    ok = [s['tool_ok'] for s in seen if s.get('tool_ok') is not None]
    progress = [s['progress'] for s in seen if s.get('progress') is not None]
    return {
        'f_steps': len(seen),
        'f_ended': float(len(steps) <= k),
        'f_broken_share': float(np.mean([s['broken'] for s in seen])),
        'f_broken_last': float(seen[-1]['broken']),
        'f_repeat_share': float(np.mean([s['repeat'] for s in seen])),
        'f_no_call_share': float(np.mean([s['n_calls'] == 0 for s in seen])),
        'f_tool_ok_mean': float(np.mean(ok)) if ok else np.nan,
        'f_tool_ok_last': float(ok[-1]) if ok else np.nan,
        'f_progress_max': float(max(progress)) if progress else np.nan,
        'f_progress_seen': float(bool(progress)),
        'f_cut_share': float(np.mean([s['finish'] == 'length' for s in seen])),
        'f_repetition_max': float(max(s['repetition'] for s in seen)),
        'f_entropy_mean': mean('mean_entropy'),
        'f_logprob_mean': mean('mean_logprob'),
        'f_logprob_min': float(np.nanmin([s['min_logprob'] if s.get('min_logprob') is not None else np.nan for s in seen]))
        if any(s.get('min_logprob') is not None for s in seen) else np.nan,
        'f_tokens_log': float(np.log1p(sum(s['completion_tokens'] for s in seen))),
        'f_thinking_log': float(np.log1p(sum(s['thinking_chars'] for s in seen))),
        'f_skill_read': float(any(s.get('skill_read') for s in seen)),
        'f_nudged': float(any(s.get('nudge') for s in seen)),
    }


def split_of(task_id: str) -> str:
    import hashlib
    bucket = int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % 100
    return 'screen' if bucket < 15 else 'train' if bucket < 40 else 'test'
