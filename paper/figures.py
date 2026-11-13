"""The paper's figures, from data/results/*.json → paper/figures/*.pdf.

    python -m paper.figures        (after the analysis/claims scripts)

Series carry a marker as well as a colour, so the figures read in black and white.
"""

import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analysis.data import ROOT  # noqa: E402

RESULTS, OUT = ROOT / 'data/results', ROOT / 'paper/figures'
BLUE, ORANGE, GREEN, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#8a8f98'
NAMES = {'qwen3.5-9b': 'Qwen3.5-9B', 'qwen3.5-4b': 'Qwen3.5-4B', 'lfm2.5-2.6b': 'LFM2.5-2.6B', 'lfm2.5-350m': 'LFM2.5-350M',
         'lfm2.5-1.2b': 'LFM2.5-1.2B', 'lfm2.5-8b-a1b': 'LFM2.5-8B-A1B', 'qwen3.5-2b': 'Qwen3.5-2B', 'gemma-4-e2b': 'Gemma-4-E2B',
         'gemma-4-e4b': 'Gemma-4-E4B'}
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e3e6ea', 'grid.linewidth': 0.6, 'legend.frameon': False, 'pdf.fonttype': 42})


def load(name: str) -> dict:
    return json.loads((RESULTS / f'{name}.json').read_text())


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f'{name}.pdf', metadata={'CreationDate': None})  # no timestamp, so a rerun leaves the file as it was
    plt.close(fig)


def replaceability() -> None:
    """F1: share of qwen3.8's solved test tasks a model also solves, against its active parameters."""

    c1 = load('c1')
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for track, colour, marker in (('multiturn', BLUE, 'o'), ('shell', ORANGE, 's'), ('code', GREEN, '^')):
        pts = sorted((r['active_b'], r['replace'][track]) for r in c1.values() if track in r['replace'])
        ax.scatter(*zip(*pts), s=14, color=colour, marker=marker, label=track, zorder=3)
    ax.set_xscale('log')
    ax.set_xticks([0.25, 0.5, 1, 2, 4, 8], ['0.25', '0.5', '1', '2', '4', '8'])
    ax.set_xlabel('active parameters (B)')
    ax.set_ylabel('solves what qwen3.8 solves')
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(loc='upper left', handletextpad=0.2)
    save(fig, 'replaceability')


def levelling() -> None:
    """F2: each executor alone and with qwen3.8's plan, on multi-turn (test)."""

    executors = load('c4')['executors']
    order = sorted(executors, key=lambda m: executors[m]['active_b'])
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    alone = [executors[m]['alone']['multiturn'] for m in order]
    planned = [executors[m]['planned']['multiturn'] for m in order]
    ax.hlines(y, alone, planned, color=GREY, linewidth=1, zorder=2)
    ax.scatter(alone, y, s=16, color=BLUE, marker='o', label='alone', zorder=3)
    ax.scatter(planned, y, s=16, color=ORANGE, marker='s', label='with qwen3.8 planning', zorder=3)
    ax.axvline(load('c4')['p_to_b_by_track']['multiturn'], color=GREY, linestyle='--', linewidth=0.8)  # qwen3.8 executing its own plan
    ax.set_yticks(y, [NAMES[m] for m in order])
    ax.set_xlabel('multi-turn tasks solved (test)')
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.2)
    save(fig, 'levelling')


def frontier() -> None:
    """F3: stage-4 cost against quality — single models, always-on plans, the router's threshold sweep, the tiny stack."""

    frontier = load('c5')['frontier']
    systems, sweep = frontier['systems'], frontier['router']

    def point(system):
        return systems[system]['flops'], systems[system]['q']

    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    singles = [s for s in systems if s.startswith('S(')]
    plans = [s for s in systems if s.startswith('D(') and s != 'D(lfm2.5-2.6b→lfm2.5-350m)']
    ax.scatter(*zip(*map(point, singles)), s=18, color=BLUE, marker='o', label='one model', zorder=3)
    ax.scatter(*zip(*map(point, plans)), s=18, color=ORANGE, marker='s', label='always planned', zorder=3)
    ax.plot([r['flops'] for r in sweep], [r['q'] for r in sweep], color=GREEN, marker='^', markersize=3, linewidth=1,
            label='router R(4B; 9B)', zorder=3)
    ax.scatter(*point('D(lfm2.5-2.6b→lfm2.5-350m)'), s=30, color='black', marker='x', label='tiny stack (B)', zorder=4)
    for s in singles:
        x, q = point(s)
        left = x < 1e14
        ax.annotate(NAMES[s[2:-1]], (x, q), fontsize=6, xytext=(4 if left else -4, 5), textcoords='offset points',
                    ha='left' if left else 'right')
    ax.set_xscale('log')
    ax.set_xlabel('FLOPs per episode')
    ax.set_ylabel('tasks solved (macro)')
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.5, handletextpad=0.2)
    save(fig, 'frontier')


def protocol() -> None:
    """F4: how often a prompted planner breaks its protocol, by its size."""

    rows = load('c5')['protocol']
    fig, ax = plt.subplots(figsize=(3.4, 1.7))
    names = list(rows)
    rates = [rows[n]['violation_rate'] for n in names]
    ax.barh(range(len(names)), rates, color=[ORANGE, GREY, BLUE], height=0.55, zorder=3)
    for i, r in enumerate(rates):
        ax.text(r + 0.01, i, f'{r:.0%}', va='center', fontsize=7)
    ax.set_yticks(range(len(names)), [f"{NAMES[n]} ({rows[n]['active_b']:g}B)" for n in names])
    ax.set_xlim(0, 1.08)
    ax.set_xlabel('planner turns that break the protocol')
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    save(fig, 'protocol')


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for figure in (replaceability, levelling, frontier, protocol):
        figure()
        print(figure.__name__)


if __name__ == '__main__':
    main()
