"""Run a model over tracks × conditions and append episodes to a JSONL file.

    python -m ear.run --model qwen3.5-0.8b --base-url http://localhost:8000/v1 --active-b 0.87 \
        --tracks code,tools,shell --conditions base,skill,watchdog --out out/qwen3.5-0.8b.jsonl

Resumable: episodes already in --out (same task, condition, repeat) are skipped.
"""

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ear import delegate, harness
from ear.llm import Model
from ear.tracks import base, code, multiturn, shell, tools

ENVS = {'code': code.Env, 'tools': tools.Env, 'shell': shell.Env, **{t: multiturn.Env for t in multiturn.CATEGORIES}}
STEPS = {'code': 8, 'tools': 4, 'shell': 10, **{t: 10 for t in multiturn.CATEGORIES}}  # multi-turn: per request
SKILLS = Path(__file__).resolve().parent.parent / 'skills'


def skill(track: str) -> dict:
    text = (SKILLS / track / 'SKILL.md').read_text()
    _, front, body = text.split('---', 2)
    meta = dict(line.split(':', 1) for line in front.strip().splitlines())
    return {'name': meta['name'].strip(), 'description': meta['description'].strip(), 'body': body.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--label', help='name in the results (default: --model)')
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--api-key', default='')
    parser.add_argument('--active-b', type=float, required=True, help='active parameters, billions')
    parser.add_argument('--tracks', default='code,tools,shell')
    parser.add_argument('--conditions', default='base')
    parser.add_argument('--splits', default='screen,train,test')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0, help='tasks per track (0 = all)')
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--max-tokens', type=int, default=3072)
    parser.add_argument('--token-budget', type=int, default=24000)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--no-think', action='store_true', help='chat_template_kwargs enable_thinking=false')
    parser.add_argument('--memory', help='JSON {task_id: worked examples} for the memory condition')
    parser.add_argument('--planner-model', help='delegate* conditions: the planner (PREREG-PD)')
    parser.add_argument('--planner-label')
    parser.add_argument('--planner-url')
    parser.add_argument('--planner-active-b', type=float, default=0)
    parser.add_argument('--planner-max-tokens', type=int, default=1024)
    parser.add_argument('--examples', help='JSON bank of worked calls per tool for the executor (data/memory/examples.json)')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    extra = {'chat_template_kwargs': {'enable_thinking': False}} if args.no_think else {}
    model = Model(args.model, args.base_url, args.api_key, args.active_b, args.max_tokens, args.temperature, extra)
    label = args.label or args.model
    memory = json.load(open(args.memory)) if args.memory else {}
    # the planner never thinks: its cost is the point of the design (PREREG-PD)
    planner = (Model(args.planner_model, args.planner_url or args.base_url, '', args.planner_active_b, args.planner_max_tokens,
                     args.temperature, {'chat_template_kwargs': {'enable_thinking': False}}) if args.planner_model else None)
    bank = json.load(open(args.examples)) if args.examples else {}
    splits = set(args.splits.split(','))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in open(out):
            e = json.loads(line)
            if e['failure'] != 'agent_failed':
                done.add((e['task_id'], e['condition'], e['repeat']))

    jobs = []
    for track in args.tracks.split(','):
        tasks = [t for t in base.load(track) if t.split in splits]
        if args.limit:
            tasks = random.Random(0).sample(tasks, min(args.limit, len(tasks)))
        for item in args.conditions.split(','):
            # `base@1` is one explicit repeat (the noise floor); a bare name takes --repeats
            condition, _, at = item.partition('@')
            repeats = [int(at)] if at else range(args.repeats)
            for r in repeats:
                jobs += [(t, condition, r) for t in tasks if (t.id, condition, r) not in done]
    random.Random(1).shuffle(jobs)
    print(f'{label}: {len(jobs)} episodes to run, {len(done)} done', flush=True)

    lock, counts, started = threading.Lock(), {'n': 0, 'passed': 0}, time.time()

    def one(task, condition, r):
        env = ENVS[task.track](task)
        if condition.startswith('delegate'):
            episode = delegate.run(task, env, planner, model, condition=condition, repeat=r, max_steps=STEPS[task.track],
                                   token_budget=args.token_budget, bank=bank)
            episode.model, episode.planner = label, args.planner_label or args.planner_model
            return episode
        episode = harness.run(
            task, env, model, condition=condition, repeat=r, max_steps=STEPS[task.track],
            token_budget=args.token_budget,
            skill=skill(task.track) if {'skill', 'skill_inline'} & set(condition.split('+')) else None,
            memory=memory.get(task.id, '') if 'memory' in condition.split('+') else '')
        episode.model = label
        return episode

    with ThreadPoolExecutor(args.workers) as pool, open(out, 'a') as sink:
        futures = [pool.submit(one, *job) for job in jobs]
        for future in as_completed(futures):
            try:
                episode = future.result()
            except Exception as error:  # noqa: BLE001 - one broken episode must not stop the run
                print(f'episode crashed: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
                continue
            with lock:
                sink.write(harness.to_json(episode) + '\n')
                sink.flush()
                counts['n'] += 1
                counts['passed'] += episode.passed
                if counts['n'] % 50 == 0 or counts['n'] == len(jobs):
                    rate = counts['n'] / (time.time() - started)
                    print(f"{label}: {counts['n']}/{len(jobs)} pass {counts['passed'] / counts['n']:.3f} "
                          f'{rate * 60:.0f}/min', flush=True)


if __name__ == '__main__':
    main()
