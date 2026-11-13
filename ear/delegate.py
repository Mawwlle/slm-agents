"""Plan & delegate (stage 3, PREREG-PD): a planner that only plans and checks, an executor that does one step.

Closed loop (`delegate`): the planner sees the request, every tool and what each step did, and answers with one
JSON object, `{"step": ..., "tools": [...]}` or `{"finish": ...}`. The executor gets only that step, only the tools
it names and up to two worked calls of those tools, and has at most three turns (call → result → correction).
Open loop (`delegate_open`): the planner writes every step at once and nobody checks. `+noex` drops the examples.

The episode is logged like a solo one; steps carry `role` (planner / executor), FLOPs are split.
"""

import json
import re
import time

from ear.harness import OBSERVATION_CHARS, PROTOCOL, Episode, Step, classify, describe, signature
from ear.llm import Model, Reply, parse_calls
from ear.tracks.multiturn import CATEGORIES

EXECUTOR_TURNS = 3
EXAMPLES = 2

PLANNER = """You plan and check; an assistant does the work, one step at a time. The assistant sees only the step
you write and the tools you name: not the user's request, not earlier steps or results. So each step is one concrete
action that carries every value it needs (names, paths, numbers, ids, text), copied from the request or from
earlier results.

Answer with exactly one JSON object and nothing else:
{{"step": "<one concrete action with its exact values>", "tools": ["<tool name>", ...]}}
{{"finish": "<the final answer or a short reply to the user>"}}
Use finish when the request is done or cannot be done. After each step you see the assistant's calls and their
results; if a step went wrong, write a corrected one.

Tools:
{tools}"""

PLANNER_OPEN = """You plan; an assistant does the work, one step at a time. The assistant sees only the step you write and
the tools you name: not the user's request and not the results of other steps. So each step is one concrete action
that carries every value it needs (names, paths, numbers, ids, text). You will not see the results.

Answer with exactly one JSON object and nothing else:
{{"steps": [{{"step": "<one concrete action with its exact values>", "tools": ["<tool name>", ...]}}, ...],
  "finish": "<the final answer or a short reply to the user, if you can give it without the results>"}}

Tools:
{tools}"""

EXECUTOR_DONE = {'name': 'finish', 'description': 'Call when the step is done.', 'parameters': {}}
OBJECT = re.compile(r'\{.*\}', re.S)


def decision(text: str) -> dict | None:
    """The planner's JSON object, tolerating prose or a fence around it."""

    match = OBJECT.search(text)
    if not match:
        return None
    for candidate in (match.group(0), match.group(0).replace("'", '"')):
        try:
            value = json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def examples(bank: dict, task, tools: list[str]) -> str:
    """Up to two worked calls of the step's tools, from the teacher's passed train episodes of the same track."""

    picked = []
    track = 'multiturn' if task.track in CATEGORIES else task.track
    for round_ in range(EXAMPLES):
        for name in tools:
            calls = [c for c in bank.get(track, {}).get(name, []) if c['task_id'] != task.id]
            if round_ < len(calls) and len(picked) < EXAMPLES:
                picked.append(calls[round_]['call'])
    return '\n'.join(f'<tool_call>\n{c}\n</tool_call>' for c in picked)


class Delegation:
    def __init__(self, task, env, planner: Model, executor: Model, episode: Episode, bank: dict, *,
                 examples_on: bool, max_steps: int, token_budget: int) -> None:
        self.task, self.env, self.planner, self.executor, self.episode = task, env, planner, executor, episode
        self.bank, self.examples_on, self.max_steps, self.token_budget = bank, examples_on, max_steps, token_budget
        self.tools = {t['name']: t for t in env.tools}
        self.used = self.index = self.turn = 0

    def account(self, model: Model, reply: Reply, role: str) -> Step:
        self.used += reply.completion_tokens
        flops = 2 * model.active_b * 1e9 * (reply.prompt_tokens + reply.completion_tokens)
        self.episode.flops += flops
        if role == 'planner':
            self.episode.planner_flops += flops
        step = Step(self.index, model.name, reply.finish, reply.prompt_tokens, reply.completion_tokens,
                    reply.thinking_chars, reply.latency_s, reply.mean_logprob, reply.mean_entropy, reply.min_logprob,
                    reply.repetition, 0, False, [], repeat=False, tool_ok=None, progress=None, text=reply.text[:600],
                    turn=self.turn, role=role)
        self.index += 1
        self.episode.steps.append(step)
        return step

    def execute(self, instruction: str, names: list[str]) -> str:
        """One step by the executor; returns what the planner gets to see."""

        tools = [self.tools[n] for n in names if n in self.tools and n != 'finish'] or \
                [t for t in self.env.tools if t['name'] != 'finish']
        system = PROTOCOL.format(tools=describe(tools + [EXECUTOR_DONE]))
        shown = examples(self.bank, self.task, [t['name'] for t in tools]) if self.examples_on else ''
        if shown:
            system += f'\n\nExamples of calls to these tools:\n{shown}'
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': instruction}]
        report, previous = [], None
        for _ in range(EXECUTOR_TURNS):
            reply = self.executor.chat(messages)
            if reply.finish == 'error':
                self.episode.failure, self.episode.error = 'agent_failed', reply.error
                return '\n'.join(report)
            step = self.account(self.executor, reply, 'executor')
            calls, broken, recovered = parse_calls(reply.text, [t['name'] for t in tools] + ['finish'])
            step.n_calls, step.broken, step.recovered = len(calls), broken, recovered
            step.names = [c['name'] for c in calls]
            step.repeat = calls != [] and signature(calls) == previous
            previous = signature(calls) if calls else previous
            results, verdicts, done = [], [], False
            for call in calls[:4]:
                if call['name'] == 'finish':
                    done = True
                    break
                outcome = self.env.call(call['name'], call['arguments'])
                results.append(outcome.text[:OBSERVATION_CHARS])
                report.append(f'call {json.dumps(call, ensure_ascii=False)}\nresult {outcome.text[:OBSERVATION_CHARS]}')
                if outcome.ok is not None:
                    verdicts.append(outcome.ok)
                if outcome.progress is not None:
                    step.progress = outcome.progress
            step.tool_ok = sum(verdicts) / len(verdicts) if verdicts else None
            if done or (not calls and not broken):
                if not calls and reply.text.strip():
                    report.append(f'the assistant said: {reply.text.strip()[:1000]}')
                break
            observation = ('Your tool call could not be parsed. Write exactly <tool_call>{"name": ..., "arguments": {...}}</tool_call>.'
                           if broken else '\n'.join(f'<tool_response>\n{r}\n</tool_response>' for r in results))
            messages += [{'role': 'assistant', 'content': reply.text or '(empty)'},
                         {'role': 'user', 'content': observation + '\n\nIf the step is done, call finish.'}]
        return '\n'.join(report) or 'the assistant made no call'

    def finish_turn(self, answer: str) -> str | None:
        self.env.finish({'answer': answer})
        self.episode.answer = answer[:4000]
        follow = getattr(self.env, 'next_prompt', lambda: None)()
        if follow:
            self.turn, self.used = self.turn + 1, 0
        return follow

    def closed(self) -> None:
        system = PLANNER.format(tools=describe(self.env.tools))
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': self.task.prompt}]
        planned = 0
        while planned < self.max_steps and self.used < self.token_budget:
            reply = self.planner.chat(messages)
            if reply.finish == 'error':
                self.episode.failure, self.episode.error = 'agent_failed', reply.error
                return
            self.account(self.planner, reply, 'planner')
            planned += 1
            choice = decision(reply.text) or {}
            if 'finish' in choice:
                follow = self.finish_turn(str(choice['finish']))
                if not follow:
                    self.episode.done = True
                    return
                messages += [{'role': 'assistant', 'content': reply.text}, {'role': 'user', 'content': follow}]
                planned = 0
                continue
            if 'step' not in choice:
                feedback = 'Answer with exactly one JSON object: {"step": ..., "tools": [...]} or {"finish": ...}.'
            else:
                names = choice.get('tools') if isinstance(choice.get('tools'), list) else []
                feedback = 'Step result:\n' + self.execute(str(choice['step']), [str(n) for n in names])
                if self.episode.failure == 'agent_failed':
                    return
            messages += [{'role': 'assistant', 'content': reply.text}, {'role': 'user', 'content': feedback}]

    def open(self) -> None:
        messages = [{'role': 'system', 'content': PLANNER_OPEN.format(tools=describe(self.env.tools))}]
        prompt = self.task.prompt
        while prompt:
            # a later request sees the earlier requests and plans, never their results
            messages.append({'role': 'user', 'content': prompt})
            reply = self.planner.chat(messages)
            if reply.finish == 'error':
                self.episode.failure, self.episode.error = 'agent_failed', reply.error
                return
            self.account(self.planner, reply, 'planner')
            plan = decision(reply.text) or {}
            steps = plan.get('steps') if isinstance(plan.get('steps'), list) else []
            for item in steps[:self.max_steps]:
                if not isinstance(item, dict) or 'step' not in item:
                    continue
                names = item.get('tools') if isinstance(item.get('tools'), list) else []
                self.execute(str(item['step']), [str(n) for n in names])
                if self.episode.failure == 'agent_failed' or self.used >= self.token_budget:
                    return
            messages.append({'role': 'assistant', 'content': reply.text})
            prompt = self.finish_turn(str(plan.get('finish', '')))
        self.episode.done = True


def run(task, env, planner: Model, executor: Model, *, condition: str, repeat: int, max_steps: int,
        token_budget: int, bank: dict) -> Episode:
    episode = Episode(task.id, task.track, task.split, executor.name, condition, repeat, planner=planner.name)
    started = time.time()
    parts = set(condition.split('+'))
    delegation = Delegation(task, env, planner, executor, episode, bank, examples_on='noex' not in parts,
                            max_steps=max_steps, token_budget=token_budget)
    delegation.open() if 'delegate_open' in parts else delegation.closed()
    episode.score = env.grade()
    episode.passed = episode.score >= task.pass_at
    episode.seconds = time.time() - started
    if not episode.failure:
        episode.failure = classify(episode, delegation.used >= token_budget)
    env.close()
    return episode
