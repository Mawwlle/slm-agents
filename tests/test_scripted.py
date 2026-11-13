"""A scripted model that answers each request with its ground-truth calls must pass, solo and as an executor."""

import ast
import json

from ear import delegate, harness
from ear.llm import Reply
from ear.tracks import base, multiturn


def as_call(source: str) -> dict:
    node = ast.parse(source, mode='eval').body
    return {'name': ast.unparse(node.func), 'arguments': {k.arg: ast.literal_eval(k.value) for k in node.keywords}}


def reply(text: str) -> Reply:
    return Reply(text, 0, 'stop', 10, 5, 0.0, None, None, None)


def keyword_only_task():
    return next(t for t in base.load('multiturn')
                if all(calls and not ast.parse(c, mode='eval').body.args for calls in t.payload['answer'] for c in calls))


class Solver:
    """Each request: all its ground-truth calls in one reply, then finish."""

    name, active_b = 'solver', 1.0

    def __init__(self, answer):
        self.answer, self.turn, self.called = answer, 0, False

    def chat(self, messages, **_):
        if self.called:
            self.called, self.turn = False, self.turn + 1
            return reply('<tool_call>\n{"name": "finish", "arguments": {}}\n</tool_call>')
        self.called = True
        return reply(''.join(f'<tool_call>\n{json.dumps(as_call(c))}\n</tool_call>' for c in self.answer[self.turn]))


class Planner:
    """Each request: one step naming its ground-truth calls, then finish."""

    name, active_b = 'planner', 1.0

    def __init__(self, answer):
        self.answer, self.turn, self.stepped = answer, 0, False

    def chat(self, messages, **_):
        if self.stepped:
            self.stepped, self.turn = False, self.turn + 1
            return reply('{"finish": "done"}')
        self.stepped = True
        names = sorted({as_call(c)['name'] for c in self.answer[self.turn]})
        return reply(json.dumps({'step': ' then '.join(self.answer[self.turn]), 'tools': names}))


def test_solo():
    task = keyword_only_task()
    assert harness.run(task, multiturn.Env(task), Solver(task.payload['answer']), max_steps=10).passed


def test_delegated():
    task = keyword_only_task()
    episode = delegate.run(task, multiturn.Env(task), Planner(task.payload['answer']), Solver(task.payload['answer']),
                           condition='delegate', repeat=0, max_steps=10, token_budget=24000, bank={})
    assert episode.passed and {s.role for s in episode.steps} == {'planner', 'executor'}
