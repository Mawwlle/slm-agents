"""BFCL v3 (simple, multiple, parallel, parallel_multiple) as a tool-use episode.

The task's functions are the agent's tools. A call is recorded, checked against the
function's schema (the label-free verdict a harness sees), and graded at the end
against BFCL's possible answers with an AST-style matcher.
"""

import itertools
import json
import re

from ear.tracks.base import FINISH, Outcome, Task

TYPES = {'string': str, 'integer': int, 'float': (int, float), 'number': (int, float), 'boolean': bool,
         'array': list, 'tuple': list, 'dict': dict, 'object': dict, 'any': object}


def tool(function: dict) -> dict:
    parameters = function.get('parameters', {})
    required = set(parameters.get('required', []))
    arguments = {
        name: f"{spec.get('type', 'any')}{'' if name in required else ', optional'}: {spec.get('description', '')}"
              + (f" (one of {spec['enum']})" if 'enum' in spec else '')
        for name, spec in parameters.get('properties', {}).items()}
    return {'name': function['name'], 'description': function.get('description', ''), 'parameters': arguments}


class Env:
    def __init__(self, task: Task) -> None:
        self.task = task
        self.functions = {f['name']: f for f in task.payload['functions']}
        self.tools = [tool(f) for f in task.payload['functions']] + [FINISH]
        self.calls: list[dict] = []

    def call(self, name: str, arguments: dict) -> Outcome:
        function = self.functions.get(name) or self.functions.get(name.replace('_', '.'))
        if function is None:
            return Outcome(f'error: there is no function {name!r}', ok=0)
        problems = schema_problems(function, arguments)
        self.calls.append({'name': function['name'], 'arguments': arguments})
        if problems:
            return Outcome('recorded, but: ' + '; '.join(problems), ok=0)
        return Outcome('recorded. Make any other calls the request needs, then call finish.', ok=1)

    def finish(self, arguments: dict) -> None:
        pass

    def grade(self) -> float:
        return float(match(self.calls, self.task.payload['answer']))

    def close(self) -> None:
        pass


def schema_problems(function: dict, arguments: dict) -> list[str]:
    parameters = function.get('parameters', {})
    properties = parameters.get('properties', {})
    problems = [f'missing required `{name}`' for name in parameters.get('required', []) if name not in arguments]
    for name, value in arguments.items():
        spec = properties.get(name)
        if spec is None:
            problems.append(f'unknown parameter `{name}`')
            continue
        expected = TYPES.get(spec.get('type', 'any'), object)
        if not isinstance(value, expected) or (spec.get('type') == 'integer' and isinstance(value, bool)):
            problems.append(f'`{name}` should be {spec.get("type")}')
    return problems


def _norm(value):
    if isinstance(value, str):
        return re.sub(r"[\s,./\-_*^'\"]", '', value.lower())
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_norm(v) for v in value]
    if isinstance(value, dict):
        return {k: _norm(v) for k, v in value.items()}
    return value


def _value_ok(value, allowed: list) -> bool:
    target = _norm(value)
    for option in allowed:
        if isinstance(option, dict) and isinstance(value, dict):
            if all(k in option and _value_ok(v, option[k]) for k, v in value.items()) and all(
                    k in value or '' in (option[k] if isinstance(option[k], list) else [option[k]]) for k in option):
                return True
        elif _norm(option) == target:
            return True
    return False


def _call_ok(call: dict, expected: dict) -> bool:
    (name, parameters), = expected.items()
    if call['name'] != name:
        return False
    arguments = call['arguments']
    for key in arguments:
        if key not in parameters:
            return False
    for key, allowed in parameters.items():
        if key not in arguments:
            if '' not in allowed:
                return False
            continue
        if not _value_ok(arguments[key], allowed):
            return False
    return True


def match(calls: list[dict], answer: list[dict]) -> bool:
    if len(calls) != len(answer):
        return False
    if len(calls) > 6:
        return all(_call_ok(c, a) for c, a in zip(calls, answer))
    return any(all(_call_ok(c, a) for c, a in zip(order, answer)) for order in itertools.permutations(calls))


def build(raw_dir) -> list[dict]:
    """Rows for data/tasks/tools.jsonl from BFCL's question and possible_answer files."""

    rows = []
    for part in ('simple', 'multiple', 'parallel', 'parallel_multiple'):
        questions = [json.loads(line) for line in open(f'{raw_dir}/BFCL_v3_{part}.json')]
        answers = {r['id']: r['ground_truth'] for r in map(json.loads, open(f'{raw_dir}/possible_answer/BFCL_v3_{part}.json'))}
        if part == 'simple':
            questions = questions[::2]
        for q in questions:
            text = '\n'.join(m['content'] for m in q['question'][0] if m['role'] == 'user')
            rows.append({'id': f"tools/{q['id']}", 'prompt': text + '\n\nUse the tools to do what the request asks, then call finish.',
                         'payload': {'functions': q['function'], 'answer': answers[q['id']], 'part': part}})
    return rows
