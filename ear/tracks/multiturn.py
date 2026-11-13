"""BFCL v4 multi-turn base as a multi-turn tool-use episode.

The user's requests come one at a time: a request ends when the agent calls `finish` (or answers in plain text),
and the next one follows in the same conversation. The task's API classes are live simulators (ear.tracks.bfcl_api).
Graded as BFCL's checker does, without its global instance cache: after each request the simulators' public state
must equal the state the ground-truth calls leave, the ground truth's results must all appear among the agent's
results so far, and a request with ground-truth calls must get at least one call (one without, none). The score is the share of requests
passed before the first failure; an episode passes when all of them do.
"""

import ast
import copy
import importlib
import inspect
import json

from ear.tracks.base import TASKS, Outcome, Task
from ear.tracks.tools import tool

MODULES = {'GorillaFileSystem': 'gorilla_file_system', 'MathAPI': 'math_api', 'MessageAPI': 'message_api',
           'TwitterAPI': 'posting_api', 'TicketAPI': 'ticket_api', 'TradingBot': 'trading_bot',
           'TravelAPI': 'travel_booking', 'VehicleControlAPI': 'vehicle_control'}
STATELESS = {'MathAPI'}
# BFCL v4 multi-turn categories on the same simulators and grader, one track each (PREREG-CP); they share the
# multi-turn track's step examples, since their tools are the same
CATEGORIES = {'multiturn': 'base', 'multiturn_miss_param': 'miss_param', 'multiturn_long_context': 'long_context'}
FINISH = {'name': 'finish', 'description': 'Call when the current request is done. The user may then ask for more.',
          'parameters': {'answer': 'string, optional: a short reply to the user'}}


def instances(task: Task) -> dict:
    objects = {}
    for name in task.payload['classes']:
        instance = getattr(importlib.import_module(f'ear.tracks.bfcl_api.{MODULES[name]}'), name)()
        if name not in STATELESS:
            instance._load_scenario(copy.deepcopy(task.payload['config'].get(name, {})),
                                    long_context=task.payload.get('long_context', False))
        objects[name] = instance
    return objects


def methods(objects: dict) -> dict:
    return {name: method for instance in objects.values()
            for name, method in inspect.getmembers(instance, inspect.ismethod) if not name.startswith('_')}


def execute(method, args: list, kwargs: dict) -> str:
    """A call's result as BFCL renders it: a string as is, a dict as JSON, anything else through str()."""

    try:
        result = method(*args, **kwargs)
    except Exception as error:  # noqa: BLE001 - the simulators raise freely on bad arguments
        return f'Error during execution: {error}'
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        try:
            return json.dumps(result)
        except (TypeError, ValueError):
            return str(result)
    return str(result)


def replay(table: dict, call: str) -> str:
    """Run one ground-truth call written as Python (`mv(source='a', destination='b')`)."""

    node = ast.parse(call, mode='eval').body
    assert isinstance(node, ast.Call), call
    args = [ast.literal_eval(a) for a in node.args]
    kwargs = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
    return execute(table[ast.unparse(node.func)], args, kwargs)


def snapshot(objects: dict) -> dict:
    return {name: {k: copy.deepcopy(v) for k, v in vars(o).items() if not k.startswith('_')} for name, o in objects.items()}


def contains(results: list[str], expected: list[str]) -> bool:
    pool = list(results)
    for item in expected:
        if item not in pool:
            return False
        pool.remove(item)
    return True


class Env:
    def __init__(self, task: Task) -> None:
        self.task = task
        self.objects = instances(task)
        self.methods = methods(self.objects)
        docs = json.loads((TASKS / 'multiturn_docs.json').read_text())
        self.tools = [tool(d) for name in task.payload['classes'] for d in docs[name]] + [FINISH]
        self.results: list[list[str]] = [[]]
        self.states: list[dict] = []

    def call(self, name: str, arguments: dict) -> Outcome:
        method = self.methods.get(name)
        if method is None:
            return Outcome(f'error: there is no function {name!r}', ok=0)
        text = execute(method, [], arguments)
        self.results[-1].append(text)
        return Outcome(text, ok=0 if text.startswith(('Error during execution', '{"error"')) else 1)

    def finish(self, arguments: dict) -> None:
        self.states.append(snapshot(self.objects))
        self.results.append([])

    def next_prompt(self) -> str | None:
        turns = self.task.payload['turns']
        return turns[len(self.states)] if len(self.states) < len(turns) else None

    def grade(self) -> float:
        truth = instances(self.task)
        table, seen = methods(truth), []
        answer = self.task.payload['answer']
        for turn, calls in enumerate(answer):
            if turn >= len(self.states):
                return turn / len(answer)
            expected = [replay(table, c) for c in calls]
            seen += self.results[turn]
            state = self.states[turn]
            same = all(state[name].get(k) == v for name, o in truth.items()
                       for k, v in vars(o).items() if not k.startswith('_'))
            # a request with ground-truth calls needs a call; one without (cannot be done) must get none
            if bool(calls) != bool(self.results[turn]) or not same or not contains(seen, expected):
                return turn / len(answer)
        return 1.0

    def close(self) -> None:
        pass


def build(package_dir, track: str = 'multiturn') -> tuple[list[dict], dict]:
    """Rows for data/tasks/<track>.jsonl and the function docs per class, from an unpacked bfcl_eval package."""

    category = CATEGORIES[track]
    questions = [json.loads(line) for line in open(f'{package_dir}/data/BFCL_v4_multi_turn_{category}.json')]
    answers = {r['id']: r['ground_truth'] for r in map(json.loads, open(f'{package_dir}/data/possible_answer/BFCL_v4_multi_turn_{category}.json'))}
    docs = {}
    for name, module in MODULES.items():
        docs[name] = [json.loads(line) for line in open(f'{package_dir}/data/multi_turn_func_doc/{module}.json')]
    rows = []
    for q in questions:
        turns = ['\n'.join(m['content'] for m in turn if m['role'] == 'user') for turn in q['question']]
        turns = [t + '\n\nUse the tools to do what the request asks, then call finish.' for t in turns]
        payload = {'turns': turns, 'classes': q['involved_classes'], 'config': q['initial_config'], 'answer': answers[q['id']]}
        if category == 'long_context':
            payload['long_context'] = True
        rows.append({'id': f"{track}/{q['id']}", 'prompt': turns[0], 'payload': payload})
    return rows, docs
