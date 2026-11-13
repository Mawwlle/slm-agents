"""MBPP as an agentic task: write the function, run the visible test, fix, finish.

The visible test is the first of the three; the grade runs all three.
"""

import re

from ear.tracks.base import FINISH, Outcome, Task, Workdir

TOOLS = [
    {'name': 'write_solution', 'description': 'Write the full content of solution.py (replaces it).',
     'parameters': {'code': 'string, Python source'}},
    {'name': 'run_tests', 'description': 'Run the visible test against solution.py.', 'parameters': {}},
    {'name': 'python', 'description': 'Run a Python snippet and see its output (solution.py is importable).',
     'parameters': {'code': 'string'}},
    FINISH,
]

FENCE = re.compile(r'```(?:python)?\n(.*?)```', re.S)


def prompt(text: str, test: str) -> str:
    return (f'{text}\n\nYour function must pass tests like this one:\n{test}\n\n'
            'Write it to solution.py with write_solution, check it with run_tests, then call finish.')


class Env:
    tools = TOOLS

    def __init__(self, task: Task) -> None:
        self.task = task
        self.work = Workdir()
        self.answer = ''

    def _tests(self, tests: list[str]) -> tuple[int, str]:
        setup = self.task.payload.get('setup', '')
        body = '\n'.join([setup, 'from solution import *', *tests, 'print("ALL TESTS PASSED")'])
        return self.work.python(body, timeout=15)

    def call(self, name: str, arguments: dict) -> Outcome:
        if name == 'write_solution':
            code = arguments.get('code')
            if not isinstance(code, str) or not code.strip():
                return Outcome('error: `code` must be a non-empty string', ok=0)
            (self.work.path / 'solution.py').write_text(code)
            status, out = self.work.python(code, timeout=10)
            return Outcome('written' if status == 0 else f'written, but it does not run:\n{out}', ok=float(status == 0))
        if name == 'run_tests':
            if not (self.work.path / 'solution.py').exists():
                return Outcome('error: solution.py does not exist yet', ok=0, progress=0)
            status, out = self._tests(self.task.payload['tests'][:1])
            passed = status == 0 and 'ALL TESTS PASSED' in out
            return Outcome('passed' if passed else f'failed:\n{out[-1500:]}', ok=float(passed), progress=float(passed))
        if name == 'python':
            status, out = self.work.python(str(arguments.get('code', '')), timeout=15)
            return Outcome(f'exit {status}\n{out}', ok=float(status == 0))
        return Outcome(f'error: unknown tool {name!r}', ok=0)

    def finish(self, arguments: dict) -> None:
        self.answer = str(arguments.get('answer', '') or '')

    def grade(self) -> float:
        solution = self.work.path / 'solution.py'
        if not solution.exists():
            blocks = FENCE.findall(self.answer)
            if not blocks:
                return 0.0
            solution.write_text(blocks[-1])
        status, out = self._tests(self.task.payload['tests'])
        return float(status == 0 and 'ALL TESTS PASSED' in out)

    def close(self) -> None:
        self.work.close()
