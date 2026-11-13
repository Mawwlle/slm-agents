"""Small multi-step shell tasks.

Each task is generated from a family and a seed: files go into the episode's
working directory, and the check reads either the final answer or the files.
"""

import csv
import io
import json
import random
import sys
from datetime import date, timedelta

from ear.tracks.base import FINISH, Outcome, Task, Workdir

TOOLS = [
    {'name': 'bash', 'description': 'Run a shell command in the task directory; returns the exit code and output.',
     'parameters': {'command': 'string'}},
    {'name': 'write_file', 'description': 'Write a text file (relative path in the task directory).',
     'parameters': {'path': 'string', 'content': 'string'}},
    FINISH,
]

WORDS = ('alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa '
         'quebec romeo sierra tango uniform victor whiskey xray yankee zulu').split()
STOP = ['the', 'a', 'of', 'and', 'to', 'in']


# ---- families: (files, prompt, check spec) -------------------------------------------------

def count_errors(r):
    codes = ['E1', 'E2', 'E3', 'E4']
    target = r.choice(codes)
    lines, n = [], 0
    for i in range(r.randint(300, 900)):
        level = r.choice(['INFO', 'INFO', 'WARN', 'ERROR'])
        code = r.choice(codes)
        n += level == 'ERROR' and code == target
        lines.append(f'2026-10-{r.randint(1, 28):02d}T{r.randint(0, 23):02d}:{r.randint(0, 59):02d} {level} [{code}] event {i}')
    return ({'logs/app.log': '\n'.join(lines) + '\n'},
            f'How many ERROR lines with code [{target}] are in logs/app.log? Answer with the number.',
            {'answer': str(n)})


def csv_sum(r):
    regions = ['north', 'south', 'east', 'west']
    target = r.choice(regions)
    out, total = io.StringIO(), 0.0
    writer = csv.writer(out)
    writer.writerow(['date', 'region', 'amount'])
    for _ in range(r.randint(100, 400)):
        region, amount = r.choice(regions), round(r.uniform(1, 500), 2)
        total += amount if region == target else 0
        writer.writerow([f'2026-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}', region, amount])
    return ({'sales.csv': out.getvalue()},
            f'What is the total `amount` for region "{target}" in sales.csv? Answer rounded to 2 decimals.',
            {'answer': f'{total:.2f}', 'number': True})


def top_word(r):
    weights = {w: r.randint(1, 30) for w in r.sample(WORDS, 12)}
    winner = max(weights, key=weights.get)
    weights[winner] += 5
    text = []
    for word, count in weights.items():
        text += [word] * count + r.choices(STOP, k=count * 2)
    r.shuffle(text)
    lines = [' '.join(text[i:i + 12]) for i in range(0, len(text), 12)]
    return ({'notes/text.txt': '\n'.join(lines) + '\n'},
            f'Which word is most frequent in notes/text.txt, ignoring these stop words: {", ".join(STOP)}? Answer with the word.',
            {'answer': winner})


BUGS = [
    ('def mean(xs):\n    return sum(xs) / (len(xs) - 1)\n', 'assert mean([2, 4, 6]) == 4\nassert mean([5]) == 5\n'),
    ('def last(xs):\n    return xs[len(xs)]\n', 'assert last([1, 2, 3]) == 3\nassert last(["a"]) == "a"\n'),
    ('def is_even(n):\n    return n % 2 == 1\n', 'assert is_even(4)\nassert not is_even(7)\n'),
    ('def count_vowels(s):\n    return sum(1 for c in s if c in "aeiou")\n', 'assert count_vowels("Apple") == 2\nassert count_vowels("xyz") == 0\n'),
    ('def clamp(x, lo, hi):\n    return max(hi, min(lo, x))\n', 'assert clamp(5, 0, 10) == 5\nassert clamp(-1, 0, 10) == 0\nassert clamp(11, 0, 10) == 10\n'),
    ('def fizz(n):\n    if n % 3 == 0:\n        return "Fizz"\n    if n % 15 == 0:\n        return "FizzBuzz"\n    if n % 5 == 0:\n        return "Buzz"\n    return str(n)\n',
     'assert fizz(15) == "FizzBuzz"\nassert fizz(9) == "Fizz"\nassert fizz(10) == "Buzz"\nassert fizz(7) == "7"\n'),
    ('def word_count(s):\n    return len(s.split(" "))\n', 'assert word_count("a  b") == 2\nassert word_count(" hi there ") == 2\n'),
    ('def reverse_words(s):\n    return " ".join(s.split()[::1])\n', 'assert reverse_words("a b c") == "c b a"\n'),
]


def fix_bug(r):
    code, tests = BUGS[r.randrange(len(BUGS))]
    return ({'lib/util.py': code, 'tests/test_util.py': 'import sys\nsys.path.insert(0, "lib")\nfrom util import *\n' + tests + 'print("ok")\n'},
            'The tests in tests/test_util.py fail. Fix the bug in lib/util.py (do not edit the tests) so that '
            '`python3 tests/test_util.py` prints ok.',
            {'run': 'tests/test_util.py', 'expect': 'ok', 'tests': tests})


def json_filter(r):
    users = [{'name': f'{r.choice(WORDS).title()}{i}', 'age': r.randint(16, 70), 'active': r.random() < 0.6} for i in range(40)]
    age = r.choice([25, 30, 35, 40])
    expected = sorted(u['name'] for u in users if u['age'] > age and u['active'])
    return ({'data/users.json': json.dumps(users, indent=1)},
            f'Write out.json: a JSON list of the names of active users older than {age} in data/users.json, sorted alphabetically.',
            {'file': 'out.json', 'json': expected})


def largest_file(r):
    files, best = {}, ('', -1)
    for i in range(r.randint(10, 25)):
        path = f'{r.choice(["docs", "docs/old", "src", "misc"])}/{r.choice(WORDS)}_{i}.{r.choice(["txt", "txt", "log", "md"])}'
        size = r.randint(10, 5000)
        files[path] = 'x' * size
        if path.endswith('.txt') and size > best[1]:
            best = (path, size)
    return (files, 'Which .txt file in this directory tree is the largest? Answer with its relative path.',
            {'answer': best[0]})


def primes(r):
    n = r.choice([100, 250, 500, 1000, 2000])
    expected = [p for p in range(2, n) if all(p % d for d in range(2, int(p ** 0.5) + 1))]
    return ({}, f'Write primes.txt with every prime number below {n}, one per line, in increasing order.',
            {'file': 'primes.txt', 'lines': [str(p) for p in expected]})


def dedupe(r):
    base = [f'{r.choice(WORDS)}.{r.choice(WORDS)}{r.randint(1, 99)}@example.com' for _ in range(30)]
    noisy = [r.choice([e, e.upper(), e.title()]) for e in base for _ in range(r.randint(1, 3))]
    r.shuffle(noisy)
    return ({'emails.txt': '\n'.join(noisy) + '\n'},
            'emails.txt has duplicate addresses that differ only in case. Write unique.txt: each address once, '
            'lowercased, sorted, one per line.',
            {'file': 'unique.txt', 'lines': sorted({e.lower() for e in base})})


def date_span(r):
    kinds = ['deploy', 'incident', 'review']
    target = r.choice(kinds)
    start = date(2026, 1, 1)
    rows, seen = ['date,kind'], []
    for _ in range(r.randint(20, 60)):
        day, kind = start + timedelta(days=r.randint(0, 300)), r.choice(kinds)
        rows.append(f'{day.isoformat()},{kind}')
        if kind == target:
            seen.append(day)
    span = (max(seen) - min(seen)).days if seen else 0
    return ({'events.csv': '\n'.join(rows) + '\n'},
            f'How many days are there between the first and the last "{target}" event in events.csv? Answer with the number.',
            {'answer': str(span)})


def rename(r):
    names = [f'IMG_{r.randint(1000, 9999)}' for _ in range(r.randint(5, 12))]
    files = {f'photos/{n}.JPG': 'jpeg' for n in names} | {'photos/keep.txt': 'keep'}
    return (files, 'In photos/, rename every file ending in .JPG so that it ends in .jpg instead (keep the rest of the name). '
            'Leave other files alone.',
            {'tree': sorted([f'photos/{n}.jpg' for n in names] + ['photos/keep.txt'])})


FAMILIES = [count_errors, csv_sum, top_word, fix_bug, json_filter, largest_file, primes, dedupe, date_span, rename]


def build(per_family: int = 5) -> list[dict]:
    rows = []
    for family in FAMILIES:
        for seed in range(per_family):
            files, text, check = family(random.Random(f'{family.__name__}-{seed}'))
            rows.append({'id': f'shell/{family.__name__}-{seed}',
                         'prompt': text + '\n\nWork in the current directory with the tools; call finish with the answer when done.',
                         'payload': {'files': files, 'check': check}})
    return rows


class Env:
    tools = TOOLS

    def __init__(self, task: Task) -> None:
        self.task = task
        self.work = Workdir()
        for path, content in task.payload['files'].items():
            target = self.work.path / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        self.answer = ''

    def call(self, name: str, arguments: dict) -> Outcome:
        if name == 'bash':
            status, out = self.work.run(['bash', '-c', str(arguments.get('command', ''))], timeout=30)
            return Outcome(f'exit {status}\n{out}', ok=float(status == 0))
        if name == 'write_file':
            path = str(arguments.get('path', ''))
            target = (self.work.path / path).resolve()
            if not path or self.work.path.resolve() not in target.parents:
                return Outcome('error: give a relative path inside the task directory', ok=0)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(arguments.get('content', '')))
            return Outcome(f'wrote {path}', ok=1)
        return Outcome(f'error: unknown tool {name!r}', ok=0)

    def finish(self, arguments: dict) -> None:
        self.answer = str(arguments.get('answer', '') or '')

    def grade(self) -> float:
        check, root = self.task.payload['check'], self.work.path
        if 'answer' in check:
            given = self.answer.strip().strip('`"\'.').strip()
            if check.get('number'):
                try:
                    return float(abs(float(given.replace(',', '')) - float(check['answer'])) < 0.011)
                except ValueError:
                    return 0.0
            return float(given.lower() == check['answer'].lower() or given.lower().lstrip('./') == check['answer'].lower())
        if 'run' in check:
            original = (root / 'tests/test_util.py')
            original.write_text('import sys\nsys.path.insert(0, "lib")\nfrom util import *\n' + check['tests'] + 'print("ok")\n')
            status, out = self.work.run([sys.executable, '-I', check['run']], timeout=15)
            return float(status == 0 and out.strip().endswith(check['expect']))
        if 'file' in check:
            target = root / check['file']
            if not target.exists():
                return 0.0
            text = target.read_text()
            if 'json' in check:
                try:
                    return float(json.loads(text) == check['json'])
                except json.JSONDecodeError:
                    return 0.0
            return float([line.strip() for line in text.strip().splitlines()] == check['lines'])
        if 'tree' in check:
            tree = sorted(str(p.relative_to(root)) for p in (root / 'photos').rglob('*') if p.is_file())
            return float(tree == check['tree'])
        return 0.0

    def close(self) -> None:
        self.work.close()
