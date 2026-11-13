import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

TASKS = Path(__file__).resolve().parents[2] / 'data' / 'tasks'


@dataclass
class Task:
    id: str
    track: str
    prompt: str
    payload: dict = field(default_factory=dict)
    pass_at: float = 1.0

    @property
    def split(self) -> str:
        """screen 15% · train 25% · test 60%, fixed by the id (PREREG)."""

        bucket = int(hashlib.sha256(self.id.encode()).hexdigest(), 16) % 100
        return 'screen' if bucket < 15 else 'train' if bucket < 40 else 'test'


@dataclass
class Outcome:
    text: str
    ok: float | None = None
    progress: float | None = None


def load(track: str) -> list[Task]:
    rows = [json.loads(line) for line in open(TASKS / f'{track}.jsonl')]
    return [Task(r['id'], track, r['prompt'], r.get('payload', {}), r.get('pass_at', 1.0)) for r in rows]


class Workdir:
    def __init__(self) -> None:
        self.path = Path(tempfile.mkdtemp(prefix='ear-'))

    def run(self, argv: list[str], timeout: float = 20, stdin: str | None = None) -> tuple[int, str]:
        env = {k: v for k, v in os.environ.items() if not k.startswith(('OPENAI', 'HF_TOKEN'))}
        try:
            done = subprocess.run(argv, cwd=self.path, input=stdin, capture_output=True, text=True,
                                  timeout=timeout, env=env)
            return done.returncode, (done.stdout + ('\n' + done.stderr if done.stderr else '')).strip()
        except subprocess.TimeoutExpired as error:
            out = (error.stdout or b'')
            out = out.decode(errors='replace') if isinstance(out, bytes) else out
            return 124, f'{out}\n[timed out after {timeout:.0f}s]'.strip()

    def python(self, code: str, timeout: float = 20) -> tuple[int, str]:
        script = self.path / f'_snippet_{abs(hash(code)) % 10**8}.py'
        script.write_text(code)
        try:
            return self.run([sys.executable, '-E', '-s', script.name], timeout)
        finally:
            script.unlink(missing_ok=True)

    def close(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


FINISH = {'name': 'finish', 'description': 'End the task. Put the final answer in `answer` if the task asks for one.',
          'parameters': {'answer': 'string, optional'}}
