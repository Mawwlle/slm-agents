"""An episode: a model works on one task through tools, step by step.

Every step is logged with what a router could see without labels (call validity,
repeats, the tool's verdict, logprobs, tokens), so the analysis can ask at step k
whether the episode will fail, and which lever would have saved it.

Levers (``condition``):
- ``base``      — the task and the tools;
- ``skill``     — a per-track skill listed by name and description, loaded on
                  demand with ``read_skill`` (progressive disclosure, as in
                  Agent Skills);
- ``memory``    — worked examples of a strong model on similar train tasks;
- ``watchdog``  — the harness interrupts repeats, broken calls and runaway
                  generations with a nudge;
- ``skill_inline`` — the skill's full text in the system prompt, no ``read_skill``;
- ``strict``    — after ``<tool_call>`` the decoder is held to the tools' JSON schema
                  (vLLM structural tag), so a started call is always well-formed.

Components combine with ``+`` (``strict+skill_inline+memory``).
"""

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ear.llm import PARSER_VERSION, Model, Reply, parse_calls

ROOT = Path(__file__).resolve().parent.parent
OBSERVATION_CHARS = 2500
# under the grammar some models (Granite 4) spell their end-of-sequence as text and run on: stop there
STRICT = {'stop': ['</s>', '<|end_of_text|>', '<|endoftext|>', '<|im_end|>']}  # the API takes at most 4
JSON_TYPES = {'string': 'string', 'integer': 'integer', 'float': 'number', 'number': 'number', 'boolean': 'boolean',
              'array': 'array', 'tuple': 'array', 'list': 'array', 'dict': 'object', 'object': 'object'}

PROTOCOL = """You are an agent that solves the task by calling tools.
To call a tool, write exactly one block per call:
<tool_call>
{{"name": "<tool name>", "arguments": {{...}}}}
</tool_call>
You may write a short sentence of reasoning before the call. After each call you get the
result in <tool_response>. When the task is complete, call `finish`.

Tools:
{tools}"""

SKILL_TOOL = {
    'name': 'read_skill',
    'description': 'Load the full instructions of a skill listed under "Skills".',
    'parameters': {'name': 'string, the skill name'},
}

NUDGES = {
    'repeat': 'You have made the same call several times with the same result. Change your approach.',
    'broken': 'Your tool call could not be parsed. Write exactly <tool_call>{"name": ..., "arguments": {...}}</tool_call> and nothing else after it.',
    'runaway': 'Your last answer was cut off because it was too long. Think briefly and call a tool now.',
}


@dataclass
class Step:
    index: int
    model: str
    finish: str
    prompt_tokens: int
    completion_tokens: int
    thinking_chars: int
    latency_s: float
    mean_logprob: float | None
    mean_entropy: float | None
    min_logprob: float | None
    repetition: float
    n_calls: int
    broken: bool
    names: list[str]
    repeat: bool          # same calls as the previous step
    tool_ok: float | None # the tools' own verdict: 1 ok, 0 error, a share for tests
    progress: float | None
    nudge: str = ''
    skill_read: bool = False
    text: str = ''        # the visible output, cut (for failure examples)
    recovered: bool = False  # a malformed call the parser rebuilt from the task's tool names
    turn: int = 0         # the user's request it serves (multi-turn tracks)
    role: str = ''        # planner / executor under delegation (ear.delegate)


@dataclass
class Episode:
    task_id: str
    track: str
    split: str
    model: str
    condition: str
    repeat: int
    score: float = 0.0
    passed: bool = False
    failure: str = ''
    done: bool = False
    steps: list[Step] = field(default_factory=list)
    flops: float = 0.0
    seconds: float = 0.0
    skill_read: bool = False
    answer: str = ''
    error: str = ''
    parser: int = PARSER_VERSION
    planner: str = ''     # under delegation: the planner; `model` is the executor
    planner_flops: float = 0.0


def describe(tools: list[dict]) -> str:
    return '\n'.join(
        f"- {t['name']}: {t['description']}\n  arguments: {json.dumps(t['parameters'], ensure_ascii=False)}"
        for t in tools)


def signature(calls: list[dict]) -> str:
    return json.dumps(calls, sort_keys=True, ensure_ascii=False)


def run(task, env, model: Model, *, condition: str = 'base', repeat: int = 0, max_steps: int = 8,
        token_budget: int = 24000, skill: dict | None = None, memory: str = '') -> Episode:
    episode = Episode(task.id, task.track, task.split, model.name, condition, repeat)
    started = time.time()

    parts = set(condition.split('+'))
    inline = skill if 'skill_inline' in parts else None
    skill = None if inline else skill
    tools = list(env.tools)
    system = PROTOCOL.format(tools=describe(tools + ([SKILL_TOOL] if skill else [])))
    if skill:
        system += f"\n\nSkills (load one with read_skill before you start, if it fits the task):\n- {skill['name']}: {skill['description']}"
    if inline:
        system += f"\n\nHow to do this kind of task ({inline['name']}):\n{inline['body']}"
    if memory:
        system += f'\n\nWorked examples of similar tasks solved well:\n{memory}'
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': task.prompt}]

    previous, used, index, turn, turn_start = None, 0, 0, 0, 0
    next_prompt = getattr(env, 'next_prompt', lambda: None)
    while index - turn_start < max_steps:
        reply: Reply = model.chat(messages, **(STRICT | {'response_format': grammar(tools)} if 'strict' in parts else {}))
        used += reply.completion_tokens
        episode.flops += 2 * model.active_b * 1e9 * (reply.prompt_tokens + reply.completion_tokens)
        if reply.finish == 'error':
            episode.failure, episode.error = 'agent_failed', reply.error
            break

        known = [t['name'] for t in tools] + (['read_skill'] if skill else [])
        calls, broken, recovered = parse_calls(reply.text, known)
        step = Step(index, model.name, reply.finish, reply.prompt_tokens, reply.completion_tokens,
                    reply.thinking_chars, reply.latency_s, reply.mean_logprob, reply.mean_entropy,
                    reply.min_logprob, reply.repetition, len(calls), broken, [c['name'] for c in calls],
                    repeat=calls != [] and signature(calls) == previous, tool_ok=None, progress=None,
                    text=reply.text[:600], recovered=recovered, turn=turn)
        previous = signature(calls) if calls else previous

        results, verdicts = [], []
        for call in calls[:4]:
            if call['name'] == 'read_skill' and skill:
                step.skill_read = episode.skill_read = True
                results.append(skill['body'])
                continue
            if call['name'] == 'finish':
                episode.done = True
                episode.answer = json.dumps(call['arguments'], ensure_ascii=False)
                env.finish(call['arguments'])
                break
            outcome = env.call(call['name'], call['arguments'])
            results.append(outcome.text[:OBSERVATION_CHARS])
            if outcome.ok is not None:
                verdicts.append(outcome.ok)
            if outcome.progress is not None:
                step.progress = outcome.progress
        step.tool_ok = sum(verdicts) / len(verdicts) if verdicts else None

        if not calls and not broken and reply.finish != 'length':
            # a plain answer ends the episode: the env decides what it is worth
            episode.done = True
            episode.answer = reply.text[:4000]
            env.finish({'answer': reply.text})

        nudge = ''
        if not episode.done:
            if broken:
                observation = NUDGES['broken']
            elif reply.finish == 'length' and not calls:
                observation = NUDGES['runaway'] if 'watchdog' in parts else 'Continue.'
            else:
                observation = '\n'.join(f'<tool_response>\n{r}\n</tool_response>' for r in results) or 'No result.'
            if 'watchdog' in parts:
                nudge = _watchdog(episode.steps + [step])
                if nudge:
                    observation += '\n\n' + NUDGES[nudge]
            step.nudge = nudge
            messages += [{'role': 'assistant', 'content': reply.text or '(empty)'},
                         {'role': 'user', 'content': observation}]

        episode.steps.append(step)
        index += 1
        if episode.done and (follow := next_prompt()):
            # a multi-turn task: the user's next request continues the conversation, with fresh step and token limits
            observation = '\n'.join(f'<tool_response>\n{r}\n</tool_response>' for r in results)
            messages += [{'role': 'assistant', 'content': reply.text or '(empty)'},
                         {'role': 'user', 'content': f'{observation}\n\n{follow}'.strip()}]
            episode.done, turn, turn_start, used = False, turn + 1, index, 0
            continue
        if episode.done or used >= token_budget:
            break

    episode.score = env.grade()
    episode.passed = episode.score >= task.pass_at
    episode.seconds = time.time() - started
    if not episode.failure:
        episode.failure = classify(episode, used >= token_budget)
    env.close()
    return episode


def grammar(tools: list[dict]) -> dict:
    """A vLLM structural tag: free text, but a started call must fit a tool's schema.

    Two openings: `<tool_call>` and, for models whose `<tool_call>` is a special token cut from the text
    (Granite 4, MNN), the bare `{"name": "` they write instead."""

    structures = []
    for t in tools:
        schema = {'type': 'object', 'properties': {k: _schema(v) for k, v in t['parameters'].items()}}
        structures += [{'begin': f'<tool_call>\n{{"name": "{t["name"]}", "arguments": ', 'schema': schema, 'end': '}\n</tool_call>'},
                       {'begin': f'{{"name": "{t["name"]}", "arguments": ', 'schema': schema, 'end': '}'}]
    return {'type': 'structural_tag', 'structures': structures, 'triggers': ['<tool_call>', '{"name": "']}


def _schema(described) -> dict:
    """Tracks describe an argument as "type[, optional]: text"; the grammar keeps only the plain JSON type."""

    word = re.split(r'[,:(]', str(described), maxsplit=1)[0].strip().lower()
    word = JSON_TYPES.get(word, word)
    return {'type': word} if word in JSON_TYPES.values() else {}


def _watchdog(steps: list[Step]) -> str:
    last = steps[-1]
    if len(steps) >= 2 and last.repeat and steps[-2].repeat:
        return 'repeat'
    if last.broken and len(steps) >= 2 and steps[-2].broken:
        return 'broken'
    if last.finish == 'length':
        return 'runaway'
    return ''


def classify(episode: Episode, over_budget: bool) -> str:
    if episode.passed:
        return ''
    steps = episode.steps
    if not steps:
        return 'agent_failed'
    longest = run_length = 0
    for step in steps:
        run_length = run_length + 1 if step.repeat else 0
        longest = max(longest, run_length)
    runaway = [s for s in steps if s.finish == 'length']
    if longest >= 2 or (runaway and max(s.repetition for s in runaway) > 0.3):
        return 'loop'
    if runaway and (steps[-1].finish == 'length' or len(runaway) >= 2):
        return 'model_stopped'
    if all(s.n_calls == 0 for s in steps) and any(s.broken for s in steps):
        return 'broken_tool_call'
    if steps[-1].broken:
        return 'broken_tool_call'
    if not episode.done:
        return 'budget' if over_budget else 'timeout'
    return 'wrong_answer'


def to_json(episode: Episode) -> str:
    return json.dumps(asdict(episode), ensure_ascii=False)
