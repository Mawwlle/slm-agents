"""One chat call: text, finish reason, tokens, latency and logprob statistics.

The harness never uses the server's tool-call parser: every model gets the same
text protocol (``<tool_call>{json}</tool_call>``), so a broken call is measured
the same way for every model and server.
"""

import ast
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass

THINK = re.compile(r'<think>.*?(</think>|$)', re.S)


@dataclass
class Reply:
    text: str            # visible part, thinking removed
    thinking_chars: int
    finish: str          # stop | length | error
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    mean_logprob: float | None
    mean_entropy: float | None   # over top-k, a lower bound of the true entropy
    min_logprob: float | None
    error: str = ''
    repetition: float = 0.0  # share of repeated 8-word shingles in the whole output, thinking included


class Model:
    def __init__(self, name: str, base_url: str, api_key: str = '', active_b: float = 0.0,
                 max_tokens: int = 3072, temperature: float = 0.6, extra: dict | None = None):
        self.name = name
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY', 'none')
        self.active_b = active_b
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra = extra or {}

    def chat(self, messages: list[dict], max_tokens: int | None = None, response_format: dict | None = None,
             stop: list[str] | None = None) -> Reply:
        body = {
            'model': self.name,
            'messages': messages,
            'max_tokens': max_tokens or self.max_tokens,
            'temperature': self.temperature,
            'top_p': 0.95,
            'logprobs': True,
            'top_logprobs': 5,
            **self.extra,
        }
        if response_format:
            body['response_format'] = response_format
        if stop:
            body['stop'] = stop
        started = time.time()
        last = ''
        for attempt in range(4):
            try:
                request = urllib.request.Request(
                    f'{self.base_url}/chat/completions', json.dumps(body).encode(),
                    {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'})
                with urllib.request.urlopen(request, timeout=900) as response:
                    data = json.loads(response.read())
                return _reply(data, time.time() - started)
            except urllib.error.HTTPError as error:
                last = f'http {error.code}: {error.read()[:300]!r}'
                if error.code == 400:
                    break
            except Exception as error:  # noqa: BLE001 - network errors of any kind are retried
                last = f'{type(error).__name__}: {error}'
            time.sleep(5 * (attempt + 1))
        return Reply('', 0, 'error', 0, 0, time.time() - started, None, None, None, last)


def _reply(data: dict, latency: float) -> Reply:
    choice = data['choices'][0]
    message = choice.get('message') or {}
    raw = message.get('content') or ''
    reasoning = message.get('reasoning_content') or message.get('reasoning') or ''
    thinking = len(reasoning) + sum(len(m.group(0)) for m in THINK.finditer(raw))
    text = THINK.sub('', raw).strip()
    if '</think>' in text:  # templates that open the think block in the prompt
        thinking += len(text.split('</think>')[0])
        text = text.split('</think>', 1)[1].strip()
    usage = data.get('usage') or {}

    tokens = ((choice.get('logprobs') or {}).get('content')) or []
    logprobs = [t['logprob'] for t in tokens if t.get('logprob') is not None]
    entropies = []
    for token in tokens:
        top = [math.exp(t['logprob']) for t in token.get('top_logprobs') or [] if t.get('logprob') is not None]
        if top:
            entropies.append(-sum(p * math.log(p) for p in top if p > 0))

    return Reply(
        text=text,
        thinking_chars=thinking,
        finish=choice.get('finish_reason') or 'stop',
        prompt_tokens=usage.get('prompt_tokens', 0),
        completion_tokens=usage.get('completion_tokens', 0),
        latency_s=latency,
        mean_logprob=sum(logprobs) / len(logprobs) if logprobs else None,
        mean_entropy=sum(entropies) / len(entropies) if entropies else None,
        min_logprob=min(logprobs) if logprobs else None,
        repetition=repetition(reasoning + raw),
    )


def repetition(text: str, n: int = 8) -> float:
    words = text.split()
    if len(words) < 2 * n:
        return 0.0
    shingles = [' '.join(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1 - len(set(shingles)) / len(shingles)


CALL = re.compile(r'<tool_call>\s*(.*?)\s*(</tool_call>|$)', re.S)
FENCE = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.S)
FUNCTION = re.compile(r'<function=([\w.\-]+)>\s*(.*?)\s*(?:</function>|</tool_call>|$)', re.S)
PARAMETER = re.compile(r'<parameter=([\w.\-]+)>\n?(.*?)\n?</parameter>', re.S)


GEMMA = re.compile(r'call:([\w.\-]+)\s*\{')
BARE = re.compile(r'\{\s*"name"\s*:')
KEY = re.compile(r'([{,]\s*)([A-Za-z_][\w\-]*)\s*:')


def _gemma_calls(text: str) -> tuple[list[dict], bool]:
    """Gemma 4's shape: call:name{key: value, …}. Served without its <|"|> string tokens, values
    often come unquoted (`command:ls -la\n}`), so a value runs to the next `, key:` or the last `}`."""

    parts = text.split('<|"|>')
    text = ''.join(json.dumps(p) if i % 2 else p for i, p in enumerate(parts))
    calls, broken = [], False
    matches = list(GEMMA.finditer(text))
    for n, match in enumerate(matches):
        stop = matches[n + 1].start() if n + 1 < len(matches) else len(text)
        segment = text[match.end() - 1:stop]
        close = segment.rfind('}')
        body = segment[:close + 1] if close > 0 else segment + '}'
        arguments = _relaxed(body)
        if not isinstance(arguments, dict):
            arguments = _gemma_pairs(body[1:-1])
        if isinstance(arguments, dict):
            calls.append({'name': match.group(1), 'arguments': arguments})
        else:
            broken = True
    return calls, broken and not calls


PAIR = re.compile(r',\s*(?=[A-Za-z_][\w\-]*\s*[:=])')
HEAD = re.compile(r'\s*([A-Za-z_][\w\-]*)\s*[:=]\s*(.*)', re.S)


def _gemma_pairs(inner: str):
    if not inner.strip():
        return {}
    arguments = {}
    for part in PAIR.split(inner):
        head = HEAD.match(part)
        if not head:
            return None
        key, raw = head.group(1), head.group(2).strip()
        value = raw
        for candidate in (raw, raw.replace("\\'", "'")):
            try:
                value = json.loads(candidate, strict=False)
                break
            except json.JSONDecodeError:
                continue
        arguments[key] = value
    return arguments


def _plain(value):
    """A Python literal as JSON-able data (a set or tuple inside model text becomes a list)."""

    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v) for v in value]
    if value is Ellipsis:
        return '...'
    if value is not None and not isinstance(value, (str, int, float, bool)):
        return str(value)  # complex, bytes: not JSON, and an episode holding one could not be written
    return value


def _relaxed(body: str):
    candidates = [body, KEY.sub(r'\1"\2":', body)]
    for candidate in candidates:
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            pass
        try:
            fixed = re.sub(r'\btrue\b', 'True', re.sub(r'\bfalse\b', 'False', re.sub(r'\bnull\b', 'None', candidate)))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                value = _plain(ast.literal_eval(fixed))
            if isinstance(value, dict):
                return value
        except Exception:  # noqa: BLE001 - literal_eval raises many kinds on model text
            pass
    return None


def _native_arguments(body: str):
    parameters = PARAMETER.findall(body)
    if parameters:
        arguments = {}
        for key, value in parameters:
            try:
                arguments[key] = json.loads(value)
            except json.JSONDecodeError:
                arguments[key] = value
        return arguments
    if not body.strip():
        return {}
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return _loose(body) or 'unparsed'


LFM = re.compile(r'<\|tool_call_start\|>\s*(.*?)\s*(?:<\|tool_call_end\|>|$)', re.S)


def _pythonic_calls(text: str) -> tuple[list[dict], bool]:
    """LFM2.5's native shape: <|tool_call_start|>[name(key=value, …), …]<|tool_call_end|>."""

    calls, broken = [], False
    for blob in LFM.findall(text):
        try:
            tree = ast.parse(blob if blob.startswith('[') else f'[{blob}]', mode='eval').body
            items = tree.elts if isinstance(tree, ast.List) else [tree]
            for item in items:
                if not isinstance(item, ast.Call) or item.args:
                    raise ValueError
                calls.append({'name': ast.unparse(item.func),
                              'arguments': {k.arg: _plain(ast.literal_eval(k.value)) for k in item.keywords}})
        except (SyntaxError, ValueError, TypeError):
            broken = True
    return calls, broken and not calls


PARSER_VERSION = 5
NAME_KEYS = ('name', 'function', 'tool', 'tool_name')
ARGUMENT_KEYS = ('arguments', 'parameters', 'args', 'input')


def _decode(blob: str):
    """JSON, JSON with a missing or extra tail, or a Python literal (True/None)."""

    try:
        return json.loads(blob, strict=False)
    except json.JSONDecodeError:
        pass
    value = _loose(blob)
    if value is not None:
        return value
    start = blob.find('{')
    if start >= 0:
        piece = blob[start:blob.rfind('}') + 1]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                value = _plain(ast.literal_eval(re.sub(r'\btrue\b', 'True', re.sub(r'\bfalse\b', 'False', re.sub(r'\bnull\b', 'None', piece)))))
            if isinstance(value, (dict, list)):
                return value
        except Exception:  # noqa: BLE001 - literal_eval raises many kinds on model text
            pass
    return None


def _as_call(item, names: set[str] | None):
    """A call from one decoded object, in the shapes models use; None if it is not one."""

    if not isinstance(item, dict):
        return None
    for key in NAME_KEYS:
        value = item.get(key)
        if isinstance(value, dict):  # {"function": {"name": …, "arguments": …}}
            nested = _as_call(value, names)
            if nested:
                return nested
        if isinstance(value, str):
            arguments = next((item[k] for k in ARGUMENT_KEYS if k in item), {})
            if isinstance(arguments, str):
                decoded = _decode(arguments)
                if not isinstance(decoded, dict):
                    return {'name': value, 'arguments': {'_raw': arguments}} if names and value in names else None
                arguments = decoded
            return {'name': value, 'arguments': arguments if isinstance(arguments, dict) else {}}
    if names and len(item) == 1:  # {"tool_name": {…arguments}}
        (key, value), = item.items()
        if key in names and isinstance(value, dict):
            return {'name': key, 'arguments': value}
    return None


def _recover(blob: str, names: set[str]):
    """Malformed, but unambiguous: exactly one known tool is named; its arguments are the first object after it."""

    mentioned = {n for n in names if re.search(rf'(?<![\w.]){re.escape(n)}(?![\w.])', blob)}
    if len(mentioned) != 1:
        return None
    name = mentioned.pop()
    after = re.search(r'"?(?:arguments|parameters|args)"?\s*[:=]\s*', blob)
    arguments = {}
    if after:
        value = _decode(blob[after.end():])
        if isinstance(value, dict):
            arguments = value
    return {'name': name, 'arguments': arguments}


def parse_calls(text: str, names=None) -> tuple[list[dict], bool, bool]:
    """Tool calls in ``text``, whether it tried to make one and failed, and whether a malformed call
    was recovered (``names``: the tools of the task, needed to recover)."""

    names = set(names) if names else None
    if '<|tool_call_start|>' in text:
        calls, broken = _pythonic_calls(text)
        return calls, broken, False
    blobs = ([m.group(1) for m in CALL.finditer(text)] or [m.group(0) for m in FUNCTION.finditer(text)]
             or [m.group(1) for m in FENCE.finditer(text)])
    if not blobs and GEMMA.search(text):
        calls, broken = _gemma_calls(text)
        return calls, broken, False
    if not blobs:
        # served without its <tool_call> special token (Granite 4): bare top-level JSON objects
        decoder, end = json.JSONDecoder(strict=False), 0
        for start in (m.start() for m in re.finditer(r'\{\s*"', text)):
            if start < end:
                continue  # inside an object already taken
            try:
                value, end = decoder.raw_decode(text, start)
            except json.JSONDecodeError:
                value = _loose(text[start:]) if re.match(r'\{\s*"name"', text[start:]) else None
                end = start + 1
            call = _as_call(value, names)
            if call and (not names or call['name'] in names or any(k in value for k in ARGUMENT_KEYS)):
                blobs.append(json.dumps(value))
    calls, broken, recovered = [], False, False
    for blob in blobs:
        native = FUNCTION.search(blob)
        if native:
            # Qwen3.5 / Qwen3-Coder native shape: <function=name> then JSON or <parameter=k>v</parameter>
            value = {'name': native.group(1), 'arguments': _native_arguments(native.group(2))}
        else:
            value = _decode(blob)
        items = value if isinstance(value, list) else [value]
        found = [c for c in (_as_call(i, names) for i in items) if c]
        if not found and names:
            rescued = _recover(blob, names)
            if rescued:
                found, recovered = [rescued], True
        if found:
            calls += found
        else:
            broken = True
    if not calls and ('tool_call' in text or '"name"' in text or 'call:' in text):
        broken = True
    return calls, broken and not calls, recovered


def _loose(blob: str):
    """A JSON object with a trailing brace missing or extra text after it."""

    start = blob.find('{')
    if start < 0:
        return None
    ends = [i + 1 for i, c in enumerate(blob) if c == '}'][-40:][::-1] + [len(blob)]
    for end in ends:
        piece = blob[start:end]
        try:
            return json.loads(piece + '}' * max(0, piece.count('{') - piece.count('}')), strict=False)
        except json.JSONDecodeError:
            continue
    return None
