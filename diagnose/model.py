"""This calls one local model at a time and saves the exact request and reply so runs can resume."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def read_jsonl(path):
    if not Path(path).exists():
        return []
    rows = []
    with Path(path).open() as handle:
        for index, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f'Broken JSONL at {path}:{index}; repair before resuming') from error
    return rows


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        handle.write(canonical(row) + '\n')
        handle.flush()


class LocalModel:
    def __init__(self, name, cache='.cache/model', endpoint='http://127.0.0.1:11434',
                 timeout=180, transport=None):
        url = urlparse(endpoint)
        if url.hostname not in ('127.0.0.1', 'localhost', '::1') or url.scheme != 'http':
            raise ValueError('Only a local Ollama endpoint is enabled')
        self.name, self.endpoint, self.timeout = name, endpoint.rstrip('/'), timeout
        self.transport = transport or self._http
        tags = self.transport('/api/tags', None)
        matches = [m for m in tags['models'] if m['name'] == name]
        if len(matches) != 1:
            raise ValueError(f'Model {name} is not installed; no automatic downloads')
        self.digest = matches[0]['digest']
        self.info = self.transport('/api/show', {'model': name})
        self.version = self.transport('/api/version', None)
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)

    def _http(self, route, payload):
        body = None if payload is None else canonical(payload).encode()
        request = Request(self.endpoint + route, data=body,
                          headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def ask(self, prompt, seed=0, thinking=False, max_tokens=512, schema=None,
            temperature=.7, history=(), use_cache=True, logprobs=False):
        supported = self.info.get('thinking', {}).get('values')
        if supported is not None and thinking not in supported:
            raise ValueError(f'Unsupported thinking value {thinking}')
        request = {'model': self.name, 'stream': False, 'think': thinking,
                   'messages': [*history, {'role': 'user', 'content': prompt}],
                   'options': {'seed': seed, 'temperature': temperature,
                               'num_predict': max_tokens, 'num_ctx': 8192},
                   'keep_alive': '5m'}
        if logprobs:
            request.update(logprobs=True, top_logprobs=20)
        if schema is not None:
            request['format'] = schema
        identity = {'digest': self.digest, 'version': self.version, 'request': request}
        key = hashlib.sha256(canonical(identity).encode()).hexdigest()
        path = self.cache / f'{key}.json'
        if use_cache and path.exists():
            cached = json.loads(path.read_text())
            if cached['identity'] != identity:
                raise ValueError('Cache identity mismatch')
            return {**cached, 'cache_hit': True}
        started = time.monotonic()
        try:
            raw = self.transport('/api/chat', request)
            error = None
        except (TimeoutError, HTTPError, URLError, OSError, json.JSONDecodeError) as exc:
            raw, error = {}, f'{type(exc).__name__}: {exc}'
        result = {'key': key, 'identity': identity, 'raw': raw, 'error': error,
                  'wall_seconds': time.monotonic() - started, 'cache_hit': False,
                  'created_at': datetime.now(timezone.utc).isoformat()}
        if error is None:
            temporary = path.with_suffix('.tmp')
            temporary.write_text(canonical(result))
            temporary.replace(path)
        return result

    def action_scores(self, prompt, seed=0):
        result = self.ask(prompt + '\nReply with exactly one token: A for U, B for S, C for D.',
                          seed=seed, thinking=False, max_tokens=1, temperature=1., logprobs=True)
        try:
            first = result['raw']['logprobs'][0]
            candidates = list(first['top_logprobs']) + [first]
            unique = {r['token']: float(r['logprob']) for r in candidates}
            scores = {}
            for action, token in (('U', 'A'), ('S', 'B'), ('D', 'C')):
                values = [p for t, p in unique.items() if t.strip() == token]
                if not values:
                    raise ValueError(f'{token} missing from top log probabilities')
                peak = max(values)
                scores[action] = peak + math.log(sum(math.exp(v-peak) for v in values))
            return {'scores': scores, 'response': result, 'error': None}
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            return {'scores': None, 'response': result, 'error': f'logprobs unavailable: {exc}'}


def parse_reply(result, required):
    if result['error']:
        return None, result['error']
    raw = result['raw']
    if raw.get('done_reason') == 'length':
        return None, 'truncated'
    try:
        content = raw['message']['content'].strip()
        value = json.loads(content)
        if not isinstance(value, dict) or not set(required) <= value.keys():
            raise ValueError('missing fields')
        return value, None
    except (ValueError, KeyError, TypeError) as error:
        return None, f'malformed: {error}'
