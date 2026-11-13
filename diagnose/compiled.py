"""This reuses saved model-written code to choose moves, so it need not ask the model every turn."""
import argparse
from datetime import datetime
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

from .composed import forecast_with_code, program_error
from .live import episode
from .model import append_jsonl, canonical, parse_reply, read_jsonl
from .probes import make_task
from .repairs import ArithmeticProgram, bound_action
from .runner import source_fingerprint
from .world import initial, step

PROTOCOL = 'compiled-controller-v2'
LEGACY_PROTOCOL = 'compiled-controller-v1'


def load_programs(path):
    programs = {}
    expected_prompt = make_task(initial(1000), 'B', repair='code_v2')['prompt']
    if Path(path).suffix == '.gz':
        with gzip.open(path, 'rt') as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    else:
        records = read_jsonl(path)
    for record in records:
        task = record['task']
        if (task['stage'] != 'B' or task['repair'] != 'code_v2'
                or task['kind'] != 'position' or task['paraphrase'] != 0):
            continue
        seed, response = task['seed'], record['response']
        if seed not in range(5):
            continue
        request = response['identity']['request']
        if (request['messages'] != [{'role': 'user', 'content': expected_prompt}]
                or request['options']['seed'] != seed or request['model'] != record['model']
                or response['key'] != hashlib.sha256(canonical(response['identity']).encode()).hexdigest()):
            raise ValueError('Expected an isolated saved generation per seed')
        if seed in programs and programs[seed]['response']['key'] != response['key']:
            raise ValueError('Several programs for one seed; do not select by test performance')
        if seed in programs and programs[seed]['response']['raw'] != response['raw']:
            raise ValueError('Different replies for one generation key')
        if response.get('auxiliary'):
            raise ValueError('Expected one isolated generation call')
        programs[seed] = {'model': record['model'], 'seed': seed, 'response': response}
    if set(programs) != set(range(5)):
        raise ValueError('All five generation seeds are required, including failed programs')
    if len({p['model'] for p in programs.values()}) != 1:
        raise ValueError('One source file must contain one model')
    prompts = {p['response']['identity']['request']['messages'][0]['content'] for p in programs.values()}
    if len(prompts) != 1:
        raise ValueError('Programs must share the same state-independent generation prompt')
    return [programs[seed] for seed in range(5)]


def prepare(response, validation='checked', native_inputs=True):
    if validation not in ('checked', 'executable'):
        raise ValueError('Unknown program gate')
    training_error = None
    value, error = parse_reply(response, ('code', 'confidence'))
    program = None
    if not error:
        try:
            confidence = value['confidence']
            if type(confidence) not in (int, float) or not 0 <= confidence <= 1:
                raise ValueError('Invalid generation confidence')
            program = ArithmeticProgram(value['code'], extended=True, native_inputs=native_inputs)
            failure = program_error(program)
            if failure is not None:
                training_error = 'Training-case failure: ' + canonical(failure)
                if validation == 'checked':
                    raise ValueError(training_error)
        except (ValueError, KeyError, TypeError, SyntaxError, ArithmeticError) as exc:
            program, error = None, str(exc)
    return program, error, training_error


class CompiledPolicy:
    def __init__(self, response, condition='amortized', horizon=3, validation='checked'):
        if condition not in ('amortized', 'per_turn') or not 1 <= horizon <= 10:
            raise ValueError('Invalid compiled-controller condition or horizon')
        self.response, self.condition, self.horizon = response, condition, horizon
        started = time.monotonic()
        self.program, self.error, self.training_error = prepare(response, validation)
        self.setup_seconds = time.monotonic()-started
        self.first = True

    def __call__(self, state, history):
        started = time.monotonic()
        beliefs, action = None, 'S'
        if self.program is not None:
            try:
                beliefs = forecast_with_code(state, self.program, self.horizon)
                action = bound_action(beliefs) or 'S'
            except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
                self.program, self.error = None, str(exc)
        cpu = time.monotonic()-started + (self.setup_seconds if self.first else 0.)
        responses = [self.response] if self.first or self.condition == 'per_turn' else []
        self.first = False
        return {'action': action, 'collision': beliefs, 'error': self.error,
                'seconds': cpu + sum(r['wall_seconds'] for r in responses), 'cpu_seconds': cpu,
                'responses': responses, 'calls': len(responses), 'fresh_requests': 0,
                'tokens': sum(r['raw'].get('eval_count', 0) for r in responses),
                'program_executable': self.program is not None, 'training_check_error': self.training_error,
                'generation_reused': True,
                'policy_revision': PROTOCOL, 'repairs': ['code', 'bound']}


@lru_cache(maxsize=64)
def prepared(serialized, validation, native_inputs):
    return prepare(json.loads(serialized), validation, native_inputs=native_inputs)


def verify(record):
    if record['protocol'] not in (PROTOCOL, LEGACY_PROTOCOL) or record['condition'] not in ('amortized', 'per_turn'):
        raise ValueError('Unknown compiled-controller protocol')
    generation = record['generation_response']
    request = generation['identity']['request']
    if (record['program_seed'] not in range(5) or request['options']['seed'] != record['program_seed']
            or request['model'] != record['model'] or not 1 <= record['horizon'] <= 10):
        raise ValueError('Compiled source identity mismatch')
    program, error, training_error = prepared(canonical(generation), record['validation'],
                                            record['protocol'] == PROTOCOL)
    current = initial(record['seed'])
    for index, frame in enumerate(record['trace']):
        response = frame['response']
        beliefs, action = None, 'S'
        if program is not None:
            try:
                beliefs = forecast_with_code(current, program, record['horizon'])
                action = bound_action(beliefs) or 'S'
            except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
                program, error = None, str(exc)
        expected = [generation] if index == 0 or record['condition'] == 'per_turn' else []
        cpu = response['cpu_seconds']
        if not math.isfinite(cpu) or cpu < 0:
            raise ValueError('Invalid compiled CPU cost')
        if (response['responses'] != expected or response['fresh_requests'] != 0
                or response['seconds'] != cpu+sum(r['wall_seconds'] for r in expected)
                or response['policy_revision'] != record['protocol'] or not response['generation_reused']):
            raise ValueError('Compiled generation-charge mismatch')
        if (response['action'] != action or response['collision'] != beliefs
                or response['error'] != error or response['program_executable'] != (program is not None)
                or response['training_check_error'] != training_error):
            raise ValueError('Compiled policy does not follow its saved program')
        for _ in range(frame['delay_ticks']):
            current, _ = step(current, 'S')
            if current.lane == 0 or current.turn == 100:
                break
        if frame['action_applied']:
            current, _ = step(current, response['action'])
    return True


def run(source, output, stop_at, seeds=range(60000, 60050)):
    source, output = Path(source), Path(output)
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Need distinct episode seeds')
    samples = load_programs(source)
    revision_file = Path(__file__).resolve().parents[1]/'SOURCE_REVISION'
    revision = (revision_file.read_text().strip() if revision_file.exists() else
                subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    design = {'protocol': PROTOCOL, 'source': source_fingerprint(), 'source_commit': revision,
              'generation_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'model': samples[0]['model'], 'program_seeds': list(range(5)), 'seeds': list(seeds),
              'conditions': ['amortized', 'per_turn'], 'validations': ['executable', 'checked'],
              'native_inputs': True,
              'horizon': 3, 'tick_seconds': 1.,
              'fresh_requests': 0, 'generation_charge': 'saved measured duration, once or each turn',
              'training_check_seeds': list(range(1000, 1006))}
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['design'] != design:
        raise ValueError('Compiled design changed; choose a new output folder')
    manifest.write_text(json.dumps({'design': design, 'status': 'running'}, indent=2))
    path = output/'episodes.jsonl'
    done = {(r['program_seed'], r['seed'], r['condition'], r['validation']) for r in read_jsonl(path)}
    for sample in samples:
        for seed in seeds:
            for validation in design['validations']:
                for condition in design['conditions']:
                    if (sample['seed'], seed, condition, validation) in done:
                        continue
                    if time.time() >= stop_at:
                        manifest.write_text(json.dumps({'design': design, 'status': 'time_limit'}, indent=2))
                        return read_jsonl(path)
                    policy = CompiledPolicy(sample['response'], condition, validation=validation)
                    result = episode(policy, seed, tick_seconds=1., stop_at=stop_at)
                    append_jsonl(path, {'protocol': PROTOCOL, 'model': sample['model'],
                         'program_seed': sample['seed'], 'condition': condition, 'validation': validation,
                         'horizon': 3, 'generation_response': sample['response'], **result})
                    print(f'COMPILED {sample["model"]} p={sample["seed"]} env={seed} '
                          f'{condition}/{validation} goal={result["success"]}', flush=True)
    manifest.write_text(json.dumps({'design': design, 'status': 'complete'}, indent=2))
    return read_jsonl(path)


def main():
    parser = argparse.ArgumentParser(description='Replay saved code with no model-server access')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline', required=True)
    args = parser.parse_args()
    stop = datetime.fromisoformat(args.deadline.replace('Z', '+00:00')).timestamp()
    run(args.source, args.output, stop)


if __name__ == '__main__':
    main()
