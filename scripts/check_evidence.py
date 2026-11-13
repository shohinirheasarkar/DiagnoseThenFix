"""This checks saved forecast scores and replays live episodes, without calling a model."""
import argparse
from dataclasses import asdict, replace
import gzip
import hashlib
import json
import math
from pathlib import Path

from diagnose.model import canonical, parse_reply
from diagnose.probes import score
from diagnose.repairs import ArithmeticProgram
from diagnose.world import Car, State, futures, initial, repeated_outcome, step
from diagnose.commitment import score as commitment_score, truth as commitment_truth, validate_beliefs
from diagnose.repairs import bound_action


def read(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                yield number, json.loads(line)


def verify_live(record, legacy_terminal_reset=False):
    from realtimegym.environments.freeway import FreewayEnv
    reference = FreewayEnv()
    reference.set_seed(record['seed'])
    reference.reset()
    current = initial(record['seed'])
    collisions = ticks = calls = tokens = 0

    def advance(state, action):
        goal = reference.pos == 1 and action == 'U'
        _, terminal, _, collision = reference.step(action)
        expected_event = 'goal' if goal else 'collision' if collision else 'timeout' if terminal else 'neutral'
        following, event = step(state, action)
        reference_cars = tuple(Car(*car) for car in reference.cars if car[3] is not None)
        if (event != expected_event or following.turn != reference.game_turn or
                following.lane != (0 if goal else reference.pos) or following.cars != reference_cars):
            raise ValueError('Upstream trajectory mismatch')
        if legacy_terminal_reset and following.turn == 100 and event == 'collision':
            following = replace(initial(state.source_seed), turn=100)
        return following, event

    for frame in record['trace']:
        if canonical(asdict(current)) != canonical(frame['before']):
            raise ValueError('Live before-state mismatch')
        response = frame['response']
        expected_delay = math.ceil(max(0., response.get('seconds', 0.))/record['tick_seconds'])
        if frame['delay_ticks'] != expected_delay:
            raise ValueError('Live delay mismatch')
        raw_calls = response.get('responses', [response['response']] if 'response' in response else [])
        generated = sum(r['raw'].get('eval_count', 0) for r in raw_calls)
        if response.get('calls', 0) != len(raw_calls) or response.get('tokens', 0) != generated:
            raise ValueError('Live request-cost mismatch')
        calls += len(raw_calls)
        tokens += generated
        events = []
        if frame['delay_ticks'] < 0:
            raise ValueError('Negative delay')
        for _ in range(frame['delay_ticks']):
            current, event = advance(current, 'S')
            events.append(event)
            if current.lane == 0 or current.turn == 100:
                break
        applied = current.lane != 0 and current.turn < 100
        if applied:
            current, event = advance(current, frame['response']['action'])
            events.append(event)
        if applied != frame['action_applied'] or events != frame['events']:
            raise ValueError('Live action or event mismatch')
        if canonical(asdict(current)) != canonical(frame['after']):
            raise ValueError('Live after-state mismatch')
        collisions += events.count('collision')
        ticks += len(events)
    if record['success'] != (current.lane == 0):
        raise ValueError('Live success label mismatch')
    if record['censored'] != (current.lane != 0 and current.turn < 100):
        raise ValueError('Live censoring label mismatch')
    if record['final_lane'] != current.lane:
        raise ValueError('Live final-lane mismatch')
    if record['calls'] != calls or record['tokens'] != tokens:
        raise ValueError('Live cost totals mismatch')
    if record['collisions'] != collisions or record['ticks'] != ticks:
        raise ValueError('Live episode totals mismatch')


def verify_forecast(record, situation):
    task = record['task']
    state = State.from_dict(situation['state'])
    expected = (futures(state, task['k'])[-1][0] if task['kind'] == 'position'
                else repeated_outcome(state, 'U', task['k']) == 'collision')
    if task['truth'] != expected:
        raise ValueError('Forecast target mismatch')
    code = task['repair'] in ('code', 'code_v2')
    required = ('code', 'confidence') if code else ('answer', 'confidence')
    value, error = parse_reply(record['response'], required)
    if not error and code:
        try:
            c = state.cars[0]
            answer = ArithmeticProgram(value['code'], extended=task['repair'] == 'code_v2')(
                c.x, c.timer, c.speed, task['k'])
            value = {**value, 'answer': answer}
        except (ValueError, KeyError, TypeError, SyntaxError, ArithmeticError) as exc:
            error = str(exc)
    computed = score(task, value, error, state, situation['oracle'])
    if any(computed[key] != record['score'][key] for key in ('valid', 'correct')):
        raise ValueError('Forecast score mismatch')


def verify_commitment(record, situation):
    state = State.from_dict(situation['state'])
    target = commitment_truth(state, record['horizon'])
    if target != record['target'] or state.source_seed != record['source_seed']:
        raise ValueError('Commitment target mismatch')
    responses = record['responses']
    forecast, forecast_error = parse_reply(responses[0], ('unsafe', 'confidence'))
    beliefs = None
    if not forecast_error:
        try:
            beliefs = validate_beliefs(forecast['unsafe'])
            confidence = forecast['confidence']
            if type(confidence) not in (int, float) or not 0 <= confidence <= 1:
                raise ValueError('Invalid forecast confidence')
        except (ValueError, KeyError, TypeError) as exc:
            forecast_error = str(exc)
    condition = record['condition']
    own = error = value = None
    if condition in ('bound', 'shield'):
        if len(responses) != 1:
            raise ValueError('Unexpected commitment request count')
        if condition == 'bound' and forecast_error:
            error = forecast_error
        else:
            own = beliefs if condition == 'bound' else target['unsafe']
            value = {'action': bound_action(own), 'confidence': 1.}
    elif condition == 'own' and forecast_error:
        error = forecast_error
        if len(responses) != 1:
            raise ValueError('Unexpected commitment request count')
    else:
        combined = condition in ('combined', 'generic', 'retry')
        required = ('action', 'confidence', 'unsafe') if combined else ('action', 'confidence')
        value, error = parse_reply(responses[1], required)
        own = (value.get('unsafe') if combined and not error
               else target['unsafe'] if condition == 'oracle' else beliefs)
        retry = False
        if condition == 'retry' and not error:
            proposed = commitment_score(value, None, target, own)
            retry = bool(proposed.get('contradiction') or proposed.get('failed_abstention'))
        if len(responses) != 2+int(retry):
            raise ValueError('Unexpected commitment request count')
        if retry:
            value, error = parse_reply(responses[-1], ('action', 'confidence'))
    if value != record['value']:
        raise ValueError('Commitment parsed-value mismatch')
    computed = commitment_score(value, error, target, own)
    if {k: v for k, v in computed.items() if k != 'error'} != {k: v for k, v in record['score'].items() if k != 'error'}:
        raise ValueError('Commitment score mismatch')


def check(root, bank_path=Path('data/freeway.json')):
    root = Path(root)
    bank = {r['id']: r for r in json.loads(Path(bank_path).read_text())['situations']}
    counts = {'forecast_trials': 0, 'commitment_trials': 0, 'live_episodes': 0, 'files': 0}
    archive = root / 'archive.json'
    cycle_path = root / 'cycle.json'
    legacy_terminal_reset = (cycle_path.exists() and
                            json.loads(cycle_path.read_text()).get('source_commit') ==
                            'f71b9259abd055291c7c865659363aa1b3d9ec0f')
    if archive.exists():
        for record in json.loads(archive.read_text())['files']:
            path = (root / record['artifact']).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError('Archive path escapes its root')
            if hashlib.sha256(path.read_bytes()).hexdigest() != record['artifact_sha256']:
                raise ValueError(f'Archive hash mismatch: {path}')
            opener = gzip.open if path.suffix == '.gz' else open
            digest, size = hashlib.sha256(), 0
            with opener(path, 'rb') as handle:
                while chunk := handle.read(1024*1024):
                    digest.update(chunk)
                    size += len(chunk)
            if digest.hexdigest() != record['source_sha256'] or size != record['source_bytes']:
                raise ValueError(f'Archive round-trip mismatch: {path}')
    for path in sorted(root.rglob('*')):
        if not path.name.endswith(('.jsonl', '.jsonl.gz')):
            continue
        counts['files'] += 1
        identifiers = set()
        for number, record in read(path):
            try:
                if 'trial_id' in record:
                    if record['trial_id'] in identifiers:
                        raise ValueError('Duplicate trial ID')
                    identifiers.add(record['trial_id'])
                if 'task' in record and record['task']['stage'] == 'B':
                    verify_forecast(record, bank[record['state_id']])
                    counts['forecast_trials'] += 1
                elif record.get('protocol') == 'repeated-action-commitment-v1':
                    verify_commitment(record, bank[record['state_id']])
                    counts['commitment_trials'] += 1
                elif 'trace' in record:
                    verify_live(record, legacy_terminal_reset)
                    counts['live_episodes'] += 1
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f'{path}:{number}: {error}') from error
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--bank', type=Path, default=Path('data/freeway.json'))
    args = parser.parse_args()
    print(json.dumps(check(args.results, args.bank), indent=2))


if __name__ == '__main__':
    main()
