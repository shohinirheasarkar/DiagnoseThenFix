"""This plays with a saved forecast function and checks each recorded move and time charge."""
import ast
import math
import time

from .composed import forecast_with_code
from .live import episode
from .repairs import ArithmeticProgram, bound_action
from .targeted import correct_forecast
from .world import ACTIONS, State, initial, repeated_outcome, step

REGIMES = ('zero', 'native', 'two_ticks')


def prepare(source):
    if source is correct_forecast:
        return source, None, None, 'assisted_reference'
    if isinstance(source, ArithmeticProgram):
        source = ast.unparse(source.function)
    if source is None:
        return None, None, 'No executable program', 'candidate'
    if not isinstance(source, str):
        raise ValueError('Need source, ArithmeticProgram, None, or correct_forecast reference')
    try:
        return ArithmeticProgram(source, extended=True, native_inputs=True), source, None, 'candidate'
    except (SyntaxError, ValueError, TypeError) as exc:
        return None, source, str(exc), 'candidate'


def decision(program, error, state):
    action, beliefs = 'S', None
    if program is not None:
        try:
            beliefs = forecast_with_code(state, program, 3)
            action = bound_action(beliefs) or 'S'
        except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
            program, error = None, str(exc)
    return program, error, action, beliefs


def evaluate_episode(source, env_seed, regime):
    if regime not in REGIMES:
        raise ValueError('Unknown delay regime')
    program, source, error, kind = prepare(source)

    def policy(state, history):
        nonlocal program, error
        started = time.perf_counter()
        program, error, action, beliefs = decision(program, error, state)
        cpu = time.perf_counter() - started
        seconds = cpu if regime == 'native' else 2. if regime == 'two_ticks' else 0.
        return {'action': action, 'collision': beliefs, 'error': error,
                'cpu_seconds': cpu, 'seconds': seconds, 'tokens': 0, 'calls': 0,
                'program_executable': program is not None}

    result = episode(policy, env_seed, tick_seconds=1., max_decisions=100)
    return {'protocol': 'targeted-live-v1', 'source': source, 'program_kind': kind,
            'regime': regime, 'horizon': 3, 'preparation_cost_charged': False, **result}


def verify_episode(record):
    if (record['protocol'] != 'targeted-live-v1' or record['regime'] not in REGIMES
            or record['horizon'] != 3 or record['tick_seconds'] != 1.
            or record['preparation_cost_charged'] is not False):
        raise ValueError('Unknown live protocol')
    if record['program_kind'] not in ('candidate', 'assisted_reference'):
        raise ValueError('Unknown program kind')
    if record['program_kind'] == 'assisted_reference' and record['source'] is not None:
        raise ValueError('Reference must not contain candidate source')
    source = correct_forecast if record['program_kind'] == 'assisted_reference' else record['source']
    program, _, error, _ = prepare(source)
    state = initial(record['seed'])
    collisions = ticks = stays = longest = 0
    actions = []
    if not record['trace'] or len(record['trace']) > 100:
        raise ValueError('Invalid trace length')
    for index, frame in enumerate(record['trace']):
        if state.turn >= 100 or state.lane == 0:
            raise ValueError('Trace continues after terminal state')
        if frame['decision'] != index or State.from_dict(frame['before']) != state:
            raise ValueError('Observation mismatch')
        program, error, action, beliefs = decision(program, error, state)
        response = frame['response']
        cpu = response['cpu_seconds']
        if type(cpu) not in (int, float) or not math.isfinite(cpu) or cpu < 0:
            raise ValueError('Invalid CPU measurement')
        seconds = cpu if record['regime'] == 'native' else 2. if record['regime'] == 'two_ticks' else 0.
        expected = {'action': action, 'collision': beliefs, 'error': error,
                    'cpu_seconds': cpu, 'seconds': seconds, 'tokens': 0, 'calls': 0,
                    'program_executable': program is not None}
        if response != expected or frame['delay_ticks'] != math.ceil(seconds):
            raise ValueError('Policy or time charge mismatch')
        actual = {a: repeated_outcome(state, a, min(3, 100-state.turn)) == 'collision'
                  for a in ACTIONS}
        labels = {'forecast_correct': beliefs == actual if beliefs is not None else None,
                  'contradiction': beliefs[action] if beliefs is not None else None,
                  'history_mode_match': action == max(ACTIONS, key=actions.count)
                  if actions else None}
        if frame['stage_labels'] != labels:
            raise ValueError('Stage labels mismatch')
        actions.append(action)
        events = []
        for _ in range(math.ceil(seconds)):
            state, event = step(state, 'S')
            events.append(event)
            if state.lane == 0 or state.turn == 100:
                break
        applied = state.lane != 0 and state.turn < 100
        if applied:
            state, event = step(state, action)
            events.append(event)
        if (frame['action_applied'] != applied or frame['events'] != events
                or State.from_dict(frame['after']) != state):
            raise ValueError('Transition mismatch')
        ticks += len(events)
        collisions += events.count('collision')
        stays = stays + 1 if action == 'S' else 0
        longest = max(longest, stays)
    expected = {'success': state.lane == 0, 'collisions': collisions, 'tokens': 0,
                'calls': 0, 'ticks': ticks, 'longest_stay_loop': longest,
                'final_lane': state.lane, 'censored': state.lane != 0 and state.turn < 100}
    if any(record[key] != value for key, value in expected.items()):
        raise ValueError('Episode summary mismatch')
    if expected['censored']:
        raise ValueError('Incomplete episode')
    return True
