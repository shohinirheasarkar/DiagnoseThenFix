"""This plays with a saved forecast function and checks each recorded move and time charge."""
import ast
import math
import time

from .live import episode
from .repairs import ArithmeticProgram, bound_action
from .targeted import correct_forecast
from .world import ACTIONS, State, initial, step

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


def delayed_beliefs(state, program, delay):
    beliefs = {}
    for action in ACTIONS:
        lane, collision = state.lane, False
        for tick in range(1, 3 * (delay + 1) + 1):
            if tick % (delay + 1) == 0:
                lane = max(0, min(9, lane + {'U': -1, 'S': 0, 'D': 1}[action]))
            if lane == 0:
                break
            for car in state.cars:
                head = program(car.x, car.timer, car.speed, tick)
                if type(head) is not int:
                    raise ValueError('Prediction must be an integer')
                direction = -1 if car.speed > 0 else 1
                if car.lane == lane and any(head + offset * direction == 4
                                            for offset in range(car.length)):
                    collision = True
                    break
            if collision:
                break
        beliefs[action] = collision
    return beliefs


def decision(program, error, state, delay):
    action, beliefs = 'S', None
    if program is not None:
        try:
            beliefs = delayed_beliefs(state, program, delay)
            action = bound_action(beliefs) or 'S'
        except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
            program, error = None, str(exc)
    return program, error, action, beliefs


def evaluate_episode(source, env_seed, regime):
    if regime not in REGIMES:
        raise ValueError('Unknown delay regime')
    program, source, error, kind = prepare(source)
    planned = {'zero': 0, 'native': 1, 'two_ticks': 2}[regime]

    def policy(state, history):
        nonlocal program, error
        started = time.perf_counter()
        program, error, action, beliefs = decision(program, error, state, planned)
        cpu = time.perf_counter() - started
        seconds = cpu if regime == 'native' else 2. if regime == 'two_ticks' else 0.
        return {'action': action, 'collision': beliefs, 'error': error,
                'cpu_seconds': cpu, 'seconds': seconds, 'tokens': 0, 'calls': 0,
                'program_executable': program is not None,
                'planned_delay_ticks': planned, 'actual_delay_ticks': math.ceil(seconds)}

    result = episode(policy, env_seed, tick_seconds=1., max_decisions=100)
    for frame in result['trace']:
        frame.pop('stage_labels')
    return {'protocol': 'targeted-live-v3', 'source': source, 'program_kind': kind,
            'regime': regime, 'horizon': 3, 'preparation_cost_charged': False, **result}


def verify_episode(record):
    if (record['protocol'] != 'targeted-live-v3' or record['regime'] not in REGIMES
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
    planned = {'zero': 0, 'native': 1, 'two_ticks': 2}[record['regime']]
    collisions = ticks = stays = longest = 0
    if not record['trace'] or len(record['trace']) > 100:
        raise ValueError('Invalid trace length')
    for index, frame in enumerate(record['trace']):
        if state.turn >= 100 or state.lane == 0:
            raise ValueError('Trace continues after terminal state')
        if frame['decision'] != index or State.from_dict(frame['before']) != state:
            raise ValueError('Observation mismatch')
        program, error, action, beliefs = decision(program, error, state, planned)
        response = frame['response']
        cpu = response['cpu_seconds']
        if type(cpu) not in (int, float) or not math.isfinite(cpu) or cpu < 0:
            raise ValueError('Invalid CPU measurement')
        seconds = cpu if record['regime'] == 'native' else 2. if record['regime'] == 'two_ticks' else 0.
        expected = {'action': action, 'collision': beliefs, 'error': error,
                    'cpu_seconds': cpu, 'seconds': seconds, 'tokens': 0, 'calls': 0,
                    'program_executable': program is not None,
                    'planned_delay_ticks': planned, 'actual_delay_ticks': math.ceil(seconds)}
        if response != expected or frame['delay_ticks'] != math.ceil(seconds):
            raise ValueError('Policy or time charge mismatch')
        if 'stage_labels' in frame:
            raise ValueError('Legacy stage labels do not apply to delayed decisions')
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
