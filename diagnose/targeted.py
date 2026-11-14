"""This gives two copies of a forecast program equally sized feedback and checks their repairs."""
from dataclasses import replace
import json
import random

from .model import parse_reply
from .repairs import ArithmeticProgram
from .world import Car, initial

CATEGORIES = ('boundary_reset', 'fractional_motion', 'timer_left', 'timer_right')
RULES = '''Write forecast(x, timer, p, k), returning the integer head position after k ticks.
The road has columns 0..8. Each tick uses exactly one of these rules, in order:
If x < 0, set x=8 and leave timer unchanged. Else if x > 8, set x=0 and leave timer unchanged.
Else if abs(p)<1, add int(1/p) to x and leave timer unchanged.
Else subtract 1 from timer; only if timer == -1, add abs(p) to timer and move x by sign(p).
Do not wrap immediately after movement. k is an integer from 0 to 10.
Use one plain Python function forecast(x, timer, p, k). Allowed: numeric arithmetic,
assignments, if/else, for over bounded range, return, comparisons, abs/int/min/max/sign.
No imports, attributes, subscripts, helper functions, while loops, or comments.
Return a JSON object with code (string) and confidence (number from 0 to 1). No markdown.'''


def generation_prompt():
    return RULES


def extract_program(response):
    value, error = parse_reply(response, ('code',))
    if error:
        return None, None, error
    source = value['code']
    try:
        if not isinstance(source, str):
            raise ValueError('code must be a string')
        return ArithmeticProgram(source, extended=True, native_inputs=True), source, None
    except (ValueError, TypeError, SyntaxError) as exc:
        return None, source if isinstance(source, str) else None, str(exc)


def correct_forecast(x, timer, p, k):
    car = Car(x, 1, timer, p, 1)
    for _ in range(k):
        car = car.tick()
    return int(car.x)


def make_cases(seeds, split, horizons=range(1, 11), snapshot_ticks=(0, 3, 7)):
    seeds, horizons, snapshot_ticks = tuple(seeds), tuple(horizons), tuple(snapshot_ticks)
    if split not in ('train', 'validation', 'test'):
        raise ValueError('Unknown split')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Need distinct source episode seeds')
    if not horizons or len(set(horizons)) != len(horizons) or any(
            type(k) is not int or not 1 <= k <= 10 for k in horizons):
        raise ValueError('Need distinct horizons from 1 to 10')
    if not snapshot_ticks or len(set(snapshot_ticks)) != len(snapshot_ticks) or any(
            type(t) is not int or not 0 <= t < 100 for t in snapshot_ticks):
        raise ValueError('Need distinct nonterminal snapshot ticks')
    rows = []
    for seed in seeds:
        state = initial(seed)
        for tick in range(max(snapshot_ticks)+1):
            if tick in snapshot_ticks:
                for index, car in enumerate(state.cars):
                    for k in horizons:
                        current, boundary = car, False
                        for _ in range(k):
                            boundary |= current.x < 0 or current.x > 8
                            current = current.tick()
                        category = ('boundary_reset' if boundary else 'fractional_motion'
                                    if abs(car.speed) < 1 else 'timer_left'
                                    if car.speed < 0 else 'timer_right')
                        rows.append({'id': f'{seed}:{tick}:{index}:{k}', 'split': split,
                                     'source_seed': seed, 'state_id': state.key,
                                     'snapshot_tick': tick, 'car_index': index,
                                     'input': [int(car.x), float(car.timer), float(car.speed), k],
                                     'expected': int(current.x), 'category': category})
            state = replace(state, cars=tuple(c.tick() for c in state.cars), turn=tick+1)
    return rows


def check_splits(*collections):
    seen = {}
    ids = set()
    for rows in collections:
        for row in rows:
            seed, split = row['source_seed'], row['split']
            if seed in seen and seen[seed] != split:
                raise ValueError('Source episode crosses splits')
            if row['id'] in ids:
                raise ValueError('Duplicate case')
            seen[seed] = split
            ids.add(row['id'])
    return True


def evaluate(program, cases):
    rows = []
    for case in cases:
        predicted, error = None, None
        try:
            if program is None:
                raise ValueError('No executable program')
            predicted = program(*case['input'])
            if type(predicted) is not int:
                raise ValueError('Prediction must be an integer')
        except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
            error = str(exc)
        rows.append({**case, 'predicted': predicted, 'error': error,
                     'correct': error is None and predicted == case['expected']})
    return rows


def diagnose(rows):
    if any(r['category'] not in CATEGORIES for r in rows):
        raise ValueError('Unknown diagnostic category')
    result = {}
    for category in CATEGORIES:
        group = [r for r in rows if r['category'] == category]
        failed = sum(not r['correct'] for r in group)
        result[category] = {'total': len(group), 'failed': failed,
                            'error_rate': failed/len(group) if group else None}
    return result


def select_feedback(rows, arm, count, seed):
    if arm not in ('uniform', 'targeted'):
        raise ValueError('Unknown repair arm')
    if not rows or any(r['split'] != 'train' for r in rows):
        raise ValueError('Feedback must use training cases only')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate feedback candidates')
    if type(count) is not int or count < 1:
        raise ValueError('Feedback count must be positive')
    stats = diagnose(rows)
    rng = random.Random(seed)
    failures = [r for r in rows if not r['correct']]
    rng.shuffle(failures)
    category = None
    if arm == 'targeted' and failures:
        category = max((c for c in CATEGORIES if stats[c]['failed']),
                       key=lambda c: stats[c]['error_rate'])
        failures = ([r for r in failures if r['category'] == category]
                    + [r for r in failures if r['category'] != category])
    selected = failures[:count]
    return {'examples': selected, 'target_category': category, 'diagnostics': stats,
            'failure_examples': sum(not r['correct'] for r in selected)}


def repair_prompt(source, examples):
    if any(row['split'] != 'train' for row in examples):
        raise ValueError('Repair prompt requires training examples')
    feedback = [{'input': r['input'], 'expected': r['expected'],
                 'predicted': r['predicted'], 'error': r['error']} for r in examples]
    instruction = ('Repair this candidate. Return a complete replacement.' if examples else
                   'All checker cases pass; preserve the function.')
    return (RULES + '\n' + instruction + '\n'
            + json.dumps({'candidate': source, 'examples': feedback}, sort_keys=True))
