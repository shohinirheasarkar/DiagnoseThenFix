"""These checks keep feedback fair and keep test answers out of repairs."""
import json

import pytest

from diagnose.composed import forecast_with_code
from diagnose.targeted import (CATEGORIES, check_splits, correct_forecast, diagnose,
                              evaluate, extract_program, make_cases, repair_prompt,
                              select_feedback)
from diagnose.world import initial, repeated_outcome


def response(code):
    return {'error': None, 'raw': {'message': {'content': json.dumps({'code': code})}}}


def test_oracle_cases_and_categories():
    rows = make_cases(range(700000, 700040), 'train')
    assert all(r['correct'] for r in evaluate(correct_forecast, rows))
    assert set(r['category'] for r in rows) <= set(CATEGORIES)
    assert diagnose(evaluate(correct_forecast, rows))['timer_left']['error_rate'] is None
    assert len({r['id'] for r in rows}) == len(rows)
    assert rows == make_cases(range(700000, 700040), 'train')


@pytest.mark.parametrize('code', ['import os', 'def forecast(x,timer,p,k):\n return x.foo',
                                 'def forecast(x,timer,p,k):\n while True: pass', 42])
def test_invalid_programs(code):
    program, _, error = extract_program(response(code))
    assert program is None and error


def test_invalid_is_wrong_and_native_execution():
    rows = make_cases([700000], 'train', horizons=(1,), snapshot_ticks=(0,))
    assert all(not r['correct'] and r['error'] for r in evaluate(None, rows))
    program, _, error = extract_program(response('def forecast(x,timer,p,k):\n return x'))
    assert not error and program(3, 0., 2., 1) == 3


def test_split_leakage_and_duplicates():
    train = make_cases([700000], 'train')
    test = make_cases([700001], 'test')
    assert check_splits(train, test)
    with pytest.raises(ValueError, match='crosses'):
        check_splits(train, make_cases([700000], 'test'))
    with pytest.raises(ValueError, match='Duplicate'):
        check_splits(train, train)


def synthetic():
    rows = []
    for category, total, failures in [('boundary_reset', 20, 8), ('timer_left', 5, 4)]:
        for i in range(total):
            rows.append({'id': f'{category}:{i}', 'category': category, 'split': 'train',
                         'correct': i >= failures, 'input': [1, 0, -2, 1],
                         'expected': 0, 'predicted': 1, 'error': None})
    return rows


def test_target_uses_error_rate_and_equal_budget():
    rows = synthetic()
    assert diagnose(rows)['timer_left']['error_rate'] == .8
    target = select_feedback(rows, 'targeted', 6, 11)
    uniform = select_feedback(rows, 'uniform', 6, 11)
    assert target['target_category'] == 'timer_left'
    assert all(r['category'] == 'timer_left' for r in target['examples'][:4])
    assert target['failure_examples'] == uniform['failure_examples'] == 6
    assert len({r['id'] for r in target['examples']}) == 6
    assert target == select_feedback(rows, 'targeted', 6, 11)
    assert 'target_category' not in repair_prompt('code', target['examples'])


def test_sparse_failures_fill_identically_and_all_correct():
    rows = synthetic()
    for row in rows:
        row['correct'] = True
    for arm in ('uniform', 'targeted'):
        result = select_feedback(rows, arm, 6, 3)
        assert len(result['examples']) == 0
        assert result['failure_examples'] == 0
        assert result['target_category'] is None
    assert select_feedback(rows, 'uniform', 6, 3)['examples'] == select_feedback(
        rows, 'targeted', 6, 3)['examples']


def test_test_feedback_rejected():
    rows = synthetic()
    rows[0]['split'] = 'test'
    with pytest.raises(ValueError, match='training'):
        select_feedback(rows, 'targeted', 2, 1)
    with pytest.raises(ValueError, match='training'):
        repair_prompt('code', rows)


@pytest.mark.parametrize('horizon', [1, 3, 10])
def test_correct_forecast_controller_matches_physics(horizon):
    for seed in range(700000, 700010):
        state = initial(seed)
        beliefs = forecast_with_code(state, correct_forecast, horizon)
        for action in beliefs:
            assert beliefs[action] == (repeated_outcome(state, action, horizon) == 'collision')


@pytest.mark.parametrize('kwargs', [{'horizons': (0,)}, {'snapshot_ticks': (100,)},
                                   {'horizons': (1, 1)}, {'snapshot_ticks': ()}])
def test_bad_case_configuration(kwargs):
    with pytest.raises(ValueError):
        make_cases([1], 'train', **kwargs)


def test_one_failure_is_not_padded_and_empty_prompt_preserves():
    rows = synthetic()
    for row in rows[1:]:
        row['correct'] = True
    for arm in ('uniform', 'targeted'):
        chosen = select_feedback(rows, arm, 6, 0)
        assert chosen['examples'] == rows[:1]
    assert 'All checker cases pass; preserve the function.' in repair_prompt('code', [])


def test_boundary_reset_is_a_separate_tick():
    assert correct_forecast(8, 0, 1, 1) == 9
    assert correct_forecast(8, 0, 1, 2) == 0
    assert correct_forecast(-1, 0, -.5, 1) == 8
    assert correct_forecast(-1, 0, -.5, 2) == 6


def test_runtime_error_is_saved_without_dropping_case():
    program, _, _ = extract_program(response('def forecast(x,timer,p,k):\n return 1/0'))
    rows = make_cases([700000], 'test', horizons=(1,), snapshot_ticks=(0,))
    scored = evaluate(program, rows)
    assert len(scored) == len(rows)
    assert all(r['error'] and not r['correct'] for r in scored)
