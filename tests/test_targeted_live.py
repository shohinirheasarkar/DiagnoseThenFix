"""These tests replay saved games and catch changed actions, timing, and summaries."""
from copy import deepcopy

import pytest

from diagnose.targeted import correct_forecast
from diagnose.targeted_live import evaluate_episode, verify_episode


@pytest.mark.parametrize('regime', ['zero', 'native', 'two_ticks'])
@pytest.mark.parametrize('source', [None, 'bad code', 'def forecast(x,timer,p,k):\n return 1/0',
                                   'def forecast(x,timer,p,k):\n return x', correct_forecast])
def test_roundtrip(source, regime):
    record = evaluate_episode(source, 900010, regime)
    assert verify_episode(record)
    assert record['calls'] == record['tokens'] == 0
    assert len(record['trace']) <= 100
    assert not record['censored']
    delay = record['trace'][0]['delay_ticks']
    assert delay == (0 if regime == 'zero' else 2 if regime == 'two_ticks' else 1)


@pytest.mark.parametrize('change', ['action', 'seconds', 'after', 'events', 'success', 'cpu'])
def test_tampering(change):
    record = deepcopy(evaluate_episode(correct_forecast, 900010, 'zero'))
    frame = record['trace'][0]
    if change == 'action':
        frame['response']['action'] = 'X'
    elif change == 'seconds':
        frame['response']['seconds'] = 2
    elif change == 'after':
        frame['after']['lane'] = 0
    elif change == 'events':
        frame['events'] = []
    elif change == 'success':
        record['success'] = not record['success']
    else:
        frame['response']['cpu_seconds'] = float('nan')
    with pytest.raises(ValueError):
        verify_episode(record)


def test_invalid_program_stays_and_error_persists():
    record = evaluate_episode(None, 900010, 'zero')
    assert all(f['response']['action'] == 'S' for f in record['trace'])
    assert all(f['response']['error'] == 'No executable program' for f in record['trace'])


def test_unknown_regime_rejected():
    with pytest.raises(ValueError):
        evaluate_episode(None, 1, 'mystery')


@pytest.mark.parametrize('delay', [0, 1, 2])
def test_delay_beliefs_match_wait_then_action_physics(delay):
    from dataclasses import replace
    from diagnose.targeted_live import delayed_beliefs
    from diagnose.world import ACTIONS, initial, step
    for seed in range(900100, 900110):
        for lane in (1, 4, 8, 9):
            state = replace(initial(seed), lane=lane)
            beliefs = delayed_beliefs(state, correct_forecast, delay)
            for action in ACTIONS:
                current, collision = state, False
                for move in (['S'] * delay + [action]) * 3:
                    current, event = step(current, move)
                    if event in ('collision', 'goal'):
                        collision = event == 'collision'
                        break
                assert beliefs[action] == collision


def test_live_rejects_non_integer_forecast():
    from diagnose.targeted_live import delayed_beliefs
    from diagnose.world import initial
    with pytest.raises(ValueError, match='integer'):
        delayed_beliefs(initial(900100), lambda *args: 4., 1)


def test_planned_and_actual_delays_recorded():
    for regime, delay in [('zero', 0), ('native', 1), ('two_ticks', 2)]:
        record = evaluate_episode(correct_forecast, 900100, regime)
        assert all(f['response']['planned_delay_ticks'] == delay for f in record['trace'])
        assert all(f['response']['actual_delay_ticks'] == f['delay_ticks'] for f in record['trace'])
