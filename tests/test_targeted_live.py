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
