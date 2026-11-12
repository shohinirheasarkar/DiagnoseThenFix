"""These tests check that repairs use beliefs and that delayed actions really see a moving world."""
from dataclasses import replace

import pytest

from diagnose.live import baseline, critical_step, episode
from diagnose.repairs import (ArithmeticProgram, ablations, bound_action, calibrate,
                              choose_repairs, monitor, retry_once)
from diagnose.world import Car, State


@pytest.mark.parametrize('beliefs,expected', [({'U': False, 'S': False, 'D': False}, 'U'),
                                           ({'U': True, 'S': False, 'D': False}, 'S'),
                                           ({'U': True, 'S': True, 'D': False}, 'D'),
                                           ({'U': True, 'S': True, 'D': True}, None)])
def test_binding_does_not_peek_at_simulator(beliefs, expected):
    assert bound_action(beliefs) == expected


def test_retry_uses_one_retry_and_falls_back_to_beliefs():
    calls = []
    beliefs = {'U': True, 'S': False, 'D': False}
    def retry():
        calls.append(1)
        return 'U'
    assert retry_once('U', beliefs, retry) == ('S', True)
    assert len(calls) == 1
    assert retry_once('S', beliefs, retry) == ('S', False)
    assert len(calls) == 1


def test_calibration_uses_complete_log_scores():
    scores = {'U': -2., 'S': -1., 'D': -4.}
    prior = {'U': -3., 'S': -.2, 'D': -4.}
    assert calibrate(scores, prior)[0] == 'U'
    with pytest.raises(ValueError):
        calibrate(scores, {'U': -1})


def test_profile_does_not_fit_test_set():
    with pytest.raises(ValueError):
        choose_repairs({'split': 'test', 'A': .1})
    selected = choose_repairs({'split': 'validation', 'B': .1, 'C': 1., 'D': .2})
    assert selected == {'code', 'no_history'}
    comparisons = ablations(selected)
    assert len(comparisons['random']) == len(selected)
    assert comparisons['without_code'] == {'no_history'}


def test_monitor_has_no_oracle_inputs():
    assert monitor(3, {'U': False, 'S': False, 'D': True}, 'D', 3, False) == {
        'structured', 'code', 'bound', 'no_history', 'calibrate'}
    assert monitor() == set()


CODE = '''def forecast(x, timer, p, k):
    for i in range(k):
        if x < 0:
            x = 8
        elif x > 8:
            x = 0
        elif abs(p) >= 1:
            timer -= 1
            if timer == -1:
                timer += abs(p)
                if p > 0:
                    x += 1
                else:
                    x -= 1
        else:
            x += int(1 / p)
    return x
'''


@pytest.mark.parametrize('p', [1, -1, 3, -3, .25, -.25])
@pytest.mark.parametrize('k', range(1, 11))
def test_interpreted_forecaster_handles_actual_motion(p, k):
    c = Car(8, 1, 0, p, 1)
    expected = c
    for _ in range(k):
        expected = expected.tick()
    assert ArithmeticProgram(CODE)(c.x, c.timer, c.speed, k) == expected.x


@pytest.mark.parametrize('code', [
    "import os\ndef forecast(x,timer,p,k):\n return 1",
    "def forecast(x,timer,p,k):\n return x.__class__",
    "def forecast(x,timer,p,k):\n return __import__('os').system('whoami')",
    "def forecast(x,timer,p,k):\n while True: x=1\n return x",
    "def forecast(x,timer,p,k):\n for i in range(1000000): x=1\n return x",
])
def test_generated_code_cannot_access_python_or_run_unbounded(code):
    with pytest.raises((ValueError, KeyError)):
        ArithmeticProgram(code)(1, 0, 1, 2)


def test_world_moves_during_thinking_and_can_finish_before_action():
    state = State((Car(3, 8, 0, 1, 1),), lane=8, source_seed=1000)
    def policy(s, h):
        return {'action': 'U', 'seconds': 1., 'calls': 1, 'tokens': 3}
    result = episode(policy, start=state, max_decisions=1)
    assert result['collisions'] == 1
    assert result['trace'][0]['delay_ticks'] == 1
    near_timeout = replace(state, turn=99)
    result = episode(policy, start=near_timeout)
    assert not result['trace'][0]['action_applied']
    assert result['ticks'] == 1 and result['calls'] == 1


def test_goal_and_stay_loop_metrics():
    state = State((), lane=3)
    result = episode(baseline('advance'), start=state)
    assert result['success'] and result['ticks'] == 3
    result = episode(baseline('stay'), start=state, max_decisions=7)
    assert not result['success'] and result['longest_stay_loop'] == 7


def test_critical_step_reports_earliest_successful_intervention():
    failed = {'success': False, 'trace': [{}, {}, {}]}
    calls = []
    def replay(index, stage, row):
        calls.append((index, stage))
        return {'success': index == 1 and stage == 'C'}
    found = critical_step(failed, replay)
    assert found['decision'] == 1 and found['stage'] == 'C'
    assert calls[-1] == (1, 'C')
