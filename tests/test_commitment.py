"""These tests distinguish real commitment inconsistency from risk in an uncommitted future."""
import json

import pytest

from diagnose.commitment import CONDITIONS, prompt, run, score, truth
from diagnose.model import LocalModel
from diagnose.world import Car, State, initial, label
from test_study import FakeOllama


def test_commitment_and_forecast_use_identical_time_horizon():
    s = State((Car(2, 7, 0, 1, 1),), lane=9, source_seed=1000)
    assert truth(s, 1)['unsafe']['U'] is False
    assert truth(s, 3)['unsafe']['U'] is True
    text = prompt(s, 3)
    assert 'repeated for exactly 3 ticks' in text
    assert 'not one-tick play' in text


def test_consistency_is_conditional_on_a_feasible_own_forecast():
    target = {'choice': 'U', 'unsafe': {'U': False, 'S': False, 'D': True}}
    own = {'U': True, 'S': False, 'D': True}
    result = score({'action': 'U', 'confidence': 1}, None, target, own)
    assert result['correct'] and result['contradiction']
    assert not result['oracle_unsafe']
    all_unsafe = {'U': True, 'S': True, 'D': True}
    result = score({'action': 'U', 'confidence': 1}, None, target, all_unsafe)
    assert not result['contradiction'] and result['failed_abstention']
    result = score({'action': None, 'confidence': 1}, None, target, all_unsafe)
    assert not result['failed_abstention']


def test_reference_abstains_only_if_no_safe_commitment():
    s = State(tuple(Car(4, lane, 1, 2, 1) for lane in (4, 5, 6)), lane=5)
    target = truth(s, 1)
    assert target['choice'] is None
    assert score({'action': None, 'confidence': 1}, None, target, target['unsafe'])['correct']


@pytest.mark.parametrize('value', [{'action': 'X', 'confidence': 1}, {'action': 'U', 'confidence': 2},
                                  {'action': 'U', 'confidence': 1, 'unsafe': {'U': False}}])
def test_bad_commitment_replies_stay_in_failure_denominator(value):
    target = {'choice': 'U', 'unsafe': {'U': False, 'S': False, 'D': False}}
    assert not score(value, None, target)['valid']


def test_complete_runner_keeps_oracle_upper_bounds_separate_and_resumes(tmp_path):
    fake = FakeOllama()
    fake.reply = {'action': 'D', 'unsafe': {'U': False, 'S': False, 'D': True}, 'confidence': .8}
    model = LocalModel('test', tmp_path / 'cache', transport=fake)
    s = initial(1000)
    bank = [{'id': s.key, 'split': 'test', 'state': s.to_dict(), 'oracle': label(s)}]
    result = run(bank, model, tmp_path / 'run', seeds=(0,))
    assert len(result) == len(CONDITIONS)
    assert next(r for r in result if r['condition'] == 'bound')['value']['action'] == 'U'
    assert next(r for r in result if r['condition'] == 'bound')['upper_bound'] is False
    assert next(r for r in result if r['condition'] == 'shield')['upper_bound'] is True
    count = len(fake.calls)
    assert run(bank, model, tmp_path / 'run', seeds=(0,)) == result
    assert len(fake.calls) == count
    manifest = json.loads((tmp_path / 'run/manifest.json').read_text())
    assert manifest['design']['protocol'] == 'repeated-action-commitment-v1'


def test_skipping_own_safe_progress_is_separate_from_unsafe_choice():
    target = {'choice': 'U', 'unsafe': {'U': False, 'S': False, 'D': False}}
    result = score({'action': 'S', 'confidence': .5}, None, target, target['unsafe'])
    assert not result['contradiction']
    assert result['skipped_progress'] and result['inconsistent']
    bound = score({'action': 'U', 'confidence': .5}, None, target, target['unsafe'])
    assert not bound['inconsistent']
