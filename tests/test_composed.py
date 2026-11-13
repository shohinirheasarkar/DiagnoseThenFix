"""These checks cover the composed agent's actual calls and ensure it never substitutes oracle forecasts."""
import json

from diagnose.composed import ComposedPolicy, forecast_with_code
from diagnose.repairs import ArithmeticProgram
from diagnose.world import initial, repeated_outcome
from test_repairs_live import CODE


class Model:
    def __init__(self):
        self.calls = 0

    def ask(self, prompt, **kwargs):
        self.calls += 1
        answer = {'code': CODE, 'confidence': 1} if 'Write one Python function' in prompt else {
            'action': 'D', 'collision': {'U': False, 'S': False, 'D': True}, 'confidence': .5}
        return {'error': None, 'wall_seconds': .1, 'raw': {'eval_count': 10,
                'message': {'content': json.dumps(answer)}, 'done_reason': 'stop'}}


def test_generated_program_forecast_matches_world_on_unseen_boards():
    program = ArithmeticProgram(CODE)
    for seed in range(20, 30):
        state = initial(1000+seed)
        forecast = forecast_with_code(state, program, 10)
        assert forecast == {a: repeated_outcome(state, a, 10) == 'collision' for a in ('U', 'S', 'D')}


def test_composition_executes_selected_repairs_and_counts_cost():
    model = Model()
    profile = {'split': 'validation', 'B': .1, 'C': .1}
    policy = ComposedPolicy(model, profile)
    state = initial(1000)
    result = policy(state, [])
    assert result['error'] is None
    assert result['repairs'] == ['bound', 'code']
    assert result['calls'] == model.calls == 2
    assert result['tokens'] == 20 and result['seconds'] == .2
    second = policy(state, [])
    assert second['calls'] == 1
    assert second['action'] == result['action']


def test_monitor_changes_a_contradictory_action_without_an_oracle():
    model = Model()
    policy = ComposedPolicy(model, {'split': 'validation'}, condition='monitor')
    result = policy(initial(1000), [])
    assert result['action'] == 'U' and result['repairs'] == ['bound']
    assert result['calls'] == 1


def test_calibration_failure_is_logged_without_an_oracle_fallback():
    class MissingScores(Model):
        def action_scores(self, prompt, seed):
            return {'scores': None, 'error': 'missing candidate', 'response': self.ask(prompt)}
    policy = ComposedPolicy(MissingScores(), {'split': 'validation', 'E': .1})
    result = policy(initial(1000), [])
    assert result['error'].startswith('Calibration unavailable')
    assert result['action'] == 'S' and result['collision'] is None
    assert result['calls'] == 3 and result['tokens'] == 30


def test_all_repairs_count_scoring_calls_and_history_filtering():
    class Scores(Model):
        def action_scores(self, prompt, seed):
            return {'scores': {'U': -2., 'S': -1., 'D': -3.}, 'error': None,
                    'response': self.ask(prompt)}
    model = Scores()
    policy = ComposedPolicy(model, {'split': 'validation'}, condition='all')
    result = policy(initial(1000), [{'observation': 'old', 'answer': 'old', 'action': 'S'}])
    assert result['error'] is None and result['calls'] == 4
    assert result['seconds'] == .4 and result['tokens'] == 40
    assert result['repairs'] == ['bound', 'calibrate', 'code', 'no_history', 'structured']


def test_v2_checks_program_only_on_fixed_training_inputs_and_reuses_it():
    from diagnose.composed import program_error
    assert program_error(ArithmeticProgram(CODE, extended=True)) is None
    wrong = ArithmeticProgram('def forecast(x,timer,p,k):\n return x', extended=True)
    assert program_error(wrong)['expected'] != program_error(wrong)['predicted']
    model = Model()
    policy = ComposedPolicy(model, {'split': 'validation', 'B': .1, 'C': .1}, revision='v2')
    first = policy(initial(2222), [])
    assert first['error'] is None and first['calls'] == 2
    assert policy(initial(2223), [])['calls'] == 1


def test_v2_retries_bad_code_once_then_uses_its_own_beliefs():
    class Bad(Model):
        def ask(self, prompt, **kwargs):
            raw = super().ask(prompt, **kwargs)
            if 'Write one Python function' in prompt:
                raw['raw']['message']['content'] = json.dumps({'code': 'def forecast(x,timer,p,k):\n return x', 'confidence': 1})
            return raw
    model = Bad()
    policy = ComposedPolicy(model, {'split': 'validation', 'B': .1, 'C': .1}, revision='v2')
    first = policy(initial(2222), [])
    assert first['calls'] == 3 and first['action'] == 'U'
    assert first['collision'] == {'U': False, 'S': False, 'D': True}
    assert 'Training-case failure' in first['error']
    second = policy(initial(2223), [])
    assert second['calls'] == 1 and second['action'] == 'U'
    assert 'disabled' in second['error']
    original_failure = first['program_failure']
    for seed in range(2224, 2234):
        later = policy(initial(seed), [])
        assert later['calls'] == 1 and later['action'] == 'U'
        assert later['program_failure'] == original_failure
        assert later['error'] == second['error']
    assert model.calls == 14


def test_v2_calibration_failure_keeps_the_proposed_action_and_logs_failure():
    class Missing(Model):
        def action_scores(self, prompt, seed):
            return {'scores': None, 'error': 'missing candidate', 'response': self.ask(prompt)}
    result = ComposedPolicy(Missing(), {'split': 'validation', 'E': .1}, revision='v2')(initial(1000), [])
    assert result['action'] == 'D' and result['collision'] is not None
    assert result['error'].startswith('Calibration unavailable')
    assert result['calls'] == 3


def test_v2_counts_both_code_attempts_when_feedback_repairs_the_program():
    class Fixable(Model):
        attempts = 0
        def ask(self, prompt, **kwargs):
            raw = super().ask(prompt, **kwargs)
            if 'Write one Python function' in prompt:
                self.attempts += 1
                if self.attempts == 1:
                    raw['raw']['message']['content'] = json.dumps({'code': 'def forecast(x,timer,p,k):\n return x', 'confidence': 1})
                else:
                    assert 'fixed training-only check' in prompt
            return raw
    model = Fixable()
    result = ComposedPolicy(model, {'split': 'validation', 'B': .1}, revision='v2')(initial(2222), [])
    assert result['error'] is None and result['calls'] == 4 and model.attempts == 2
    assert result['tokens'] == 40


def test_code_forecast_reaches_action_readout_when_binding_is_off():
    class Routes(Model):
        def ask(self, prompt, **kwargs):
            raw = super().ask(prompt, **kwargs)
            if 'Forecast (repeated-action collision' in prompt:
                assert 'Choose an action.' in prompt
                raw['raw']['message']['content'] = json.dumps({'action': 'S', 'confidence': 1})
            return raw
    model = Routes()
    result = ComposedPolicy(model, {'split': 'validation', 'B': .1}, revision='v2')(initial(2222), [])
    assert result['action'] == 'S' and result['calls'] == 3
    assert result['program_validated'] and result['cpu_seconds'] > 0
    assert result['seconds'] >= .3


def test_calibrated_readout_conditions_on_repaired_forecast():
    class Scores(Model):
        prompts = []
        def action_scores(self, prompt, seed):
            self.prompts.append(prompt)
            return {'scores': {'U': -1., 'S': -2., 'D': -3.}, 'error': None, 'response': self.ask(prompt)}
    model = Scores()
    result = ComposedPolicy(model, {'split': 'validation', 'B': .1, 'E': .1}, revision='v2')(initial(2222), [])
    assert result['error'] is None
    assert 'collision forecast at horizon 3' in model.prompts[0]
    assert 'collision forecast' not in model.prompts[1] and 'BOARD MASKED' in model.prompts[1]
