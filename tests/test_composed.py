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
