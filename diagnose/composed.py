"""This turns on fixes selected from validation data and logs every extra call they cost."""
from dataclasses import asdict

from .model import parse_reply
from .probes import make_task
from .render import RULES, render
from .repairs import ArithmeticProgram, ablations, bound_action, calibrate, choose_repairs, monitor
from .world import ACTIONS


def forecast_with_code(state, program, horizon=3):
    beliefs = {}
    for action in ACTIONS:
        lane = state.lane
        collision = False
        for tick in range(1, horizon + 1):
            lane = max(0, min(9, lane + {'U': -1, 'S': 0, 'D': 1}[action]))
            if lane == 0:
                break
            for car in state.cars:
                head = program(car.x, car.timer, car.speed, tick)
                direction = -1 if car.speed > 0 else 1
                if car.lane == lane and any(head + offset * direction == 4
                                            for offset in range(car.length)):
                    collision = True
                    break
            if collision:
                break
        beliefs[action] = collision
    return beliefs


class ComposedPolicy:
    def __init__(self, model, profile, condition='selected', seed=0, thinking=False,
                 horizon=3, max_tokens=1024):
        self.model, self.profile, self.seed = model, profile, seed
        selected = choose_repairs(profile)
        comparisons = ablations(selected, seed)
        if condition == 'monitor':
            self.enabled = set()
        elif condition in comparisons:
            self.enabled = comparisons[condition]
        else:
            raise ValueError('Unknown composition condition')
        self.condition, self.thinking, self.horizon = condition, thinking, horizon
        self.max_tokens, self.program, self.previous = max_tokens, None, None

    def __call__(self, state, history):
        enabled = set(self.enabled)
        representation = self.profile.get('best_representation', 'coordinates') if 'structured' in enabled else 'ascii'
        task = make_task(state, 'C', representation, k=self.horizon, kind='combined')
        prompt = task['prompt'] + ('\nAlso return "nearby_ids" (all object IDs with a cell within '
                                  'Manhattan distance 2) and "future_heads" (all head columns '
                                  f'after {self.horizon} car ticks).')
        messages = []
        if 'no_history' not in enabled:
            for row in history[-12:]:
                messages.extend(({'role': 'user', 'content': row.get('observation', '')},
                                 {'role': 'assistant', 'content': row.get('answer', '')}))
        responses = []

        def ask(text, **kwargs):
            result = self.model.ask(text, seed=self.seed+state.turn, thinking=self.thinking,
                                    max_tokens=self.max_tokens, schema='json', **kwargs)
            responses.append(result)
            return result

        raw = ask(prompt, history=messages)
        value, error = parse_reply(raw, ('action', 'collision', 'confidence'))
        if error:
            return self._result('S', None, error, responses, prompt, raw, enabled)
        try:
            action, beliefs = value['action'], value['collision']
            if action not in ACTIONS:
                raise ValueError('Invalid proposed action')
            bound_action(beliefs)
            if self.condition == 'monitor':
                stays = 0
                for row in reversed(history):
                    if row.get('action') != 'S':
                        break
                    stays += 1
                expected = [i for i, car in enumerate(state.cars)
                            if abs(car.lane-state.lane) + min(abs(x-4) for x in car.cells()) <= 2]
                restated = value.get('nearby_ids')
                perception_ok = restated is None or restated == expected
                heads = value.get('future_heads')
                linear_error = None
                if isinstance(heads, list) and len(heads) == len(state.cars):
                    if all(type(x) in (int, float) for x in heads):
                        linear_error = max(abs(x - ((c.x + self.horizon/c.speed) % 9))
                                           for x, c in zip(heads, state.cars))
                enabled.update(monitor(stays, beliefs, action, linear_error, perception_ok))
                if 'structured' in enabled:
                    structured = make_task(state, 'C', self.profile.get('best_representation', 'coordinates'),
                                           k=self.horizon, kind='combined')
                    redone = ask(structured['prompt'], history=messages)
                    replacement, failed = parse_reply(redone, ('action', 'collision', 'confidence'))
                    if failed:
                        raise ValueError('Structured retry failed: ' + failed)
                    action, beliefs = replacement['action'], replacement['collision']
                    bound_action(beliefs)
                if 'no_history' in enabled:
                    cleared = ask(prompt)
                    replacement, failed = parse_reply(cleared, ('action', 'collision', 'confidence'))
                    if failed:
                        raise ValueError('History retry failed: ' + failed)
                    action, beliefs = replacement['action'], replacement['collision']
                    bound_action(beliefs)
            if 'code' in enabled:
                if self.program is None:
                    code_task = make_task(state, 'B', k=self.horizon, repair='code')
                    generated = ask(code_task['prompt'])
                    code_value, failed = parse_reply(generated, ('code', 'confidence'))
                    if failed:
                        raise ValueError('Code generation failed: ' + failed)
                    self.program = ArithmeticProgram(code_value['code'])
                beliefs = forecast_with_code(state, self.program, self.horizon)
            if 'calibrate' in enabled:
                common = RULES + '\nA means U, B means S, C means D.\n'
                state_scores = self.model.action_scores(common + render(state, representation), self.seed)
                null_scores = self.model.action_scores(common + 'BOARD MASKED.', self.seed)
                responses.extend((state_scores['response'], null_scores['response']))
                if state_scores['error'] or null_scores['error']:
                    raise ValueError('Calibration unavailable: ' + str(state_scores['error'] or null_scores['error']))
                action, _ = calibrate(state_scores['scores'], null_scores['scores'])
            if 'bound' in enabled:
                action = bound_action(beliefs) or 'S'
            if action not in ACTIONS:
                raise ValueError('Repair returned invalid action')
            self.previous = asdict(state)
            return self._result(action, beliefs, None, responses, prompt, raw, enabled)
        except (ValueError, KeyError, TypeError, SyntaxError, ArithmeticError) as exc:
            return self._result('S', None, str(exc), responses, prompt, raw, enabled)

    @staticmethod
    def _result(action, beliefs, error, responses, prompt, raw, enabled):
        return {'action': action, 'collision': beliefs, 'error': error,
                'seconds': sum(r['wall_seconds'] for r in responses),
                'tokens': sum(r['raw'].get('eval_count', 0) for r in responses),
                'calls': len(responses), 'responses': responses, 'repairs': sorted(enabled),
                'observation': prompt, 'answer': raw['raw'].get('message', {}).get('content', '')}
