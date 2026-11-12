"""This lets the world move while an agent thinks, and saves each decision for later replay."""
from dataclasses import asdict
import math
import random
import time

from .model import parse_reply
from .probes import make_task
from .repairs import bound_action
from .world import ACTIONS, initial, label, oracle, repeated_outcome, step


def baseline(name, seed=0, horizon=10):
    rng = random.Random(seed)

    def decide(state, history):
        if name == 'random':
            action = rng.choice(ACTIONS)
        elif name == 'stay':
            action = 'S'
        elif name == 'advance':
            action = 'U'
        elif name == 'oracle':
            action = oracle(state, horizon)['choice'] or 'S'
        elif name == 'planner':
            depth = label(state, horizon)['required_foresight'] or horizon
            action = oracle(state, depth)['choice'] or 'S'
        else:
            raise ValueError(name)
        return {'action': action, 'seconds': 0., 'tokens': 0, 'calls': 0,
                'collision': None, 'error': None}
    return decide


def model_policy(model, seed=0, thinking=False, representation='coordinates',
                 repair='base', max_tokens=512, horizon=3):
    if repair not in ('base', 'bound', 'no_history', 'generic', 'oracle_shield'):
        raise ValueError('Unsupported live repair')

    def decide(state, history):
        task = make_task(state, 'C', representation, k=horizon,
                         repair='generic' if repair == 'generic' else 'base')
        messages = []
        if repair != 'no_history':
            for row in history[-12:]:
                messages.append({'role': 'user', 'content': row['observation']})
                messages.append({'role': 'assistant', 'content': row['answer']})
        result = model.ask(task['prompt'], seed=seed + state.turn, thinking=thinking,
                           max_tokens=max_tokens, schema='json', history=messages)
        value, error = parse_reply(result, ['action', 'collision', 'confidence'])
        beliefs = None
        action = 'S'
        if not error:
            try:
                action = value['action']
                if action not in ACTIONS:
                    raise ValueError('Invalid action')
                beliefs = value['collision']
                bound_action(beliefs)
                if repair == 'bound':
                    action = bound_action(beliefs) or 'S'
                elif repair == 'oracle_shield':
                    safe = oracle(state, horizon)['safe']
                    if action not in safe:
                        action = safe[0] if safe else 'S'
            except (KeyError, TypeError, ValueError) as exc:
                action, beliefs, error = 'S', None, str(exc)
        return {'action': action, 'collision': beliefs, 'seconds': result['wall_seconds'],
                'tokens': result['raw'].get('eval_count', 0), 'calls': 1, 'error': error,
                'response': result, 'observation': task['prompt'],
                'answer': result['raw'].get('message', {}).get('content', '')}
    return decide


def episode(policy, seed=1000, tick_seconds=1., start=None, prior_history=(),
            intervention=None, max_decisions=100, stop_at=None):
    if tick_seconds <= 0:
        raise ValueError('Tick duration must be positive')
    state = start or initial(seed)
    history = list(prior_history)
    trace = []
    collisions = tokens = calls = stays = longest = elapsed_ticks = 0
    success = state.lane == 0
    for decision in range(max_decisions):
        if stop_at is not None and time.time() >= stop_at:
            break
        if state.lane == 0 or state.turn >= 100:
            break
        observation = state
        response = policy(state, history)
        if intervention is not None:
            response = intervention(decision, state, response)
        action = response['action']
        if action not in ACTIONS:
            raise ValueError('Policy returned an invalid action')
        delay = math.ceil(max(0., response.get('seconds', 0.)) / tick_seconds)
        events = []
        for _ in range(delay):
            state, event = step(state, 'S')
            elapsed_ticks += 1
            collisions += event == 'collision'
            events.append(event)
            if event in ('goal', 'timeout') or state.turn == 100:
                break
        applied = state.lane != 0 and state.turn < 100
        if applied:
            state, event = step(state, action)
            elapsed_ticks += 1
            collisions += event == 'collision'
            events.append(event)
        success = state.lane == 0
        tokens += response.get('tokens', 0)
        calls += response.get('calls', 0)
        stays = stays + 1 if action == 'S' else 0
        longest = max(longest, stays)
        forecast = response.get('collision')
        actual = {a: repeated_outcome(observation, a, min(3, 100-observation.turn)) == 'collision'
                  for a in ACTIONS}
        trace.append({'decision': decision, 'before': asdict(observation), 'after': asdict(state),
                      'response': response, 'delay_ticks': delay, 'action_applied': applied,
                      'events': events, 'stage_labels': {
                          'forecast_correct': forecast == actual if forecast is not None else None,
                          'contradiction': forecast[action] if forecast is not None else None,
                          'history_mode_match': action == max(ACTIONS, key=lambda a: sum(
                              r.get('action') == a for r in history)) if history else None}})
        history.append({'observation': response.get('observation', ''),
                        'answer': response.get('answer', action), 'action': action,
                        'outcomes': events})
        if success or state.turn >= 100:
            break
    return {'seed': seed, 'success': success, 'collisions': collisions, 'tokens': tokens,
            'calls': calls, 'ticks': elapsed_ticks, 'longest_stay_loop': longest,
            'final_lane': state.lane, 'trace': trace, 'tick_seconds': tick_seconds,
            'censored': not success and state.turn < 100}


def critical_step(failed, replay):
    if failed['success']:
        return None
    for index, row in enumerate(failed['trace']):
        for stage in ('A', 'B', 'C', 'D', 'E'):
            outcome = replay(index, stage, row)
            if outcome['success']:
                return {'decision': index, 'stage': stage, 'result': outcome}
    return None
