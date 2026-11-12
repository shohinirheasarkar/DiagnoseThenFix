"""These questions separate reading, predicting, acting, copying history, and following priors."""
import json
import random

from .render import RULES, render
from .world import ACTIONS, futures, repeated_outcome

PARAPHRASES = ('Read the board carefully.', 'Use only the given board and rules.',
               'Work from this exact state.')
LABELS = {'standard': {'U': 'U', 'S': 'S', 'D': 'D'},
          'neutral': {'U': 'A', 'S': 'B', 'D': 'C'},
          'words': {'U': 'FORWARD', 'S': 'WAIT', 'D': 'BACK'}}


def perception(state, kind, object_id=0):
    car = state.cars[object_id]
    if kind == 'occupancy':
        question = f'Is cell (lane {car.lane}, column 4) occupied? Answer boolean.'
        answer = any(c.lane == car.lane and 4 in c.cells() for c in state.cars)
    elif kind == 'nearest':
        distances = [abs(c.lane - state.lane) + min(abs(x - 4) for x in c.cells())
                     for c in state.cars]
        nearest = min(range(len(distances)), key=lambda i: (distances[i], i))
        question = ('Which object is nearest by Manhattan distance from any occupied cell '
                    'to the player? Break ties by smaller ID. Answer its ID.')
        answer = nearest
    elif kind == 'count':
        question = 'How many objects have an occupied cell within Manhattan distance 2 of player?'
        answer = sum(abs(c.lane - state.lane) + min(abs(x - 4) for x in c.cells()) <= 2
                     for c in state.cars)
    elif kind == 'motion':
        question = f'What is object {object_id} speed_parameter? Answer a number, including sign.'
        answer = car.speed
    elif kind == 'position':
        question = f'What is the head column of object {object_id}? Answer an integer.'
        answer = car.x
    else:
        raise ValueError(kind)
    return question, answer


def make_task(state, stage, representation='coordinates', k=1, seed=0,
              paraphrase=0, kind='position', repair='base', history_kind='none',
              history_length=0, skin='cars', labels='standard', name_leak=False,
              injected=None):
    if not 1 <= k <= 10 or paraphrase not in range(3):
        raise ValueError('Invalid horizon or paraphrase')
    if not state.cars:
        raise ValueError('A probe needs at least one object')
    allowed = {'A': {'base'}, 'B': {'base', 'rule', 'scratchpad', 'chained', 'tool', 'code'},
               'C': {'base', 'generic', 'forecast_then_act', 'bound', 'retry', 'oracle_shield'},
               'D': {'base', 'no_history', 'window', 'balanced'},
               'E': {'base', 'strategy'}, 'action': {'base', 'generic'}}
    if stage not in allowed or repair not in allowed[stage]:
        raise ValueError(f'Unsupported stage/repair: {stage}/{repair}')
    header = PARAPHRASES[paraphrase] + '\n' + RULES + '\n'
    header += f'All moving objects in this framing are called {skin}. Their contact rules are identical.\n'
    if name_leak:
        header += 'This game is like Frogger / Freeway.\n'
    board = render(state, representation, skin)
    truth = None
    if stage == 'A':
        question, truth = perception(state, kind)
        question += '\nReturn JSON {"answer": ..., "confidence": number from 0 to 1}.'
    elif stage == 'B':
        if kind == 'collision':
            question = f'If U is repeated for {k} ticks, will any collision occur? Answer boolean.'
            truth = repeated_outcome(state, 'U', k) == 'collision'
        else:
            question = f'What is object 0 head column after exactly {k} car ticks? Answer integer.'
            truth = futures(state, k)[-1][0]
        if repair == 'code':
            if kind != 'position':
                raise ValueError('Code offload currently scores position forecasts only')
            return {'stage': stage, 'representation': representation, 'k': k, 'seed': seed,
                    'paraphrase': paraphrase, 'kind': kind, 'repair': repair,
                    'history_kind': history_kind, 'history_length': history_length,
                    'skin': skin, 'labels': labels, 'name_leak': name_leak, 'truth': truth,
                    'prompt': PARAPHRASES[paraphrase] + '\n' + RULES +
                    '\nWrite one Python function forecast(x, timer, p, k) returning the head '
                    'column after k ticks. Only arithmetic, assignments, if/elif, and for '
                    'loops over range are supported. Calls allowed: abs,int,min,max,range. '
                    'No imports, attributes, recursion, lists, or other functions. '
                    'Return JSON {"code": "function source", "confidence": number from 0 to 1}.'}
        if repair == 'rule':
            question += '\nApply the provided timer and delayed-wrap rules explicitly.'
        elif repair == 'scratchpad':
            question += '\nFirst fill a tick-by-tick table in the JSON field "table".'
        elif repair == 'chained':
            question += '\nCompute each intermediate tick in sequence; include "table" in JSON.'
        elif repair == 'tool':
            question += '\nDeterministic forecast tool result: ' + json.dumps(truth)
        question += '\nReturn JSON {"answer": ..., "confidence": number from 0 to 1}.'
    elif stage in ('C', 'D', 'E', 'action'):
        mapping = LABELS[labels]
        header += 'Action labels: ' + ', '.join(f'{v} means {a}' for a, v in mapping.items()) + '.\n'
        if injected is not None:
            header += 'Forecast (repeated-action collision at the stated horizon): ' + json.dumps(injected) + '\n'
        if stage == 'E' and kind == 'null':
            board = 'BOARD MASKED. No state information is available.'
        question = ('Choose an action. Also forecast whether repeating each action for '
                    f'{k} ticks would collide. Return JSON '
                    '{"action": "label", "collision": {"U": bool, "S": bool, "D": bool}, '
                    '"confidence": number from 0 to 1}.')
        if stage == 'C' and kind == 'forecast':
            question = (f'Forecast collision for repeating each action for {k} ticks. '
                        'Return JSON {"collision": {"U": bool, "S": bool, "D": bool}, '
                        '"confidence": number from 0 to 1}. Do not choose an action.')
        elif stage == 'C' and kind in ('action', 'own', 'oracle'):
            question = 'Choose an action. Return JSON {"action": "label", "confidence": number from 0 to 1}.'
        if repair == 'generic':
            question += '\nThink carefully and check your answer.'
        elif repair == 'forecast_then_act':
            question += '\nForecast first, then pick a progressing action your forecast marks safe.'
        elif repair == 'strategy':
            question += '\nWait for a gap, then cross.'
    else:
        raise ValueError(stage)
    diary = history(state, history_kind, history_length, seed)
    if repair == 'no_history':
        diary = []
    elif repair == 'window':
        diary = diary[-2:]
    elif repair == 'balanced':
        diary = history(state, 'balanced', history_length, seed)
    if diary:
        header += 'Recent turns:\n' + '\n'.join(diary) + '\n'
    return {'stage': stage, 'representation': representation, 'k': k, 'seed': seed,
            'paraphrase': paraphrase, 'kind': kind, 'repair': repair,
            'history_kind': history_kind, 'history_length': history_length,
            'skin': skin, 'labels': labels, 'name_leak': name_leak, 'truth': truth,
            'prompt': header + board + '\n' + question}


def history(state, kind, length, seed=0):
    if length < 0:
        raise ValueError('Negative history length')
    if kind == 'none':
        return []
    rng = random.Random(seed)
    if kind in ('stay', 'up', 'down'):
        actions = [{'stay': 'S', 'up': 'U', 'down': 'D'}[kind]] * length
    elif kind in ('random', 'balanced'):
        actions = ([ACTIONS[i % 3] for i in range(length)] if kind == 'balanced'
                   else [rng.choice(ACTIONS) for _ in range(length)])
    elif kind == 'oracle':
        from .world import oracle
        actions = [oracle(state)['choice'] or 'S'] * length
    else:
        raise ValueError('Real model histories must come from live logs, not fabrication')
    return [f'Previous action: {a}.' for a in actions]


def score(task, value, error, state, truth):
    if error is not None:
        return {'valid': False, 'correct': False, 'error': error}
    try:
        confidence = value['confidence']
        if type(confidence) not in (int, float) or not 0 <= confidence <= 1:
            raise ValueError('confidence must be in [0,1]')
        if task['stage'] in ('A', 'B'):
            answer = value['answer']
            target = task['truth']
            if isinstance(target, bool):
                valid = isinstance(answer, bool)
            else:
                valid = type(answer) in (int, float) and not isinstance(answer, bool)
            if not valid:
                raise ValueError('answer type mismatch')
            result = {'valid': True, 'correct': answer == target, 'confidence': confidence}
            if type(target) in (int, float):
                result['signed_error'] = answer - target
            return result
        if task['stage'] == 'C' and task['kind'] == 'forecast':
            beliefs = value['collision']
            if set(beliefs) != set(ACTIONS) or any(type(v) is not bool for v in beliefs.values()):
                raise ValueError('invalid forecast')
            actual = {a: repeated_outcome(state, a, task['k']) == 'collision' for a in ACTIONS}
            return {'valid': True, 'correct': beliefs == actual, 'confidence': confidence,
                    'own_collision': beliefs, 'actual_collision': actual}
        reverse = {v: a for a, v in LABELS[task['labels']].items()}
        action = reverse[value['action']]
        if task['stage'] == 'C' and task['kind'] in ('action', 'own', 'oracle'):
            return {'valid': True, 'correct': action in truth['optimal'],
                    'safe': action in truth['safe'], 'action': action, 'confidence': confidence}
        collision = value['collision']
        if set(collision) != set(ACTIONS) or any(type(v) is not bool for v in collision.values()):
            raise ValueError('collision forecast must contain three booleans')
        actual = {a: repeated_outcome(state, a, task['k']) == 'collision' for a in ACTIONS}
        return {'valid': True, 'correct': action in truth['optimal'],
                'safe': action in truth['safe'], 'action': action,
                'contradiction': collision[action], 'confidence': confidence,
                'forecast_accuracy': sum(actual[a] == collision[a] for a in ACTIONS) / 3,
                'own_collision': collision, 'actual_collision': actual,
                'history_match': action == {'stay': 'S', 'up': 'U', 'down': 'D'}.get(task['history_kind'])}
    except (ValueError, KeyError, TypeError) as exc:
        return {'valid': False, 'correct': False, 'error': f'malformed: {exc}'}


def inventory():
    return {'A': ['occupancy', 'nearest', 'count', 'motion', 'position'],
            'B': ['position', 'collision'], 'C': ['forecast', 'action', 'own', 'oracle', 'combined'], 'D': ['action'], 'E': ['action', 'null']}
