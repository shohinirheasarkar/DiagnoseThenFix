"""This keeps the board fixed while changing the diary or story, then checks what changed."""
import hashlib
import itertools
import json
from pathlib import Path
import random
import time

from .model import append_jsonl, canonical, parse_reply, read_jsonl
from .probes import LABELS, make_task
from .runner import source_fingerprint
from .world import ACTIONS, State, initial, oracle, repeated_outcome, step

HISTORIES = ('stay', 'up', 'down', 'oracle', 'random', 'own')
HISTORY_REPAIRS = ('base', 'no_history', 'window', 'delta', 'outcomes', 'balanced')


def diary(state, kind, length, seed=0, real=()):
    if kind not in HISTORIES or length < 0:
        raise ValueError('Unknown diary or negative length')
    if kind == 'own':
        if len(real) < length:
            raise ValueError('Not enough saved model turns for this diary')
        return list(real[-length:]) if length else []
    rng = random.Random(seed)
    source = state.source_seed + 1000000 + seed
    current, rows = initial(source), []
    for index in range(length):
        if current.lane == 0 or current.turn == 100:
            current = initial(source + index)
        action = ({'stay': 'S', 'up': 'U', 'down': 'D'}.get(kind)
                  or (oracle(current)['choice'] or 'S' if kind == 'oracle' else rng.choice(ACTIONS)))
        following, outcome = step(current, action)
        rows.append({'action': action, 'applied': True, 'events': [outcome],
                     'before': current.to_dict(), 'after': following.to_dict(),
                     'source': 'simulated diary', 'optimal': action in oracle(current)['optimal']})
        current = following
    return rows


def real_diary(episodes):
    rows = []
    for episode in episodes:
        for frame in episode['trace']:
            rows.append({'action': frame['response']['action'],
                         'applied': frame['action_applied'], 'events': frame['events'],
                         'before': frame['before'], 'after': frame['after'],
                         'source': 'saved model episode', 'episode_seed': episode['seed'],
                         'optimal': frame['response']['action'] in
                         oracle(State.from_dict(frame['before']))['optimal']})
    return rows


def history_text(rows, repair='base', outcomes=False, seed=0):
    if repair not in HISTORY_REPAIRS:
        raise ValueError('Unknown history repair')
    if repair == 'no_history':
        return ''
    if repair == 'window':
        rows = rows[-2:]
    if not rows:
        return ''
    lines = ['Diary from separate episodes, transplanted for this test. '
             'The current board below is authoritative.']
    balanced = [ACTIONS[i % 3] for i in range(len(rows))]
    random.Random(seed).shuffle(balanced)
    for index, row in enumerate(rows):
        if repair == 'delta':
            before, after = row['before']['cars'], row['after']['cars']
            changes = [b['x']-a['x'] for a, b in zip(before, after)]
            lines.append('Earlier episode head-column changes: ' + canonical(changes))
            continue
        action = balanced[index] if repair == 'balanced' else row['action']
        text = f'Previous proposed action: {action}.'
        if repair == 'balanced':
            text += ' (Balanced synthetic action; no outcome attached.)'
        elif outcomes or repair == 'outcomes':
            text += f' Applied: {row["applied"]}. Observed events: {canonical(row["events"])}.'
            if repair == 'outcomes':
                text += f' Oracle-optimal on its source board: {row["optimal"]}.'
        lines.append(text)
    return '\n'.join(lines) + '\n'


def task(row, stage, representation, seed, kind='stay', length=0, repair='base',
         outcomes=False, skin='cars', labels='standard', name_leak=False,
         reversal=False, paraphrase=0, real=()):
    if stage not in ('D', 'E') or (stage == 'E' and repair not in ('base', 'strategy')):
        raise ValueError('Unsupported context task')
    state = State.from_dict(row['state'])
    base = make_task(state, 'D' if stage == 'D' else 'E', representation, k=3, seed=seed,
                     paraphrase=paraphrase, skin=skin, labels=labels, name_leak=name_leak)
    records = diary(state, kind, length, seed, real) if stage == 'D' else []
    prefix = history_text(records, repair, outcomes, seed) if stage == 'D' else ''
    harmless = [0] if reversal else []
    if reversal:
        prefix += ('RULE OVERRIDE taking priority over the shared contact rule: '
                   'object 0 is safe to touch. It still moves normally, '
                   'but contact with it never resets or harms the player. '
                   'All other objects remain dangerous.\n')
    if stage == 'E' and repair == 'strategy':
        prefix += 'Wait for a gap, then cross.\n'
    target = State(tuple(c for i, c in enumerate(state.cars) if i not in harmless),
                   state.lane, state.turn, state.source_seed)
    truth = oracle(target)
    collision = {a: repeated_outcome(target, a, 3) == 'collision' for a in ACTIONS}
    shown = records[-2:] if repair == 'window' else records
    if repair in ('no_history', 'delta'):
        shown = []
    shown_actions = [r['action'] for r in shown]
    if repair == 'balanced':
        shown_actions = [ACTIONS[i % 3] for i in range(len(records))]
        random.Random(seed).shuffle(shown_actions)
    return {'stage': stage, 'representation': representation, 'seed': seed, 'kind': kind,
            'length': length, 'repair': repair, 'outcomes': outcomes, 'skin': skin,
            'labels': labels, 'name_leak': name_leak, 'reversal': reversal,
            'paraphrase': paraphrase, 'diary': records, 'shown_actions': shown_actions, 'truth': truth,
            'actual_collision': collision, 'prompt': prefix + base['prompt']}


def score(task, response):
    value, error = parse_reply(response, ('action', 'collision', 'confidence'))
    if not error:
        try:
            action = {label: a for a, label in LABELS[task['labels']].items()}[value['action']]
            beliefs = value['collision']
            if set(beliefs) != set(ACTIONS) or any(type(v) is not bool for v in beliefs.values()):
                raise ValueError('Expected three collision booleans')
            confidence = value['confidence']
            if type(confidence) not in (int, float) or not 0 <= confidence <= 1:
                raise ValueError('Invalid confidence')
            actions = task['shown_actions']
            counts = {a: actions.count(a) for a in ACTIONS}
            modes = [a for a in ACTIONS if counts[a] == max(counts.values())] if actions else []
            return {'valid': True, 'correct': action in task['truth']['optimal'],
                    'safe': action in task['truth']['safe'], 'action': action,
                    'confidence': confidence, 'collision': beliefs,
                    'forecast_accuracy': sum(beliefs[a] == task['actual_collision'][a]
                                             for a in ACTIONS)/3,
                    'history_match': action in modes if modes else None}
        except (ValueError, KeyError, TypeError) as exc:
            error = f'malformed: {exc}'
    return {'valid': False, 'correct': False, 'safe': False, 'action': None, 'error': error}


def conditions(stage, full=False):
    if stage == 'D':
        yield {'kind': 'stay', 'length': 0}
        for kind, length, outcomes in itertools.product(HISTORIES, (3, 6, 12), (False, True)):
            yield {'kind': kind, 'length': length, 'outcomes': outcomes}
        for repair in HISTORY_REPAIRS[1:]:
            yield {'kind': 'stay', 'length': 12, 'repair': repair}
    elif stage == 'E':
        for skin, labels in itertools.product(('cars', 'logs', 'clouds', 'X'), LABELS):
            yield {'skin': skin, 'labels': labels}
        yield {'name_leak': True}
        yield {'repair': 'strategy'}
        yield {'reversal': True}
        if full:
            for paraphrase in (1, 2):
                yield {'paraphrase': paraphrase}
    else:
        raise ValueError('Expected D or E')


def run(bank, model, output, stage, representation, real=(), limit=8, seeds=range(5),
        split='test', stop_at=None, thinking=False, max_tokens=256):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selected = sorted((r for r in bank if r['split'] == split), key=lambda r: r['id'])[:limit]
    if not selected or limit < 1:
        raise ValueError('No selected states')
    design = json.loads(canonical({'source': source_fingerprint(), 'model': model.name,
             'digest': model.digest, 'version': model.version, 'stage': stage,
             'representation': representation, 'states': selected, 'real': list(real),
             'seeds': list(seeds), 'conditions': list(conditions(stage, full=True)),
             'thinking': thinking, 'max_tokens': max_tokens, 'split': split}))
    manifest = output / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != design:
        raise ValueError('Different study design; use a new output directory')
    manifest.write_text(json.dumps(design, indent=2))
    path = output / 'responses.jsonl'
    done = {r['trial_id'] for r in read_jsonl(path)}
    for row, seed, condition in itertools.product(selected, seeds, design['conditions']):
        identity = {'design': hashlib.sha256(canonical(design).encode()).hexdigest(),
                    'state': row['id'], 'seed': seed, 'condition': condition}
        trial = hashlib.sha256(canonical(identity).encode()).hexdigest()
        if trial in done:
            continue
        if stop_at is not None and time.time() >= stop_at:
            break
        query = task(row, stage, representation, seed, real=real, **condition)
        response = model.ask(query['prompt'], seed=seed, thinking=thinking,
                             max_tokens=max_tokens, schema='json')
        append_jsonl(path, {'trial_id': trial, 'state_id': row['id'],
                           'split': split,
                           'source_seed': row['state']['source_seed'], 'model': model.name,
                           'digest': model.digest, 'task': query, 'response': response,
                           'score': score(query, response)})
        done.add(trial)
        print(f'{stage} {model.name} {len(done)} {row["id"]}', flush=True)
    return read_jsonl(path)


def summarize(records, replicates=2000):
    from collections import defaultdict
    from .analysis import bootstrap, holm
    groups = defaultdict(list)
    baselines = {}
    for record in records:
        t = record['task']
        fields = ('kind', 'length', 'repair', 'outcomes') if t['stage'] == 'D' else (
            'skin', 'labels', 'name_leak', 'reversal', 'repair', 'paraphrase')
        condition = canonical({k: t[k] for k in fields})
        groups[(record['model'], t['stage'], condition)].append(record)
        base = (t['length'] == 0 if t['stage'] == 'D' else
                t['skin'] == 'cars' and t['labels'] == 'standard' and
                not t['name_leak'] and not t['reversal'] and t['repair'] == 'base'
                and t['paraphrase'] == 0)
        if base:
            baselines[(record['model'], t['stage'], record['state_id'], t['seed'])] = record
    output = []
    for (model, stage, condition), values in sorted(groups.items()):
        paired, eligible, differences = [], 0, defaultdict(list)
        for record in values:
            s, t = record['score'], record['task']
            base = baselines.get((model, stage, record['state_id'], t['seed']))
            if base is None:
                continue
            b = base['score']
            delta = int(s['correct']) - int(b['correct'])
            differences[record['source_seed']].append(delta)
            valid = s['valid'] and b['valid']
            eligible += valid
            changed = s['action'] != b['action'] if valid else None
            paired.append({'state_id': record['source_seed'], 'accuracy_change': delta,
                           **({'action_change': changed,
                               'harmful_flip': changed and b['safe'] and not s['safe'],
                               'benign_flip': changed and b['safe'] and s['safe'],
                               'forecast_change': sum(s['collision'][a] != b['collision'][a]
                                                      for a in ACTIONS)/3}
                              if valid else {})})
        cluster_deltas = [sum(v)/len(v) for v in differences.values()]
        pvalue = sign_test(cluster_deltas) if cluster_deltas else None
        metric_rows = [{**r['score'], 'state_id': r['source_seed']} for r in values]
        output.append({'model': model, 'stage': stage, 'condition': json.loads(condition),
                       'trials': len(values), 'states': len({r['state_id'] for r in values}),
                       'invalid': sum(not r['score']['valid'] for r in values),
                       'paired_valid': eligible, 'paired_all': len(paired),
                       'accuracy': bootstrap(metric_rows, replicates=replicates),
                       'safety': bootstrap(metric_rows, 'safe', replicates=replicates),
                       'history_match_valid': bootstrap(metric_rows, 'history_match', replicates=replicates)
                       if any(r.get('history_match') is not None for r in metric_rows) else None,
                       **{metric: bootstrap(paired, metric, replicates=replicates)
                          for metric in ('accuracy_change', 'action_change', 'harmful_flip',
                                         'benign_flip', 'forecast_change')},
                       'p_sign_flip': pvalue})
    for stage in ('D', 'E'):
        family = [r for r in output if r['stage'] == stage and r['p_sign_flip'] is not None]
        for row, adjusted in zip(family, holm([r['p_sign_flip'] for r in family])):
            row['p_holm'] = adjusted
    return output


def sign_test(differences, seed=0, draws=20000):
    values = [x for x in differences if x != 0]
    if not values:
        return 1.
    observed = abs(sum(values))
    if len(values) <= 16:
        sums = (abs(sum(s*v for s, v in zip(signs, values)))
                for signs in itertools.product((-1, 1), repeat=len(values)))
        return sum(x >= observed-1e-12 for x in sums)/(2**len(values))
    rng = random.Random(seed)
    extreme = sum(abs(sum(rng.choice((-1, 1))*v for v in values)) >= observed-1e-12
                  for _ in range(draws))
    return (extreme+1)/(draws+1)


def verify(record, row):
    t = record['task']
    state = State.from_dict(row['state'])
    rebuilt = task(row, t['stage'], t['representation'], t['seed'],
                   **{k: t[k] for k in ('kind', 'length', 'repair', 'outcomes', 'skin',
                                        'labels', 'name_leak', 'reversal', 'paraphrase')},
                   real=t['diary'] if t['kind'] == 'own' else ())
    if canonical(rebuilt) != canonical(t):
        raise ValueError('Context prompt or diary mismatch')
    target = State(state.cars[1:] if t['reversal'] else state.cars,
                   state.lane, state.turn, state.source_seed)
    if t['truth'] != json.loads(canonical(oracle(target))):
        raise ValueError('Context oracle mismatch')
    actual = {a: repeated_outcome(target, a, 3) == 'collision' for a in ACTIONS}
    if t['actual_collision'] != actual:
        raise ValueError('Context forecast target mismatch')
    for turn in t['diary']:
        if turn['source'] == 'simulated diary':
            following, event = step(State.from_dict(turn['before']), turn['action'])
            if canonical(following.to_dict()) != canonical(turn['after']) or turn['events'] != [event]:
                raise ValueError('Fabricated diary outcome mismatch')
    if score(t, record['response']) != record['score']:
        raise ValueError('Context score mismatch')
    return True


def main():
    import argparse
    from .model import LocalModel
    parser = argparse.ArgumentParser(description='Paired diary and framing tests')
    parser.add_argument('--model')
    parser.add_argument('--bank', type=Path, default=Path('data/freeway.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('D', 'E'), default='D')
    parser.add_argument('--representation', default='coordinates')
    parser.add_argument('--real-log', type=Path)
    parser.add_argument('--split', choices=('validation', 'test'), default='test')
    parser.add_argument('--limit', type=int, default=8)
    parser.add_argument('--thinking', action='store_true')
    parser.add_argument('--check', action='store_true', help='Rescore saved replies, with no model calls')
    args = parser.parse_args()
    bank = json.loads(args.bank.read_text())['situations']
    if args.check:
        records = read_jsonl(args.output / 'responses.jsonl')
        by_id = {r['id']: r for r in bank}
        for record in records:
            verify(record, by_id[record['state_id']])
    else:
        if not args.model or (args.stage == 'D' and not args.real_log):
            parser.error('Supply --model and, for Stage D, --real-log from live play')
        real = real_diary(read_jsonl(args.real_log)) if args.real_log else ()
        model = LocalModel(args.model)
        records = run(bank, model, args.output, args.stage, args.representation, real=real,
                      split=args.split, limit=args.limit, thinking=args.thinking)
    (args.output / 'summary.json').write_text(json.dumps(summarize(records), indent=2))
    print(f'Checked {len(records)} trials.')


if __name__ == '__main__':
    main()
