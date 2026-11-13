"""This asks an agent to commit to the same repeated action its forecast is about."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .model import LocalModel, append_jsonl, canonical, parse_reply, read_jsonl
from .render import RULES, render
from .repairs import bound_action
from .world import ACTIONS, State, repeated_outcome

CONDITIONS = ('none', 'own', 'oracle', 'combined', 'generic', 'bound', 'retry', 'shield')


def truth(state, horizon):
    outcomes = {a: repeated_outcome(state, a, horizon) for a in ACTIONS}
    unsafe = {a: outcome in ('collision', 'timeout') for a, outcome in outcomes.items()}
    return {'unsafe': unsafe, 'choice': bound_action(unsafe), 'outcomes': outcomes}


def prompt(state, horizon=3, kind='combined', representation='coordinates', injected=None):
    if not 1 <= horizon <= 10 or kind not in ('forecast', 'action', 'combined', 'generic'):
        raise ValueError('Invalid committed-action task')
    text = RULES + '\n' + render(state, representation)
    text += (f'\nMake ONE decision now. Your chosen action will be repeated for exactly {horizon} '
             'ticks with no chance to change it. A goal ends the commitment successfully; a collision '
             'or game timeout ends it unsuccessfully. Predict unsafe=true for an unsuccessful commitment. '
             'Prefer U if it is safe, otherwise S if safe, otherwise D if safe. '
             'If all three are unsafe, abstain with action=null. This is a commitment task, not one-tick play.\n')
    if injected is not None:
        text += 'Provided forecast: ' + canonical(injected) + '\n'
    if kind == 'forecast':
        text += 'Forecast only. Return JSON {"unsafe":{"U":bool,"S":bool,"D":bool},"confidence":number from 0 to 1}.'
    elif kind == 'action':
        text += 'Choose only. Return JSON {"action":"U/S/D or null","confidence":number from 0 to 1}.'
    else:
        text += 'Return JSON {"action":"U/S/D or null","unsafe":{"U":bool,"S":bool,"D":bool},"confidence":number from 0 to 1}.'
    if kind == 'generic':
        text += '\nThink carefully and check your answer.'
    return text


def validate_beliefs(beliefs):
    bound_action(beliefs)
    return beliefs


def score(value, error, target, beliefs=None):
    if error:
        return {'valid': False, 'correct': False, 'error': error}
    try:
        action, confidence = value['action'], value['confidence']
        if action not in (*ACTIONS, None) or type(confidence) not in (int, float) or not 0 <= confidence <= 1:
            raise ValueError('Invalid action or confidence')
        if beliefs is None:
            beliefs = value.get('unsafe')
        if beliefs is not None:
            validate_beliefs(beliefs)
        result = {'valid': True, 'correct': action == target['choice'], 'action': action,
                  'confidence': confidence, 'oracle_unsafe': action is not None and target['unsafe'][action]}
        if beliefs is not None:
            feasible = any(not v for v in beliefs.values())
            result.update(beliefs=beliefs, own_feasible=feasible,
                          contradiction=bool(feasible and action is not None and beliefs[action]),
                          failed_abstention=bool(not feasible and action is not None),
                          unnecessary_abstention=bool(feasible and action is None),
                          skipped_progress=bool(not beliefs['U'] and action != 'U'),
                          inconsistent=bool((feasible and action is not None and beliefs[action])
                                            or (not beliefs['U'] and action != 'U')
                                            or (not feasible and action is not None)),
                          forecast_accuracy=sum(beliefs[a] == target['unsafe'][a] for a in ACTIONS)/3)
        return result
    except (ValueError, TypeError, KeyError) as exc:
        return {'valid': False, 'correct': False, 'error': f'malformed: {exc}'}


def run(bank, model, output, limit=20, seeds=tuple(range(5)), horizon=3,
        representation='coordinates', split='test', max_tokens=512, stop_at=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selected = sorted((r for r in bank if r['split'] == split), key=lambda r: r['id'])[:limit]
    if not selected or limit < 1:
        raise ValueError('No selected states')
    folder = Path(__file__).parent
    source_files = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                    for name in ('commitment.py', 'model.py', 'render.py', 'repairs.py', 'world.py')}
    design = json.loads(canonical({'protocol': 'repeated-action-commitment-v1',
              'model': model.name, 'digest': model.digest, 'version': model.version,
              'states': [r['id'] for r in selected], 'seeds': seeds, 'horizon': horizon,
              'representation': representation, 'split': split, 'max_tokens': max_tokens,
              'thinking': False, 'source_files': source_files,
              'bank_sha256': hashlib.sha256(canonical(bank).encode()).hexdigest()}))
    manifest = output / 'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())['design'] != design:
            raise ValueError('Commitment run design changed; use a new output folder')
    else:
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
        manifest.write_text(json.dumps({'design': design, 'git_revision': revision.stdout.strip(),
                                        'created_at': datetime.now(timezone.utc).isoformat()}, indent=2))
    path = output / 'responses.jsonl'
    completed = {r['trial_id'] for r in read_jsonl(path)}
    for row in selected:
        state = State.from_dict(row['state'])
        target = truth(state, horizon)
        for seed in seeds:
            if stop_at is not None and time.time() >= stop_at:
                return read_jsonl(path)
            logical = {'state': row['id'], 'seed': seed, 'design': design}
            ids = {condition: hashlib.sha256(canonical({**logical, 'condition': condition}).encode()).hexdigest()
                   for condition in CONDITIONS}
            if all(ids[c] in completed for c in CONDITIONS):
                continue
            forecast = model.ask(prompt(state, horizon, 'forecast', representation), seed,
                                 False, max_tokens, schema='json')
            forecast_value, forecast_error = parse_reply(forecast, ('unsafe', 'confidence'))
            beliefs = None
            if not forecast_error:
                try:
                    beliefs = validate_beliefs(forecast_value['unsafe'])
                    confidence = forecast_value['confidence']
                    if type(confidence) not in (int, float) or not 0 <= confidence <= 1:
                        raise ValueError('Invalid forecast confidence')
                except (ValueError, KeyError, TypeError) as exc:
                    forecast_error = f'forecast invalid: {exc}'
            for condition in CONDITIONS:
                if ids[condition] in completed:
                    continue
                if stop_at is not None and time.time() >= stop_at:
                    return read_jsonl(path)
                responses = [forecast]
                error, value, own = None, None, None
                if condition in ('bound', 'shield'):
                    if condition == 'bound' and forecast_error:
                        error = forecast_error
                    else:
                        own = beliefs if condition == 'bound' else target['unsafe']
                        value = {'action': bound_action(own), 'confidence': 1.}
                else:
                    kind = 'combined' if condition in ('combined', 'retry') else 'generic' if condition == 'generic' else 'action'
                    injected = beliefs if condition == 'own' else target['unsafe'] if condition == 'oracle' else None
                    if condition == 'own' and forecast_error:
                        error = forecast_error
                    else:
                        response = model.ask(prompt(state, horizon, kind, representation, injected),
                                             seed, False, max_tokens, schema='json')
                        responses.append(response)
                        required = ('action', 'confidence', 'unsafe') if kind in ('combined', 'generic') else ('action', 'confidence')
                        value, error = parse_reply(response, required)
                        own = (value.get('unsafe') if not error and kind in ('combined', 'generic')
                               else target['unsafe'] if condition == 'oracle' else beliefs)
                        if condition == 'retry' and not error:
                            proposed = score(value, None, target, own)
                            if proposed.get('contradiction') or proposed.get('failed_abstention'):
                                text = prompt(state, horizon, 'action', representation, own)
                                text += '\nVerify once: your previous action was ' + canonical(value['action']) + '. Follow the provided forecast.'
                                retried = model.ask(text, seed, False, max_tokens, schema='json')
                                responses.append(retried)
                                value, error = parse_reply(retried, ('action', 'confidence'))
                result = {'trial_id': ids[condition], 'protocol': design['protocol'],
                          'split': split,
                          'state_id': row['id'], 'source_seed': state.source_seed,
                          'seed': seed, 'horizon': horizon, 'condition': condition,
                          'model': model.name, 'target': target, 'value': value,
                          'score': score(value, error, target, own), 'responses': responses,
                          'upper_bound': condition in ('oracle', 'shield')}
                append_jsonl(path, result)
                completed.add(ids[condition])
                print(f'COMMIT {model.name} {condition} {row["id"]} seed={seed} correct={result["score"]["correct"]}', flush=True)
    return read_jsonl(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--bank', type=Path, default=Path('data/freeway.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--representation', default='coordinates')
    parser.add_argument('--deadline')
    args = parser.parse_args()
    bank = json.loads(args.bank.read_text())['situations']
    deadline = datetime.fromisoformat(args.deadline.replace('Z', '+00:00')).timestamp() if args.deadline else None
    run(bank, LocalModel(args.model), args.output, args.limit, horizon=args.horizon,
        representation=args.representation, stop_at=deadline)


if __name__ == '__main__':
    main()
