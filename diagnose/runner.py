"""This runs frozen-board studies, remembers completed work, and records exactly what was measured."""
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import subprocess

from .analysis import summarize
from .model import append_jsonl, canonical, parse_reply, read_jsonl
from .probes import inventory, make_task, score
from .world import ACTIONS, State, repeated_outcome
from .repairs import ArithmeticProgram, bound_action


def source_fingerprint():
    files = sorted(Path(__file__).parent.glob('*.py'))
    return hashlib.sha256(b''.join(p.name.encode() + p.read_bytes() for p in files)).hexdigest()


def run(bank, model, output, stages=('A', 'B', 'C'), representations=('coordinates',),
        seeds=(0,), thinking=False, horizons=(1, 5, 10), paraphrases=(0,),
        repairs=('base',), split='test', limit=10, max_tokens=512,
        history_kinds=('none',), history_lengths=(0,), skins=('cars',), labels=('standard',)):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selected = sorted((r for r in bank if r['split'] == split), key=lambda r: r['id'])[:limit]
    if not selected:
        raise ValueError(f'No bank states in split {split}')
    design = {'source': source_fingerprint(), 'model': model.name, 'digest': model.digest,
              'version': model.version, 'model_details': model.info.get('details'),
              'stages': stages, 'representations': representations, 'seeds': seeds,
              'thinking': thinking, 'horizons': horizons, 'paraphrases': paraphrases,
              'repairs': repairs, 'split': split, 'state_ids': [r['id'] for r in selected],
              'bank_sha256': hashlib.sha256(canonical(bank).encode()).hexdigest(),
              'max_tokens': max_tokens, 'history_kinds': history_kinds,
              'history_lengths': history_lengths, 'skins': skins, 'labels': labels,
              'python': platform.python_version()}
    design = json.loads(canonical(design))
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())['design'] != design:
            raise ValueError('Output folder belongs to a different design; choose a new folder')
    else:
        git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
        manifest_path.write_text(json.dumps({'design': design, 'git_revision': git.stdout.strip(),
                                            'created_at': datetime.now(timezone.utc).isoformat()},
                                           indent=2))
    results_path = output / 'responses.jsonl'
    done = {r['trial_id'] for r in read_jsonl(results_path)}
    for row, stage, rep, seed, paraphrase, repair in itertools.product(
            selected, stages, representations, seeds, paraphrases, repairs):
        state = State.from_dict(row['state'])
        kinds = inventory()[stage]
        ks = horizons if stage in ('B', 'C', 'D', 'E') else (1,)
        diaries = list(itertools.product(history_kinds, history_lengths)) if stage == 'D' else [('none', 0)]
        frames = list(itertools.product(skins, labels)) if stage == 'E' else [('cars', 'standard')]
        for kind, k, (diary, length), (skin, label_set) in itertools.product(kinds, ks, diaries, frames):
            if stage == 'B' and repair == 'code' and kind != 'position':
                continue
            if stage == 'C' and repair in ('bound', 'retry', 'oracle_shield', 'forecast_then_act') and kind != 'combined':
                continue
            auxiliary = []
            injected = None
            forecast_value = None
            forecast_error = None
            if stage == 'C' and kind in ('action', 'own'):
                forecast_task = make_task(state, 'C', rep, k, seed, paraphrase, kind='forecast')
                forecast_response = model.ask(forecast_task['prompt'], seed, thinking, max_tokens, schema='json')
                auxiliary.append(forecast_response)
                forecast_value, forecast_error = parse_reply(forecast_response, ('collision', 'confidence'))
                if not forecast_error:
                    check = score(forecast_task, forecast_value, None, state, row['oracle'])
                    if not check['valid']:
                        forecast_error = check['error']
                if kind == 'own' and not forecast_error:
                    injected = forecast_value['collision']
            elif stage == 'C' and kind == 'oracle':
                injected = {a: repeated_outcome(state, a, k) == 'collision' for a in ACTIONS}
            task = make_task(state, stage, rep, k, seed, paraphrase, kind=kind, repair=repair,
                             history_kind=diary, history_length=length, skin=skin, labels=label_set,
                             injected=injected)
            trial = hashlib.sha256(canonical({'design': design, 'id': row['id'], 'task': task}).encode()).hexdigest()
            if trial in done:
                continue
            response = model.ask(task['prompt'], seed, thinking, max_tokens, schema='json')
            required = (('code', 'confidence') if repair == 'code' else
                        ('answer', 'confidence') if stage in ('A', 'B') else
                        ('collision', 'confidence') if stage == 'C' and kind == 'forecast' else
                        ('action', 'confidence') if stage == 'C' and kind in ('action', 'own', 'oracle') else
                        ('action', 'collision', 'confidence'))
            value, error = parse_reply(response, required)
            proposal = value
            if stage == 'C' and kind == 'own' and forecast_error:
                error = 'own forecast unavailable: ' + forecast_error
            if not error and repair == 'code':
                try:
                    c = state.cars[0]
                    answer = ArithmeticProgram(value['code'])(c.x, c.timer, c.speed, k)
                    value = {**value, 'answer': answer}
                except (ValueError, KeyError, TypeError, SyntaxError, ArithmeticError) as exc:
                    error = f'code failed: {exc}'
            if not error and repair in ('bound', 'retry', 'oracle_shield'):
                try:
                    beliefs = value['collision']
                    bound = bound_action(beliefs)
                    if repair == 'bound':
                        value = {**value, 'action': bound or 'S'}
                    elif repair == 'oracle_shield':
                        safe = row['oracle']['safe']
                        if value['action'] not in safe:
                            value = {**value, 'action': safe[0] if safe else 'S'}
                    elif beliefs[value['action']]:
                        retry_prompt = task['prompt'] + '\nYour proposed action was ' + value['action'] + '. Your own forecast was ' + canonical(beliefs) + '. Verify once and choose a consistent action.'
                        retry = model.ask(retry_prompt, seed, thinking, max_tokens, schema='json')
                        auxiliary.append(retry)
                        retried, retry_error = parse_reply(retry, ('action', 'collision', 'confidence'))
                        if not retry_error and retried.get('action') in ACTIONS and not beliefs[retried['action']]:
                            value = {**value, 'action': retried['action']}
                        else:
                            value = {**value, 'action': bound or 'S'}
                except (ValueError, KeyError, TypeError) as exc:
                    error = f'repair failed: {exc}'
            scored = score(task, value, error, state, row['oracle'])
            if stage == 'C' and kind in ('action', 'own') and not forecast_error and scored.get('valid'):
                scored['contradiction'] = forecast_value['collision'][scored['action']]
            response = {**response, 'auxiliary': auxiliary,
                        'wall_seconds': response['wall_seconds'] + sum(r['wall_seconds'] for r in auxiliary)}
            result = {'trial_id': trial, 'state_id': row['id'], 'source_seed': state.source_seed,
                      'split': split, 'model': model.name, 'digest': model.digest,
                      'thinking': thinking, 'task': task, 'response': response,
                      'parsed': value, 'proposal': proposal, 'score': scored}
            append_jsonl(results_path, result)
            done.add(trial)
            print(f'{model.name} {stage} {kind} k={k} {row["id"]} '
                  f'correct={result["score"]["correct"]}', flush=True)
    results = read_jsonl(results_path)
    (output / 'summary.json').write_text(json.dumps(summarize(results), indent=2))
    return results
