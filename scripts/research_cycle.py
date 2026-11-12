"""This runs the time-capped Freeway study in order, stopping cleanly before the paper deadline."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time

from diagnose.composed import ComposedPolicy
from diagnose.live import baseline, episode
from diagnose.model import LocalModel, append_jsonl, read_jsonl
from diagnose.render import FORMATS, RULES
from diagnose.runner import run

MODELS = ('qwen3:1.7b-q4_K_M', 'qwen3:4b-q4_K_M', 'qwen3:8b-q4_K_M', 'qwen3:14b-q4_K_M')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def profile_from(rows_a, rows_b, rows_c, model, seed=0):
    by_rep = {rep: [r['score']['correct'] for r in rows_a if r['task']['representation'] == rep]
              for rep in FORMATS}
    accuracy = {rep: statistics.mean(values) for rep, values in by_rep.items() if values}
    best = max(accuracy, key=accuracy.get)
    forecast = [r['score']['correct'] for r in rows_b]
    gaps = [r['score']['contradiction'] for r in rows_c if 'contradiction' in r['score']]
    prior = model.action_scores(RULES + '\nA means U, B means S, C means D.\nBOARD MASKED.', seed)
    profile = {'split': 'validation', 'model': model.name, 'digest': model.digest,
               'best_representation': best, 'representation_accuracy': accuracy,
               'A': accuracy.get('ascii', 0.), 'B': statistics.mean(forecast) if forecast else 0.,
               'C': 1-statistics.mean(gaps) if gaps else 0., 'prior': prior,
               'D': 1., 'history_not_profiled': True}
    if not prior['error']:
        scores = prior['scores']
        peak = max(scores.values())
        weights = {a: math.exp(s-peak) for a, s in scores.items()}
        probabilities = {a: w/sum(weights.values()) for a, w in weights.items()}
        profile['null_probabilities'] = probabilities
        profile['E'] = .5 if max(probabilities.values()) > .6 else 1.
    return profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deadline', required=True, help='UTC ISO timestamp; reserve time for paper')
    parser.add_argument('--output', type=Path, default=Path('results/cycle'))
    parser.add_argument('--models', nargs='+', default=MODELS)
    args = parser.parse_args()
    stop = datetime.fromisoformat(args.deadline.replace('Z', '+00:00')).timestamp()
    bank = json.loads(Path('data/freeway.json').read_text())['situations']
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    save(root / 'cycle.json', {'started_at': datetime.now(timezone.utc).isoformat(),
                             'deadline': args.deadline, 'models': args.models,
                             'status': 'running', 'scope': 'time-capped exploratory Freeway study'})

    def remaining():
        return time.time() < stop

    for model_name in args.models:
        if not remaining():
            break
        model = LocalModel(model_name, timeout=120)
        path = root / model_name.replace(':', '-')
        common = {'thinking': False, 'max_tokens': 512, 'stop_at': stop}
        a = run(bank, model, path / 'A-validation', stages=('A',), representations=FORMATS,
                seeds=(0,), split='validation', limit=10, **common)
        if not remaining():
            break
        best = max(FORMATS, key=lambda rep: statistics.mean(
            r['score']['correct'] for r in a if r['task']['representation'] == rep))
        b = run(bank, model, path / 'B-validation', stages=('B',), representations=(best,),
                horizons=(1, 5, 10), seeds=(0,), split='validation', limit=8, **common)
        c = run(bank, model, path / 'C-validation', stages=('C',), representations=(best,),
                horizons=(3,), seeds=(0,), split='validation', limit=8, **common)
        if not remaining():
            break
        profile = profile_from(a, b, c, model)
        save(path / 'profile.json', profile)
        run(bank, model, path / 'B-test', stages=('B',), representations=(best,),
            horizons=tuple(range(1, 11)), seeds=tuple(range(5)), split='test', limit=20, **common)
        model.transport('/api/generate', {'model': model.name, 'keep_alive': 0})

    for model_name in args.models:
        if not remaining():
            break
        model = LocalModel(model_name, timeout=120)
        path = root / model_name.replace(':', '-')
        profile_path = path / 'profile.json'
        if not profile_path.exists():
            continue
        profile = json.loads(profile_path.read_text())
        best = profile['best_representation']
        common = {'thinking': False, 'max_tokens': 512, 'stop_at': stop,
                  'representations': (best,), 'seeds': tuple(range(5)), 'split': 'test'}
        run(bank, model, path / 'C-test', stages=('C',), horizons=(3,), limit=10, **common)
        run(bank, model, path / 'B-repairs', stages=('B',), horizons=(1, 5, 10), limit=5,
            repairs=('base', 'rule', 'scratchpad', 'chained', 'code', 'tool'), **common)
        run(bank, model, path / 'C-repairs', stages=('C',), horizons=(3,), limit=5,
            repairs=('base', 'generic', 'forecast_then_act', 'bound', 'retry', 'oracle_shield'), **common)
        model.transport('/api/generate', {'model': model.name, 'keep_alive': 0})

    for model_name in args.models[:2]:
        if not remaining():
            break
        model = LocalModel(model_name, timeout=120)
        path = root / model_name.replace(':', '-')
        profile = json.loads((path / 'profile.json').read_text())
        for condition in ('base', 'selected', 'all', 'random', 'monitor'):
            if not remaining():
                break
            result_path = path / f'live-{condition}.jsonl'
            completed = {r['seed'] for r in read_jsonl(result_path)}
            for seed in range(50000, 50050):
                if not remaining():
                    break
                if seed in completed:
                    continue
                policy = ComposedPolicy(model, profile, condition, seed=seed, max_tokens=512)
                result = episode(policy, seed, tick_seconds=1., stop_at=stop)
                append_jsonl(result_path, {'condition': condition, **result})
                print(f'LIVE {model_name} {condition} {seed} success={result["success"]}', flush=True)
        model.transport('/api/generate', {'model': model.name, 'keep_alive': 0})

    for name in ('random', 'stay', 'advance', 'planner', 'oracle'):
        path = root / f'baseline-{name}.jsonl'
        completed = {r['seed'] for r in read_jsonl(path)}
        for seed in range(50000, 50050):
            if seed not in completed:
                append_jsonl(path, {'condition': name, **episode(baseline(name, seed), seed)})
    cycle = json.loads((root / 'cycle.json').read_text())
    cycle.update(status='complete' if remaining() else 'time_limit',
                 ended_at=datetime.now(timezone.utc).isoformat())
    save(root / 'cycle.json', cycle)


if __name__ == '__main__':
    main()
