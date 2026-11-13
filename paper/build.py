"""This turns saved experimental evidence into figures and a review paper, without inventing missing runs."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
from html import escape
import json
import math
import os
import re
from pathlib import Path
import statistics
import sys

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(__file__).resolve().parents[1] / '.cache'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate,
                               Spacer, Table, TableStyle)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnose.analysis import available_foresight, bootstrap, ece, paired_bootstrap
from diagnose.world import State

REFERENCES = [
 ('Wen et al. (2025). Real-Time Reasoning Agents in Evolving Environments.', 'https://arxiv.org/abs/2511.04898'),
 ('Okcu et al. (2026). The belief-action gap in language model agents.', 'https://arxiv.org/abs/2610.17444'),
 ('Zhu et al. (2025). Where LLM Agents Fail and How They can Learn From Failures.', 'https://arxiv.org/abs/2509.25370'),
 ('Wan et al. (2026). Mitigating Conversational Inertia in Multi-Turn Agents. ICML, PMLR 306.', 'https://proceedings.mlr.press/v306/wan26a.html'),
 ('Wang et al. (2024). Can Language Models Serve as Text-Based World Simulators? ACL.', 'https://aclanthology.org/2024.acl-short.1/'),
 ('Lehrach et al. (2025). Code World Models for General Game Playing.', 'https://arxiv.org/abs/2510.04542'),
 ('Xu et al. (2026). Code over Words: Overcoming Semantic Inertia via Code-Grounded Reasoning. Findings of ACL.', 'https://aclanthology.org/2026.findings-acl.819/'),
 ('Zhao et al. (2021). Calibrate Before Use: Improving Few-shot Performance of Language Models. ICML, PMLR 139.', 'https://proceedings.mlr.press/v139/zhao21c.html'),
 ('Yang et al. (2025). Qwen3 Technical Report.', 'https://arxiv.org/abs/2505.09388'),
]


def rows(path):
    path = Path(path)
    if not path.exists():
        compressed = path.with_suffix(path.suffix + '.gz')
        if not compressed.exists():
            return []
        path = compressed
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def observations(records, metric='correct'):
    return [{'state_id': r['source_seed'], metric: r['score'][metric]}
            for r in records if metric in r['score']]


def interval(records, metric='correct'):
    return bootstrap(observations(records, metric), metric)


def linear_answer(record, bank):
    state = State.from_dict(bank[record['state_id']]['state'])
    k = record['task']['k']
    if record['task']['kind'] == 'position':
        c = state.cars[0]
        return round((c.x + k / c.speed) % 9)
    lane = state.lane
    for tick in range(1, k+1):
        lane = max(0, lane-1)
        if lane == 0:
            return False
        for c in state.cars:
            head = round((c.x + tick / c.speed) % 9)
            direction = -1 if c.speed > 0 else 1
            if c.lane == lane and any(head + i*direction == 4 for i in range(c.length)):
                return True
    return False


def wilson(successes, count):
    if count == 0:
        return {'mean': None, 'low': None, 'high': None}
    p, z = successes/count, 1.959963984540054
    denominator = 1 + z*z/count
    center = (p + z*z/(2*count))/denominator
    half = z/denominator * math.sqrt(p*(1-p)/count + z*z/(4*count*count))
    return {'mean': p, 'low': max(0., center-half), 'high': min(1., center+half)}


def shape_fits(curve):
    points = sorted(curve.items())
    if len(points) < 3:
        return None
    x, y = np.array(points, dtype=float).T
    linear = np.polyfit(x, y, 1)
    linear_prediction = np.polyval(linear, x)
    exponentials = []
    for rate in np.linspace(-1., 1., 401):
        basis = np.exp(rate*x)
        amplitude = float(basis @ y / (basis @ basis))
        prediction = amplitude*basis
        exponentials.append((float(np.mean((prediction-y)**2)), float(rate), amplitude))
    exp_error, rate, amplitude = min(exponentials)
    cliffs = []
    for cut in range(1, len(points)):
        before, after = float(y[:cut].mean()), float(y[cut:].mean())
        prediction = np.r_[np.repeat(before, cut), np.repeat(after, len(points)-cut)]
        cliffs.append((float(np.mean((prediction-y)**2)), float(x[cut-1]), before, after))
    cliff_error, threshold, before, after = min(cliffs)
    return {'horizons': len(points),
            'linear': {'slope': float(linear[0]), 'intercept': float(linear[1]),
                       'rmse': float(np.sqrt(np.mean((linear_prediction-y)**2)))},
            'exponential': {'rate': rate, 'amplitude': amplitude, 'rmse': math.sqrt(exp_error)},
            'cliff': {'last_before': threshold, 'before': before, 'after': after,
                      'rmse': math.sqrt(cliff_error)}}


def heldout_shape_fits(records):
    from collections import defaultdict
    by_episode = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r['task']['kind'] == 'position':
            by_episode[r['source_seed']][r['task']['k']].append(int(r['score']['correct']))
    output = {'protocol': 'posthoc-leave-source-episode-out-v1', 'folds': [], 'summary': {}, 'contrasts_vs_flat': {}}
    if len(by_episode) < 3:
        return output
    errors = defaultdict(list)
    for heldout, test in sorted(by_episode.items()):
        train = defaultdict(list)
        for episode, curve in by_episode.items():
            if episode != heldout:
                for k, values in curve.items():
                    train[k].extend(values)
        curve = {k: statistics.mean(v) for k, v in train.items()}
        fits = shape_fits(curve)
        if fits is None or set(test) != set(curve):
            continue
        predictions = {name: {} for name in ('flat', 'linear', 'exponential', 'cliff')}
        for k in test:
            predictions['flat'][k] = statistics.mean(curve.values())
            predictions['linear'][k] = fits['linear']['slope']*k+fits['linear']['intercept']
            predictions['exponential'][k] = fits['exponential']['amplitude']*math.exp(fits['exponential']['rate']*k)
            predictions['cliff'][k] = fits['cliff']['before' if k <= fits['cliff']['last_before'] else 'after']
        fold = {'source_episode': heldout, 'training_episodes': sorted(set(by_episode)-{heldout}),
                'test_horizons': sorted(test), 'fits': fits, 'scores': {}}
        for name, prediction in predictions.items():
            prediction = {k: min(1., max(0., p)) for k, p in prediction.items()}
            mse = statistics.mean((prediction[k]-statistics.mean(test[k]))**2 for k in test)
            brier = statistics.mean(statistics.mean((prediction[k]-v)**2 for v in test[k]) for k in test)
            fold['scores'][name] = {'curve_mse': mse, 'brier': brier, 'predictions': prediction}
            errors[name].append({'state_id': heldout, 'curve_mse': mse, 'brier': brier})
        output['folds'].append(fold)
    for name, values in errors.items():
        output['summary'][name] = {'curve_mse': bootstrap(values, 'curve_mse'),
                                   'brier': bootstrap(values, 'brier')}
    for name in ('linear', 'exponential', 'cliff'):
        if name in errors:
            differences = [{'state_id': f['source_episode'],
                            'curve_mse': f['scores'][name]['curve_mse']-f['scores']['flat']['curve_mse'],
                            'brier': f['scores'][name]['brier']-f['scores']['flat']['brier']}
                           for f in output['folds']]
            output['contrasts_vs_flat'][name] = {
                metric: bootstrap(differences, metric) for metric in ('curve_mse', 'brier')}
    return output


def motion_events(state, horizon=10):
    car = state.cars[0]
    events = []
    for _ in range(horizon):
        following = car.tick()
        event = ('wrap' if car.x < 0 or car.x > 8 else 'fast_step' if abs(car.speed) < 1
                 else 'timer_move' if following.x != car.x else 'timer_wait')
        events.append(event)
        car = following
    return events


def first_error_events(records, bank):
    sequences = {}
    for record in records:
        if record['task']['kind'] == 'position':
            key = record['state_id'], record['task']['seed']
            sequences.setdefault(key, {})[record['task']['k']] = record['score']['correct']
    result = {'complete_sequences': 0, 'incomplete_sequences': 0, 'all_correct': 0,
              'first_error_event': dict.fromkeys(('wrap', 'fast_step', 'timer_move', 'timer_wait'), 0),
              'first_error_after_any_wrap': 0}
    for (state_id, seed), sequence in sequences.items():
        if set(sequence) != set(range(1, 11)):
            result['incomplete_sequences'] += 1
            continue
        result['complete_sequences'] += 1
        first = next((k for k in range(1, 11) if not sequence[k]), None)
        if first is None:
            result['all_correct'] += 1
            continue
        events = motion_events(State.from_dict(bank[state_id]['state']))
        result['first_error_event'][events[first-1]] += 1
        result['first_error_after_any_wrap'] += int('wrap' in events[:first])
    return result


def analyze(root, bank_path):
    bank_data = json.loads(Path(bank_path).read_text())
    bank = {r['id']: r for r in bank_data['situations']}
    output = {'bank': bank_data['manifest'], 'models': {}, 'live': [], 'live_paired': [], 'files': {},
              'generated_at': datetime.now(timezone.utc).isoformat(),
              'bank_sha256': hashlib.sha256(Path(bank_path).read_bytes()).hexdigest()}
    for path in sorted((p for p in root.rglob('*') if p.name.endswith(('.jsonl', '.jsonl.gz')))):
        output['files'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for folder in sorted(root.iterdir()):
        profile_path = folder / 'profile.json'
        if not folder.is_dir() or not profile_path.exists():
            continue
        profile = json.loads(profile_path.read_text())
        test = rows(folder / 'B-test/responses.jsonl')
        curve = []
        for kind in ('position', 'collision'):
            for k in range(1, 11):
                group = [r for r in test if r['task']['kind'] == kind and r['task']['k'] == k]
                if not group:
                    continue
                linear = [{**r, 'score': {**r['score'], 'correct': linear_answer(r, bank) == r['task']['truth']}}
                          for r in group]
                valid = [{**r['score']} for r in group]
                curve.append({'kind': kind, 'k': k, 'accuracy': interval(group),
                              'linear': interval(linear), 'ece_valid_only': ece(valid),
                              'always_collision': statistics.mean(bool(r['task']['truth']) for r in group) if kind == 'collision' else None,
                              'mean_confidence_valid': statistics.mean(
                                  r['score']['confidence'] for r in group if r['score'].get('valid'))
                                  if any(r['score'].get('valid') for r in group) else None,
                              'invalid': sum(not r['score']['valid'] for r in group),
                              'states': len({r['state_id'] for r in group})})
        pos_curve = {r['k']: r['accuracy']['mean'] for r in curve if r['kind'] == 'position'}
        data = {'profile': profile, 'trials': len(test),
                'states': len({r['state_id'] for r in test}),
                'source_episodes': len({r['source_seed'] for r in test}),
                'seeds': sorted({r['task']['seed'] for r in test}), 'curve': curve,
                'f_prefix_80': available_foresight(pos_curve),
                'f_largest_80': max((k for k, a in pos_curve.items() if a >= .8), default=0),
                'position_accuracy': interval([r for r in test if r['task']['kind'] == 'position']),
                'collision_accuracy': interval([r for r in test if r['task']['kind'] == 'collision']),
                'invalid': sum(not r['score']['valid'] for r in test)}
        data['position_shape_fits'] = shape_fits(pos_curve)
        data['position_shape_prediction'] = heldout_shape_fits(test)
        data['first_error_events'] = first_error_events(test, bank)
        data['position_signed_error_curve'] = [
            {'k': k, 'error': interval([r for r in test if r['task']['kind'] == 'position'
                                      and r['task']['k'] == k], 'signed_error')}
            for k in range(1, 11)]
        data['position_signed_error_valid'] = interval(
            [r for r in test if r['task']['kind'] == 'position'], 'signed_error')
        data['position_absolute_error_valid'] = interval(
            [{**r, 'score': {**r['score'], 'absolute_error': abs(r['score']['signed_error'])}}
             for r in test if r['task']['kind'] == 'position' and 'signed_error' in r['score']],
            'absolute_error')
        repairs = rows(folder / 'B-repairs/responses.jsonl')
        existing = {(r['state_id'], r['task']['kind'], r['task']['repair'], r['task']['k'], r['task']['seed']) for r in repairs}
        repairs += [r for r in rows(folder / 'B-repairs-primary/responses.jsonl')
                    if (r['state_id'], r['task']['kind'], r['task']['repair'], r['task']['k'], r['task']['seed']) not in existing]
        base = [r for r in repairs if r['task']['repair'] == 'base' and r['task']['kind'] == 'position']
        data['mean_confidence_valid'] = interval(test, 'confidence')
        data['position_ece_valid'] = ece([r['score'] for r in test if r['task']['kind'] == 'position'])
        data['collision_ece_valid'] = ece([r['score'] for r in test if r['task']['kind'] == 'collision'])
        data['always_collision_accuracy'] = bootstrap([{'state_id': r['source_seed'], 'correct': bool(r['task']['truth'])} for r in test if r['task']['kind'] == 'collision'])['mean']
        test_states = {r['state_id'] for r in test}
        required = [bank[k]['oracle']['required_foresight'] for k in test_states]
        feasible = [h for h in required if h is not None]
        data['required_foresight'] = {'counts': dict(Counter(str(h) for h in required)),
                                      'median_feasible': statistics.median(feasible) if feasible else None,
                                      'prefix_coverage_feasible': sum(h <= data['f_prefix_80'] for h in feasible)/len(feasible) if feasible else None}
        data['repairs'] = []
        for name in ('base', 'rule', 'scratchpad', 'chained', 'code', 'tool'):
            condition = [r for r in repairs if r['task']['repair'] == name and r['task']['kind'] == 'position']
            if condition:
                data['repairs'].append({'name': name, 'accuracy': interval(condition),
                                        'difference_from_base': paired_bootstrap(observations(base), observations(condition)),
                                        'invalid': sum(not r['score']['valid'] for r in condition),
                                        'trials': len(condition), 'states': len({r['state_id'] for r in condition})})
        consistency = rows(folder / 'C-test/responses.jsonl')
        data['consistency'] = []
        for name in ('forecast', 'action', 'own', 'oracle', 'combined'):
            group = [r for r in consistency if r['task']['kind'] == name]
            if group:
                data['consistency'].append({'kind': name, 'accuracy': interval(group),
                                            'contradiction': interval(group, 'contradiction'),
                                            'trials': len(group)})
        commitments = rows(folder / 'C-commitment/responses.jsonl')
        data['commitment'] = []
        for name in ('none', 'own', 'oracle', 'combined', 'generic', 'bound', 'retry', 'shield'):
            group = [r for r in commitments if r['condition'] == name]
            if group:
                eligible = [r for r in group if r['score'].get('own_feasible')]
                data['commitment'].append({'condition': name, 'accuracy': interval(group),
                                           'forecast_accuracy_valid_only': interval(group, 'forecast_accuracy'),
                                           'inconsistency_valid_only': interval(group, 'inconsistent'),
                                           'skipped_progress_valid_only': interval(group, 'skipped_progress'),
                                           'failed_abstention_valid_only': interval(group, 'failed_abstention'),
                                           'unsafe_choice_own_feasible': interval(eligible, 'contradiction'),
                                           'trials': len(group), 'invalid': sum(not r['score']['valid'] for r in group),
                                           'states': len({r['state_id'] for r in group}),
                                           'upper_bound': name in ('oracle', 'shield')})
        output['models'][profile['model']] = data
        for path in sorted(folder.glob('live-*.jsonl*'), key=lambda p: (
                {'base': 0, 'selected': 1, 'all': 2}.get(p.name.split('.')[0][5:], 3), p.name)):
            group = rows(path)
            complete = [r for r in group if not r.get('censored')]
            values = [{'state_id': r['seed'], 'success': r['success']} for r in complete]
            output['live'].append({'model': profile['model'], 'condition': path.name.removesuffix('.gz').removesuffix('.jsonl')[5:],
                                  'episodes': len(complete), 'censored': len(group)-len(complete),
                                  'episode_seeds': sorted(r['seed'] for r in complete),
                                  'successes': sum(r['success'] for r in complete),
                                  'success': bootstrap(values, 'success'),
                                  'success_wilson': wilson(sum(r['success'] for r in complete), len(complete)),
                                  'mean_tokens': statistics.mean(r['tokens'] for r in complete) if complete else None,
                                  'mean_calls': statistics.mean(r['calls'] for r in complete) if complete else None,
                                  'mean_collisions': statistics.mean(r['collisions'] for r in complete) if complete else None,
                                  'error_decisions': sum(bool(t['response'].get('error')) for r in complete for t in r['trace']),
                                  'decisions': sum(len(r['trace']) for r in complete)})
        base_live = {r['seed']: r for r in rows(folder / 'live-base.jsonl') if not r.get('censored')}
        for condition in ('selected', 'all', 'random', 'monitor'):
            candidate = {r['seed']: r for r in rows(folder / f'live-{condition}.jsonl') if not r.get('censored')}
            shared = sorted(base_live.keys() & candidate.keys())
            if shared:
                differences = [{'state_id': seed,
                                'success_change': int(candidate[seed]['success'])-int(base_live[seed]['success']),
                                'token_change': candidate[seed]['tokens']-base_live[seed]['tokens']}
                               for seed in shared]
                output['live_paired'].append({'model': profile['model'], 'condition': condition,
                                             'paired_episodes': len(shared),
                                             'success_change': bootstrap(differences, 'success_change'),
                                             'token_change': bootstrap(differences, 'token_change')})
    def model_size(item):
        match = re.search(r':([0-9.]+)b', item[0])
        return float(match.group(1)) if match else float('inf')
    output['models'] = dict(sorted(output['models'].items(), key=model_size))
    for path in sorted(root.glob('baseline-*.jsonl*')):
        group = rows(path)
        output['live'].append({'model': 'simulator control', 'condition': path.name.removesuffix('.gz').removesuffix('.jsonl')[9:],
                              'episodes': len(group), 'censored': 0,
                              'episode_seeds': sorted(r['seed'] for r in group),
                              'successes': sum(r['success'] for r in group),
                              'success': bootstrap([{'state_id': r['seed'], 'success': r['success']} for r in group], 'success'),
                              'success_wilson': wilson(sum(r['success'] for r in group),len(group)),
                              'mean_tokens': 0, 'mean_calls': 0,
                              'error_decisions': 0, 'decisions': sum(len(r['trace']) for r in group),
                              'mean_collisions': statistics.mean(r['collisions'] for r in group) if group else None})
    return output


def crossed_interval(values, seed=0, replicates=2000):
    import random
    programs = sorted(values)
    shared = sorted(set.intersection(*(set(values[p]) for p in programs))) if programs == list(range(5)) else []
    result = {'mean': None, 'low': None, 'high': None, 'programs': len(programs),
              'episodes': len(shared), 'observations': len(programs)*len(shared)}
    if not shared:
        return result
    grid = [[float(values[p][e]) for e in shared] for p in programs]
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        ps = rng.choices(range(5), k=5)
        es = rng.choices(range(len(shared)), k=len(shared))
        estimates.append(sum(grid[p][e] for p in ps for e in es)/(5*len(shared)))
    estimates.sort()
    result.update(mean=sum(map(sum, grid))/(5*len(shared)),
                  low=estimates[int(.025*(replicates-1))], high=estimates[int(.975*(replicates-1))])
    return result


def analyze_compiled(root):
    from collections import defaultdict
    from diagnose.compiled import prepare
    groups, source_programs = defaultdict(list), {}
    for path in sorted(Path(root).rglob('episodes.jsonl*')):
        for record in rows(path):
            if record.get('protocol') not in ('compiled-controller-v1', 'compiled-controller-v2'):
                continue
            protocol = record['protocol']
            if any(t['response']['fresh_requests'] != 0 for t in record['trace']):
                raise ValueError('Compiled replay contains fresh requests')
            key = (protocol, record['model'], record['condition'], record['validation'])
            groups[key].append(record)
            program_key = (protocol, record['model'], record['program_seed'])
            if program_key in source_programs and source_programs[program_key] != record['generation_response']:
                raise ValueError('Compiled source program changed')
            source_programs[program_key] = record['generation_response']
    output = {'groups': [], 'programs': [], 'contrasts': [], 'complete': 0, 'censored': 0,
              'source_programs': len({r['key'] for r in source_programs.values()})}
    matrices = {}
    for (protocol, model, condition, validation), records in sorted(groups.items()):
        complete = [r for r in records if not r['censored']]
        output['complete'] += len(complete)
        output['censored'] += len(records)-len(complete)
        matrix = defaultdict(dict)
        for r in complete:
            if r['seed'] in matrix[r['program_seed']]:
                raise ValueError('Duplicate completed compiled case')
            matrix[r['program_seed']][r['seed']] = int(r['success'])
        matrices[(protocol, model, condition, validation)] = matrix
        output['groups'].append({'protocol': protocol, 'model': model, 'condition': condition, 'validation': validation,
             'episodes': len(complete), 'censored': len(records)-len(complete),
             'successes': sum(r['success'] for r in complete), 'success': crossed_interval(matrix),
             'mean_tokens': statistics.mean(r['tokens'] for r in complete) if complete else None,
             'mean_calls': statistics.mean(r['calls'] for r in complete) if complete else None,
             'mean_collisions': statistics.mean(r['collisions'] for r in complete) if complete else None,
             'mean_stay_loop': statistics.mean(r['longest_stay_loop'] for r in complete) if complete else None,
             'fresh_requests': sum(t['response']['fresh_requests'] for r in records for t in r['trace'])})
        for program_seed in range(5):
            subset = [r for r in complete if r['program_seed'] == program_seed]
            if not subset:
                continue
            _, parse_error, training_error = prepare(source_programs[(protocol, model, program_seed)], 'executable',
                 native_inputs=protocol == 'compiled-controller-v2')
            output['programs'].append({'protocol': protocol, 'model': model, 'program_seed': program_seed,
                  'condition': condition, 'validation': validation, 'episodes': len(subset),
                  'successes': sum(r['success'] for r in subset),
                  'success_wilson': wilson(sum(r['success'] for r in subset), len(subset)),
                  'executable': parse_error is None, 'training_passed': parse_error is None and training_error is None,
                  'generation_key': source_programs[(protocol, model, program_seed)]['key']})
    for protocol, model in sorted({key[:2] for key in groups}):
        contrasts = [('once minus per-turn / '+v, ('amortized', v), ('per_turn', v))
                     for v in ('executable', 'checked')]
        contrasts += [('checked minus executable / '+c, (c, 'checked'), (c, 'executable'))
                      for c in ('amortized', 'per_turn')]
        for name, candidate_key, base_key in contrasts:
            candidate = matrices.get((protocol, model, *candidate_key), {})
            base = matrices.get((protocol, model, *base_key), {})
            paired = {p: {e: candidate[p][e]-base[p][e] for e in candidate[p].keys() & base[p].keys()}
                      for p in candidate.keys() & base.keys()}
            output['contrasts'].append({'protocol': protocol, 'model': model, 'contrast': name,
                                       'success_change': crossed_interval(paired)})
    return output


def analyze_followup(root):
    from collections import defaultdict
    from diagnose.context import summarize as context_summary
    from diagnose.model import canonical
    root = Path(root)
    output = {'input_sha256': {}, 'forecasts': [], 'context': [], 'live': [],
              'live_paired': [], 'profiles': {}, 'context_trials': 0}
    forecasts, context = defaultdict(list), []
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.name.endswith(('.json', '.jsonl', '.jsonl.gz')):
            output['input_sha256'][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        if not path.name.endswith(('responses.jsonl', 'responses.jsonl.gz')):
            continue
        for record in rows(path):
            task = record.get('task', {})
            if task.get('stage') == 'B':
                key = (record['model'], str(path.parent.relative_to(root)),
                       task['kind'], task['repair'])
                forecasts[key].append(record)
            elif task.get('stage') in ('D', 'E') and record.get('split') == 'test':
                context.append(record)
    for (model, cohort, kind, repair), group in sorted(forecasts.items()):
        base = forecasts.get((model, cohort, kind, 'base'), [])
        def identity(r):
            return tuple([r['state_id']] + [r['task'].get(k, 0) for k in ('seed', 'k', 'paraphrase')])
        baseline = {identity(r): r for r in base}
        pairs = [{'state_id': r['source_seed'], 'difference': int(r['score']['correct'])-
                  int(baseline[identity(r)]['score']['correct'])} for r in group if identity(r) in baseline]
        output['forecasts'].append({'model': model, 'cohort': cohort, 'kind': kind, 'repair': repair,
              'trials': len(group), 'states': len({r['state_id'] for r in group}),
              'invalid': sum(not r['score']['valid'] for r in group), 'accuracy': interval(group),
              'paired_trials': len(pairs), 'difference_from_base': bootstrap(pairs, 'difference'),
              'unique_requests': len({r.get('response', {}).get('key', canonical(identity(r))) for r in group})})
    output['context_trials'] = len(context)
    output['context'] = context_summary(context)
    for folder in sorted(p.parent for p in root.glob('*/profile.json')):
        profile = json.loads((folder/'profile.json').read_text())
        output['profiles'][profile['model']] = profile
        completed = {}
        for path in sorted(folder.glob('live-*.jsonl*')):
            group = rows(path)
            if any(r.get('policy_revision') != 'v2' for r in group):
                raise ValueError('Follow-up live records must declare v2; never pool with v1')
            complete = [r for r in group if not r['censored']]
            seeds = [r['seed'] for r in complete]
            if len(set(seeds)) != len(seeds):
                raise ValueError('Duplicate completed live seed')
            condition = path.name.removesuffix('.gz').removesuffix('.jsonl')[5:]
            completed[condition] = {r['seed']: r for r in complete}
            output['live'].append({'model': profile['model'], 'condition': condition,
                'episodes': len(complete), 'censored': len(group)-len(complete),
                'episode_seeds': sorted(seeds), 'successes': sum(r['success'] for r in complete),
                'success_wilson': wilson(sum(r['success'] for r in complete), len(complete)),
                'mean_tokens': statistics.mean(r['tokens'] for r in complete) if complete else None,
                'mean_calls': statistics.mean(r['calls'] for r in complete) if complete else None,
                'error_decisions': sum(bool(t['response'].get('error')) for r in complete for t in r['trace']),
                'decisions': sum(len(r['trace']) for r in complete),
                'validated_program_decisions': sum(bool(t['response'].get('program_validated'))
                                                    for r in complete for t in r['trace'])})
        base = completed.get('base', {})
        for condition, candidate in completed.items():
            if condition == 'base':
                continue
            shared = sorted(base.keys() & candidate.keys())
            paired = [{'state_id': seed, 'difference': int(candidate[seed]['success'])-int(base[seed]['success'])}
                      for seed in shared]
            output['live_paired'].append({'model': profile['model'], 'condition': condition,
                    'paired_episodes': len(shared), 'success_change': bootstrap(paired, 'difference')})
    for path in sorted(root.glob('baseline-*.jsonl*')):
        group = rows(path)
        output['live'].append({'model': 'simulator control',
              'condition': path.name.removesuffix('.gz').removesuffix('.jsonl')[9:],
              'episodes': len(group), 'censored': 0, 'successes': sum(r['success'] for r in group),
              'success_wilson': wilson(sum(r['success'] for r in group), len(group)),
              'mean_tokens': 0, 'mean_calls': 0, 'error_decisions': 0, 'decisions': 0,
              'validated_program_decisions': 0})
    output['compiled'] = analyze_compiled(root)
    return output


def curves_figure(evidence, output):
    models = list(evidence['models'])
    if not models:
        return None
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'font.family': 'DejaVu Sans'})
    figure, axes = plt.subplots(2, len(models), figsize=(7.33, 4.1), squeeze=False,
                                sharex=True, sharey=True, layout='constrained')
    for col, name in enumerate(models):
        model = evidence['models'][name]
        for row, kind in enumerate(('position', 'collision')):
            curve = [r for r in model['curve'] if r['kind'] == kind]
            ax = axes[row][col]
            x = [r['k'] for r in curve]
            ax.plot(x, [r['accuracy']['mean'] for r in curve], 'o-', color='#215c8c', label='Model')
            ax.fill_between(x, [r['accuracy']['low'] for r in curve], [r['accuracy']['high'] for r in curve],
                            alpha=.16, color='#215c8c')
            ax.plot(x, [r['linear']['mean'] for r in curve], '--', color='#bb7134', label='Linear')
            ax.axhline(1/17 if kind == 'position' else .5, color='#7e8590', linewidth=.8, linestyle='-.', label='Random')
            ax.axhline(1., color='#63866d', linewidth=.8, linestyle=':', label='Oracle')
            ax.set_ylim(-.02, 1.05)
            ax.set_xticks([1, 3, 5, 7, 10])
            ax.grid(alpha=.15)
            if row == 0:
                ax.set_title(name.replace('qwen3:', '').replace('-q4_K_M', '') + f' | {model["states"]} boards')
            if col == 0:
                ax.set_ylabel(('Exact head position' if row == 0 else 'Collision prediction') + '\nAccuracy')
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='outside upper center', ncol=4, fontsize=8)
    figure.supxlabel('Forecast horizon (car ticks)', fontsize=9)
    figure.savefig(output, dpi=180, bbox_inches='tight')
    plt.close(figure)
    return output


def signed_error_figure(evidence, output):
    models = [(name, m) for name, m in evidence['models'].items()
              if any(r['error']['mean'] is not None for r in m['position_signed_error_curve'])]
    if not models:
        return None
    figure, axes = plt.subplots(1, len(models), figsize=(7.33, 2.25), squeeze=False,
                                layout='constrained', sharey=True)
    for ax, (name, model) in zip(axes[0], models):
        data = model['position_signed_error_curve']
        x = [r['k'] for r in data]
        ax.plot(x, [r['error']['mean'] if r['error']['mean'] is not None else np.nan for r in data],
                color='#24577a')
        ax.fill_between(x, [r['error']['low'] if r['error']['low'] is not None else np.nan for r in data],
                        [r['error']['high'] if r['error']['high'] is not None else np.nan for r in data], alpha=.2, color='#24577a')
        ax.axhline(0, color='#777777', linestyle=':', linewidth=1)
        ax.set_title(name.replace('qwen3:', '').replace('-q4_K_M', ''), fontsize=9)
        ax.set_xticks([1, 3, 5, 7, 10])
        ax.grid(alpha=.15)
    axes[0][0].set_ylabel('Predicted x minus true x (cells)', fontsize=8)
    figure.supxlabel('Forecast horizon (car ticks)', fontsize=9)
    figure.savefig(output, dpi=180, bbox_inches='tight')
    plt.close(figure)
    return output


def percent(value):
    return 'unavailable' if value is None else f'{100*value:.1f}%'


def ci(value, difference=False):
    if value['mean'] is None:
        return 'unavailable'
    mean = f'{100*value["mean"]:.1f}' if difference else percent(value['mean'])
    return f'{mean} [{100*value["low"]:.1f}, {100*value["high"]:.1f}]'


def build(evidence, output, figures, root):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('PaperTitle', fontName='Helvetica-Bold', fontSize=21, leading=25,
                              spaceAfter=12, textColor=colors.HexColor('#193b55')))
    styles.add(ParagraphStyle('PaperBody', fontName='Helvetica', fontSize=10, leading=14,
                              spaceAfter=8, allowWidows=0, allowOrphans=0))
    styles.add(ParagraphStyle('SmallNote', fontName='Helvetica', fontSize=8.2, leading=11,
                              spaceAfter=7, textColor=colors.HexColor('#465361')))
    styles.add(ParagraphStyle('TableText', fontName='Helvetica', fontSize=8, leading=10))
    styles.add(ParagraphStyle('SubtitleCenter', fontSize=10, leading=14, alignment=TA_CENTER,
                              spaceAfter=14, textColor=colors.HexColor('#465361')))
    styles['Heading2'].keepWithNext = True
    styles['Heading2'].spaceBefore = 10
    story = []
    markdown = []

    def paragraph(text, style='PaperBody'):
        block = Paragraph(text, styles[style])
        story.append(KeepTogether([block]) if style == 'SmallNote' else block)
        markdown.append(text.replace('<b>', '**').replace('</b>', '**'))

    def heading(text):
        story.append(Paragraph(text, styles['Heading2']))
        markdown.append('\n## ' + text)

    def table(headers, body, widths=None):
        data = [[Paragraph(escape(str(v)), styles['TableText']) for v in headers]]
        data.extend([[Paragraph(escape(str(v)), styles['TableText']) for v in row] for row in body])
        t = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8eef3')),
                               ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                               ('LINEBELOW', (0, 0), (-1, 0), .6, colors.HexColor('#9aafbd')),
                               ('LINEBELOW', (0, 1), (-1, -1), .25, colors.HexColor('#dce3e7')),
                               ('LEFTPADDING', (0, 0), (-1, -1), 6),
                               ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                               ('TOPPADDING', (0, 0), (-1, -1), 6),
                               ('BOTTOMPADDING', (0, 0), (-1, -1), 6)]))
        story.extend((KeepTogether([t]) if len(body) <= 8 else t, Spacer(1, 10)))
        markdown.extend([' | '.join(map(str, headers)), ' | '.join('---' for _ in headers),
                         *[' | '.join(map(str, r)) for r in body]])

    models = evidence['models']
    total = sum(m['trials'] for m in models.values())
    complete = all(m['trials'] == 2000 for m in models.values()) and len(models) == 4
    position_values = [m['position_accuracy']['mean'] for m in models.values()
                       if m['position_accuracy']['mean'] is not None]
    finding = (f'Episode-weighted exact-head accuracy spans {percent(min(position_values))}-'
               f'{percent(max(position_values))} across the collected artifacts. '
               if position_values else '')
    agent_live = [r for r in evidence['live'] if r['model'] != 'simulator control']
    live_seed_count = len({seed for r in agent_live for seed in r['episode_seeds']})
    live_finding = (f'Live collection produced {sum(r["episodes"] for r in agent_live)} completed '
                    f'model-condition runs over {live_seed_count} environment seeds '
                    f'({sum(r["successes"] for r in agent_live)} goals) and '
                    f'{sum(r["censored"] for r in agent_live)} censored runs. ' if agent_live else '')
    title = 'Diagnose, Then Repair'
    paragraph(title, 'PaperTitle')
    paragraph('A simulator-checked study of small language-model agents in real-time Freeway', 'Heading2')
    paragraph('Research paper for review | November 12-13, 2026 | Two capped local cycles'
              if evidence.get('followup') else 'Research paper for review | November 12, 2026 | Time-capped local study', 'SmallNote')
    heading('Abstract')
    paragraph(f'A correct action in a changing world requires a readable observation, an adequate forecast, '
              f'and a decision that follows that forecast. We implement a stage-oriented diagnostic suite '
              f'with replayable states, six renderers, simulator-checked labels, and targeted inference-time '
              f'repairs. This report analyzes {total:,} held-out forecast trials from {len(models)} installed '
              f'Qwen3 model artifacts collected on a 16 GB Apple M3 laptop. The situation bank contains '
              f'500 states with disjoint source episodes across splits. We compare forecast accuracy at '
              f'horizons 1-10 against constant-velocity extrapolation, and separately evaluate consistency '
              f'repairs and measured-delay live play where completed. {finding}{live_finding}Missing or malformed answers count '
              f'as incorrect. This is an exploratory, single-game artifact; the full multi-game, '
              f'thinking-mode, and training-based study remains incomplete. We make no first-of-its-kind '
              f'or causal root-cause claim from this cycle.')
    if evidence.get('followup'):
        paragraph('The first-cycle results below keep their original cohort. Section 6.1 reports '
                  'the separate November 13 follow-up; its changed code and fresh live seeds are '
                  'never pooled with the first cycle.', 'SmallNote')
        followup_agents = [r for r in evidence['followup']['live'] if r['model'] != 'simulator control']
        if followup_agents:
            paragraph(f'The separate v2 follow-up adds {sum(r["episodes"] for r in followup_agents)} '
                      f'completed model-condition runs and {sum(r["successes"] for r in followup_agents)} '
                      'goals. Its actual cohorts and limits are reported separately below.', 'SmallNote')
    heading('1. Motivation and related work')
    paragraph('RealtimeGym makes reasoning time consequential: the environment can change before a '
              'model finishes deciding [1]. We build on that simulator, rather than proposing a new game '
              'benchmark. Our question is whether the stage that appears weak on frozen states points '
              'to a useful repair during live play.')
    paragraph('The decomposition itself is established. Belief-action work distinguishes represented '
              'information from its use in decisions [2]; AgentDebug studies modular and cascading '
              'failures [3]. Conversational inertia motivates controlled history interventions [4]. '
              'Text-world simulation studies motivate explicit transition probes [5], while executable '
              'world models motivate computational offloading [6]. Semantic inertia motivates label '
              'and framing controls [7]. Those constructs guide the implementation; this report does '
              'not rebrand them as new contributions.')
    heading('2. Simulator and reference labels')
    paragraph('We pin RealtimeGym to commit 3d5b3ef37dafee25ca2e8d2e8d71f5fe842eda7b. '
              'Freeway has nine columns and ten lanes; the player stays in column 4. Its actual motion '
              'uses per-object timers, car lengths, and a separate wrap tick. A modulo update alone '
              'is therefore not simulator ground truth. Our immutable replay engine is checked '
              'against original-game trajectories, including collisions and board resets.')
    paragraph('The reference safe set contains first actions admitting a collision-free continuation '
              'through ten ticks. An empty set is retained. The optimal set first prefers survival, '
              'then reaching the goal, then forward progress; ties remain valid alternatives. Required '
              'foresight h* is the first search depth whose chosen action belongs to the ten-tick '
              'safe set. This is finite-horizon safety, not an indefinite guarantee. The collision '
              'probe asks about repeating one action, a distinct quantity from existence of a safe '
              'continuation. We never interchange those labels.')
    bank = evidence['bank']
    paragraph(f'The bank has {bank["actual"]} unique situations: 125 in each of safe advance, dangerous '
              'advance, waiting helps, and retreat required. Waiting helps requires that waiting is '
              'reference-safe and avoids a collision caused by immediate advance; retreat required '
              'means D is the sole safe action. Splits contain 300/100/100 states. Each episode '
              'contributes at most two states; source episode counts are '
              f'{bank["source_episodes"]["train"]}/{bank["source_episodes"]["validation"]}/'
              f'{bank["source_episodes"]["test"]}. These categories are sampling strata, not '
              'estimates of how often each situation occurs in ordinary play.')
    heading('3. Experimental protocol')
    paragraph('We registered hypotheses before model collection. A subsequent execution plan records '
              'the user-imposed eight-hour cap and its reduced scope. Renderer selection uses ten '
              'validation states, six representations, five perception probes, and sampling seed 0. '
              'Forecast and consistency profiles use eight validation states each. The main forecast '
              'test targets twenty held-out states, two question types, ten horizons, and five '
              'sampling seeds per model. The forecasted head is object 0; object coverage is a '
              'limitation. No test result chooses the renderer or the repair profile.')
    paragraph('The installed artifacts belong to the Qwen3 family [9]. Sizes in tables are nominal '
              'Ollama tags; exact digests and reported model metadata are retained. Thinking is off, '
              'temperature is 0.7, context limit is 8,192 tokens, and the answer '
              'budget is 512 tokens. Exact prompts, model digests, quantization metadata, raw replies, '
              'timing and software fingerprints are logged. Requests are cached by all decoding '
              'settings and model identity. Malformed outputs, truncations, and timeouts remain in '
              'the primary denominator. Calibration error is conditional on valid confidence reports. '
              'Accuracy and mean confidence weight source episodes equally; ECE pools valid trials. '
              'Intervals are percentile 95% bootstrap intervals over source episodes, with 2,000 '
              'resamples; sampling repeats do not become independent boards. Boundary estimates '
              'can yield degenerate resampling intervals, which do not establish certainty on new states.')
    representation_rows = [[name.replace('qwen3:', '').replace('-q4_K_M', ''),
                            *[percent(m['profile'].get('representation_accuracy', {}).get(rep))
                              for rep in ('ascii', 'coordinates', 'prose', 'lanes', 'relations', 'json')]]
                           for name, m in models.items()]
    if representation_rows:
        paragraph('Validation representation screen (single sampling seed, ten states, five probes). '
                  'These selection scores are not held-out evidence that one format generalizes.', 'SmallNote')
        table(['Model tag', 'ASCII', 'Coordinates', 'Prose', 'Lanes', 'Relations', 'JSON'],
              representation_rows, [60, 70, 100, 70, 70, 88, 70])
    heading('4. Foresight measurements')
    paragraph('The position task asks for an exact head column after k car ticks. The collision task '
              'asks whether repeatedly advancing causes any collision within k ticks. The linear '
              'control predicts round((x + k/p) mod 9), where p is the simulator speed parameter, '
              'then applies the same endpoint collision geometry. It intentionally ignores timer '
              'phase and delayed wrap. It is a declared cheap baseline, not an exact physics model. '
              'Random controls are uniform over the 17 possible head columns (-4 through 12) and '
              'over two collision labels. An always-collision control exposes label imbalance.')
    body = []
    for name, m in models.items():
        body.append([name.replace('qwen3:', '').replace('-q4_K_M', ''),
                     f'{m["states"]}/{m["source_episodes"]}', str(m['trials']),
                     ci(m['position_accuracy']), ci(m['collision_accuracy']) + ' / ' + percent(m['always_collision_accuracy']), str(m['invalid'])])
    if body:
        table(['Model tag', 'Boards / episodes', 'Trials', 'Position accuracy [95% CI]',
               'Collision [95% CI] / always collision', 'Invalid'], body, [56, 68, 46, 153, 153, 52])
    else:
        paragraph('No held-out forecast trials completed at report generation. No forecast result is claimed.')
    measured = [m for m in models.values() if m['position_accuracy']['mean'] is not None]
    if measured:
        position_means = [m['position_accuracy']['mean'] for m in measured]
        paragraph(f'On the sampled held-out boards, episode-weighted position accuracy ranges '
                  f'from {percent(min(position_means))} to {percent(max(position_means))} across '
                  'the collected artifacts. These are aggregate descriptive results for the '
                  'selected input formats, not estimates over a natural gameplay distribution.')
        comparisons = [(r['accuracy']['mean'], r['linear']['mean'])
                       for m in measured for r in m['curve'] if r['kind'] == 'position']
        ahead = sum(linear > model for model, linear in comparisons)
        paragraph(f'The declared linear control has higher point accuracy in {ahead} of '
                  f'{len(comparisons)} collected model-by-horizon position comparisons. '
                  'This count does not test statistical significance and does not explain which '
                  'internal stage produced the errors.')
    paragraph('Intervals in the table aggregate across horizons. Per-horizon curves below are the '
              'primary diagnostic. A periodic or nonmonotone curve is not automatically evidence '
              'of compounding error: wrap and timer phase can make individual horizons easier. '
              'The two foresight summaries are the largest horizon reaching 80% accuracy and the '
              'contiguous prefix that stays above 80%; they answer different questions. Curve-shape '
              'confirmation and multiple-model thinking comparisons are not completed in this cycle.')
    if figures:
        image_path = figures[0]
        picture = Image(str(image_path))
        scale = min(528/picture.imageWidth, 290/picture.imageHeight)
        picture.drawWidth, picture.drawHeight = picture.imageWidth*scale, picture.imageHeight*scale
        picture.hAlign = 'CENTER'
        picture.keepWithNext = True
        story.append(picture)
        paragraph('Figure 1. Forecast accuracy by horizon. Shading shows episode-cluster bootstrap '
                  'intervals. Each panel uses the actual completed trials; inspect the evidence JSON '
                  'for per-horizon counts. Oracle predictions define the horizontal ceiling. '
                  'The linear control is computed on the same states.', 'SmallNote')
    if len(figures) > 1:
        picture = Image(str(figures[1]))
        scale = min(528/picture.imageWidth, 180/picture.imageHeight)
        picture.drawWidth, picture.drawHeight = picture.imageWidth*scale, picture.imageHeight*scale
        picture.hAlign = 'CENTER'
        picture.keepWithNext = True
        story.append(picture)
        paragraph('Figure 2. Signed coordinate error on valid numeric position replies only, '
                  'with episode-cluster bootstrap intervals. Zero means no average coordinate '
                  'bias, not perfect forecasts: opposite errors can cancel. This is ordinary '
                  'predicted x minus true x, not circular distance or distance along travel. '
                  'Wrap can create large jumps. Invalid replies remain incorrect in accuracy '
                  'but cannot enter this numeric summary.', 'SmallNote')
    if models:
        body = [[name.replace('qwen3:', '').replace('-q4_K_M', ''), m['profile']['best_representation'],
                 str(m['f_largest_80']), str(m['f_prefix_80']),
                 str(m['required_foresight']['median_feasible']),
                 percent(m['required_foresight']['prefix_coverage_feasible'])] for name, m in models.items()]
        table(['Model tag', 'Selected input', 'Largest k at 80%', '80% prefix', 'Median h*', 'Prefix covers h*'], body,
              [55, 120, 82, 82, 82, 107])
    if models:
        confidence_rows = [[name.replace('qwen3:', '').replace('-q4_K_M', ''),
                            percent(m['mean_confidence_valid']['mean']),
                            percent(m['position_ece_valid']), percent(m['collision_ece_valid'])]
                           for name, m in models.items()]
        table(['Model tag', 'Mean valid confidence', 'Position ECE', 'Collision ECE'], confidence_rows,
              [80, 160, 144, 144])
        paragraph('Confidence and ECE use only valid self-reports; primary accuracy includes every '
                  'trial. ECE uses ten fixed confidence bins. ECE is descriptive and does not '
                  'establish a cognitive mechanism.', 'SmallNote')
        fit_rows = [[name.replace('qwen3:', '').replace('-q4_K_M', ''),
                     f'{m["position_signed_error_valid"]["mean"]:.2f}' if m['position_signed_error_valid']['mean'] is not None else '-',
                     f'{m["position_absolute_error_valid"]["mean"]:.2f}' if m['position_absolute_error_valid']['mean'] is not None else '-',
                     *[f'{m["position_shape_fits"][kind]["rmse"]:.3f}' for kind in ('linear', 'exponential', 'cliff')]]
                    for name, m in models.items() if m['position_shape_fits']]
        if fit_rows:
            table(['Model tag', 'Signed error', 'Absolute error', 'Linear RMSE', 'Exponential RMSE', 'Cliff RMSE'],
                  fit_rows, [70, 80, 80, 95, 108, 95])
            paragraph('Cell errors condition on valid numeric position answers. Shape fits use the '
                      'episode-weighted position means: ordinary least-squares line, a*exp(b*k) '
                      'with b on a fixed [-1,1] grid at 0.005 spacing, and two constant levels with '
                      'one breakpoint. RMSE is in-sample accuracy error. The cliff has an extra '
                      'selected parameter; lower RMSE is not evidence that it generalizes or that '
                      'compounding error caused the curve. No confirmatory shape test is claimed.', 'SmallNote')
        prediction_rows = [[name.replace('qwen3:', '').replace('-q4_K_M', ''),
                            len(m['position_shape_prediction']['folds']),
                            *[f"{m['position_shape_prediction']['summary'][kind]['curve_mse']['mean']:.4f}"
                              for kind in ('flat', 'linear', 'exponential', 'cliff')]]
                           for name, m in models.items() if m['position_shape_prediction']['summary']]
        if prediction_rows:
            table(['Model', 'Held-out episodes', 'Flat MSE', 'Line MSE', 'Exp. MSE', 'Cliff MSE'],
                  prediction_rows, [80, 105, 85, 85, 85, 88])
            paragraph('A post-hoc follow-up leaves one whole source episode out, fits the same '
                      'fixed curve families on remaining episodes, clips predictions to [0,1], '
                      'and scores the unseen episode. A flat training-mean control tests whether '
                      'shape helps at all. Training fits weight saved trials equally; scoring weights '
                      'horizons and held-out episodes equally. Malformed answers count as wrong. '
                      'Evidence JSON includes fold membership, predictions, Brier scores and '
                      'episode-bootstrap intervals. Training folds overlap, so these intervals '
                      'are descriptive and do not capture every source of fit uncertainty. This '
                      'does not confirm a preregistered shape or frozen-to-live prediction.', 'SmallNote')
        event_rows = [[name.replace('qwen3:', '').replace('-q4_K_M', ''),
                       m['first_error_events']['complete_sequences'],
                       *[m['first_error_events']['first_error_event'][kind]
                         for kind in ('timer_wait', 'timer_move', 'fast_step', 'wrap')],
                       m['first_error_events']['all_correct']]
                      for name, m in models.items() if m['first_error_events']['complete_sequences']]
        if event_rows:
            table(['Model tag', 'Sequences', 'Timer wait', 'Timer move', 'Fast step', 'Wrap tick', 'No error'],
                  event_rows, [70, 70, 80, 80, 80, 78, 70])
            paragraph('Earliest incorrect direct-forecast horizon, grouped by the actual object-0 '
                      'transition at that horizon. A sequence is one board and sampling seed with '
                      'all ten horizons; repeats are not independent boards. These are separately '
                      'asked predictions, not one chained rollout. Counts do not adjust for event '
                      'exposure and cannot establish that wrap causes errors. Freeway objects keep '
                      'their lane and direction, so direction-change onset is not an event in this '
                      'simulator. Incomplete sequences are excluded here and retained in accuracy.', 'SmallNote')
    heading('5. Targeted repairs and consistency')
    paragraph('Implemented forecast conditions include direct answers, rule reminders, scratchpad '
              'tables, chained tables, model-written code, and a supplied deterministic forecast. '
              'The table reports only completed contrasts. Code is interpreted in '
              'a bounded arithmetic subset with no imports, attributes, or arbitrary Python '
              'execution. Invalid programs count as wrong. The supplied forecast and simulator '
              'shield are upper-bound conditions, excluded from model-only claims.')
    body = []
    for name, m in models.items():
        for r in m['repairs']:
            delta = r['difference_from_base']
            body.append([name.replace('qwen3:', '').replace('-q4_K_M', ''), r['name'],
                         f'{r["states"]} / {r["trials"]}', ci(r['accuracy']), ci(delta, difference=True),
                         str(r['invalid'])])
    if body:
        table(['Model', 'Position repair', 'Boards / trials', 'Accuracy [95% CI]',
               'Paired change [pp]', 'Invalid'], body, [60, 80, 75, 132, 130, 51])
        paragraph('Repair intervals are exploratory and unadjusted; no significance claim is made. '
                  'The paired change column is candidate minus base in percentage points. The '
                  'small repair bank limits external validity. Intervals condition on the five '
                  'decoding seeds; cached programs are reused across states rather than counted '
                  'as fresh program samples. Base and repaired trials are paired '
                  'on the same source episodes, target horizons and sampling seeds where available.', 'SmallNote')
        code_conditions = [r for m in models.values() for r in m['repairs'] if r['name'] == 'code']
        if code_conditions:
            paragraph(f'The restricted-code condition has {sum(r["invalid"] for r in code_conditions)} '
                      f'invalid executions out of {sum(r["trials"] for r in code_conditions)} '
                      'reported position trials. Program reuse makes these correlated failures. '
                      'The contrast measures compatibility with the declared grammar as well as '
                      'program specification; it cannot settle the usefulness of more permissive '
                      'forecast offloading.')
    else:
        paragraph('No isolated-repair test completed at report generation. The implementation exists, '
                  'but this cycle supplies no repair-effect estimate yet.')
    paragraph('Consistency probes separate a forecast-only prompt from a fresh action prompt. '
              'We re-ask actions with no forecast, the model forecast, and the oracle forecast. '
              'The reported contradiction event is choosing an action the elicited forecast '
              'labels colliding under repetition. Because live choices apply for one tick, this '
              'is a repeated-action risk-disagreement proxy, not a logical contradiction or the '
              'formal belief-action gap. Commitment-matched results, if completed, are reported '
              'separately below. The proxy also '
              'does not capture every instance of skipped safe progress. '
              'Forecast-bound selection deterministically prioritizes U, S, D among actions '
              'the model marks noncolliding. It removes that measured contradiction by construction '
              'where its beliefs admit a noncolliding action; it cannot make an incorrect forecast true.')
    commitment_rows = [[name.replace('qwen3:', '').replace('-q4_K_M', ''),
                        r['condition'] + (' (UB)' if r['upper_bound'] else ''), f'{r["states"]}/{r["trials"]}',
                        ci(r['accuracy']), percent(r['forecast_accuracy_valid_only']['mean']),
                        ci(r['inconsistency_valid_only']) + f' (n={r["inconsistency_valid_only"]["observations"]})',
                        str(r['invalid'])]
                       for name, m in models.items() for r in m.get('commitment', [])]
    if commitment_rows:
        heading('5.1. Commitment-matched consistency')
        paragraph('A separate diagnostic explicitly commits the chosen action for exactly three '
                  'ticks without revision. Collision or deadline failure makes the commitment unsafe; '
                  'a goal ends it successfully. The preferred choice is U if safe, then S, then D; '
                  'if none is safe, the required output is null. The forecast refers to that same '
                  'commitment. We count unsafe choice when the elicited beliefs admit a safe action, '
                  'skipped believed-safe progress, and failure to abstain when all commitments are '
                  'believed unsafe. This fixes the temporal mismatch of the legacy proxy, while '
                  'remaining a behavioral report-based measurement rather than a causal mechanism test.')
        paragraph('None uses a fresh action prompt with a separately elicited forecast for '
                  'comparison; own/oracle inject model/reference forecasts. Combined jointly '
                  'reports action and forecast, and generic adds a check instruction. Bound '
                  'selects from model beliefs. Retry makes one extra action call after an unsafe '
                  'choice or failed abstention, retaining the initial beliefs. Shield applies '
                  'the priority selector to oracle beliefs, making it a construction ceiling. '
                  'Action accuracy enforces the full preference rule; inconsistency reports '
                  'the three events defined above. This follow-up diagnostic does not refit '
                  'the already-frozen live policy.')
        table(['Model tag', 'Condition', 'Boards / trials', 'Action accuracy [95% CI]',
               'Forecast accuracy', 'Inconsistency, valid records', 'Invalid'], commitment_rows,
              [50, 75, 44, 115, 81, 123, 40])
        paragraph('UB marks oracle information or a simulator shield. Bound selection uses model '
                  'beliefs only; zero inconsistency is a construction property and does not imply '
                  'correct dynamics. Invalid outputs count against primary accuracy; inconsistency '
                  'and mean per-action forecast accuracy condition on valid action/forecast records. '
                  'The parenthesized n is that conditional record count. '
                  'Separate component rates are retained in the evidence JSON. '
                  'All action and forecast prompts use '
                  'the same declared commitment horizon.', 'SmallNote')
    else:
        paragraph('The commitment-matched consistency task is implemented and tested, but no '
                  'completed empirical commitment run is included at report generation.')
    heading('6. Composed agent and live validation')
    paragraph('The composed agent selects a representation and repairs from validation diagnostics '
              'using fixed stage cutoffs: 0.8 for A/B and 0.9 for C/D/E. A is ASCII perception '
              'accuracy, B is pooled forecast accuracy, and C is one minus the legacy risk-disagreement '
              'rate. D is unprofiled and set to 1. E is set to 0.5 when the largest null-state '
              'conditional action probability exceeds 0.6, otherwise 1. A structured-input flag '
              'keeps ASCII when ASCII wins validation. Forecast code and forecast-bound selection use model '
              'beliefs. Prior calibration subtracts null-state from observed-state action log '
              'scores, following the contextual-calibration idea [8] and requiring all A/B/C '
              'candidates. A separate one-token action interface is '
              'used for those scores, which is a prompt-interface confound for causal attribution. '
              'History is explicitly unprofiled in this cycle. The all-repair condition is a '
              'control; random same-count and leave-one-out variants are implemented for '
              'subsequent tests. Policy messages retain up '
              'to twelve raw proposal turns, subject to the server context limit.')
    paragraph('Live decisions advance the world by ceil(measured request time / one second), with '
              'S as the default action during delay. The chosen move applies afterward if the '
              'episode has not ended. Retries and extra model calls add delay and generated tokens. '
              'Local Python parsing and arithmetic time is not included in this replay. A '
              'successfully constructed forecast program is reused within its episode; failed '
              'generation or construction can be attempted again on later decisions. Every '
              'attempt is counted. Cache hits reuse recorded durations and generated-token counts, '
              'so costs describe logical policy requests in the replay. Fixed null-prior scores '
              'are charged at every calibration decision even if cached. This replays observed latency; '
              'it is not an asynchronous deployment benchmark. Episodes cut short by the time '
              'cap or a shortened decision budget are censored. Simulator baselines have zero inference delay and therefore '
              'provide reference controls rather than latency-matched competitors.')
    if evidence['live']:
        body = [[r['model'].replace('qwen3:', '').replace('-q4_K_M', ''), r['condition'],
                 f'{r["episodes"]}/{r["censored"]}', ci(r['success_wilson']),
                 '-' if r['mean_tokens'] is None else f'{r["mean_tokens"]:.0f}',
                 '-' if r['mean_collisions'] is None else f'{r["mean_collisions"]:.1f}',
                 f'{r.get("error_decisions", 0)}/{r.get("decisions", 0)}']
                for r in evidence['live']]
        table(['Model', 'Condition', 'Done / censored', 'Success [Wilson 95% CI]',
               'Generated tokens / ep.', 'Collisions / ep.', 'Error decisions / total'],
              body, [70, 50, 45, 135, 80, 68, 80])
        paragraph('Costs, collisions and errors summarize completed episodes. Errors are parse, '
                  'controller or program failures, counted once per decision; the controller '
                  'falls back to S on those errors. Incorrect forecasts that parse successfully '
                  'are not counted as controller errors.', 'SmallNote')
        paragraph('Live success intervals use the Wilson score interval to avoid degenerate bounds '
                  'when every episode succeeds or fails. The requested bootstrap intervals are '
                  'also retained in the evidence JSON. These methods agree on the point estimate '
                  'but quantify uncertainty differently.', 'SmallNote')
        model_live = [r for r in evidence['live'] if r['model'] != 'simulator control' and r['episodes']]
        if model_live:
            success_rates = [r['success']['mean'] for r in model_live]
            paragraph(f'Completed model conditions have observed success rates from '
                      f'{percent(min(success_rates))} to {percent(max(success_rates))}. '
                      'Paired repair contrasts use shared episode seeds where available. '
                      'Partial conditions and timing differences limit comparisons across models; '
                      'a failed repair remains part of the evidence.')
        if evidence.get('live_paired'):
            table(['Model', 'Repair vs base', 'Shared episodes', 'Success change [pp, 95% CI]',
                   'Generated-token change / ep.'],
                  [[r['model'].replace('qwen3:', '').replace('-q4_K_M', ''), r['condition'],
                    str(r['paired_episodes']), ci(r['success_change'], difference=True),
                    f'{r["token_change"]["mean"]:.0f}'] for r in evidence['live_paired']],
                  [70, 90, 75, 183, 110])
            paragraph('Paired changes use only shared, uncensored episode seeds and percentile '
                      'bootstrap intervals across episodes. They are exploratory and unadjusted. '
                      'Identical outcomes on every observed pair can give a degenerate bootstrap '
                      'interval; this does not establish zero population difference. Token changes '
                      'include additional calls and retries, but are not a cost-matched comparison.', 'SmallNote')
    else:
        paragraph('No live episode completed at report generation. Frozen accuracy alone cannot '
                  'establish usefulness during live play.')
    followup = evidence.get('followup')
    if followup is not None:
        heading('6.1. A separate overnight follow-up')
        paragraph('After reviewing the first cycle, we widened the tiny code language and fixed '
                  'repair routing. These are exploratory follow-ups, not confirmations registered '
                  'before seeing the first results. The v2 policy retries a generated program once '
                  'using fixed training-only counterexamples. Failed repairs leave the model forecast '
                  'available. Unbound actions and calibrated action scores now receive repaired '
                  'forecasts. Code checking and execution time add to v2 delay. No training or paid '
                  'model service is used; simulator-assisted code checks are not unassisted knowledge.')
        forecast_rows = [r for r in followup['forecasts'] if r['kind'] == 'position']
        if forecast_rows:
            table(['Model', 'Cohort / repair', 'Boards / trials', 'Accuracy [95% CI]', 'Invalid', 'Paired change [pp]'],
                  [[r['model'].replace('qwen3:', '').replace('-q4_K_M', ''),
                    ('Separate calls' if 'B-chained-calls' in r['cohort'] else 'Grammar') + ' / ' +
                    r['repair'] + (' (UB)' if r['repair'] == 'tool' else ''),
                    f'{r["states"]}/{r["trials"]}', ci(r['accuracy']), r['invalid'],
                    ci(r['difference_from_base'], difference=True)] for r in forecast_rows],
                  [45, 120, 65, 115, 43, 140])
            paragraph('Grammar code samples are reused across boards and horizons. Unique request '
                      'counts and paired denominators are saved in the evidence JSON. A valid program '
                      'can still predict the wrong cell. Separate-call chaining is a different cohort '
                      'and output budget; compare each repair only with its own paired base. The old '
                      'chained condition asks for a table in one reply. UB marks the supplied '
                      'simulator forecast, not a model-only repair. Intervals are descriptive.', 'SmallNote')
        paragraph(f'The follow-up saved {followup["context_trials"]:,} held-out diary/framing trials. '
                  'The target is four boards per model with five sampling seeds. Repeated answers '
                  'are not new boards. We transplant diaries from separate episodes while keeping '
                  'the current board fixed. Outcomes are replayed; real histories record whether a '
                  'proposed move actually ran. Oracle outcome feedback concerns the source episode. '
                  'Delta summaries also concern that episode, not a hidden prior test-board frame.')
        diary_rows = [r for r in followup['context'] if r['stage'] == 'D' and (
            r['condition']['length'] == 0 or
            (r['condition']['length'] == 12 and r['condition']['kind'] in ('stay', 'own') and
             (r['condition']['kind'] == 'stay' or not r['condition']['outcomes'])))]
        if diary_rows:
            table(['Model', 'Diary / repair', 'Boards / trials', 'Accuracy [95% CI]', 'Action flip', 'Unsafe flip'],
                  [[r['model'].replace('qwen3:', '').replace('-q4_K_M', ''),
                    ('No diary' if r['condition']['length'] == 0 else
                     r['condition']['kind'] + ' / ' + r['condition']['repair'] +
                     (' + outcomes' if r['condition']['outcomes'] else '')),
                    f'{r["states"]}/{r["trials"]}', ci(r['accuracy']),
                    percent(r['action_change']['mean']), ci(r['harmful_flip'])] for r in diary_rows],
                  [42, 130, 64, 112, 65, 115])
        framing_rows = [r for r in followup['context'] if r['stage'] == 'E' and
                        r['condition']['paraphrase'] == 0 and
                        (r['condition']['skin'] == 'cars' or r['condition']['labels'] == 'standard')]
        if framing_rows:
            table(['Model', 'Story / labels', 'Boards / trials', 'Accuracy [95% CI]', 'Action flip', 'Forecast flip'],
                  [[r['model'].replace('qwen3:', '').replace('-q4_K_M', ''),
                    'Safe object 0' if r['condition']['reversal'] else
                    'Named game' if r['condition']['name_leak'] else
                    'Strategy hint' if r['condition']['repair'] == 'strategy' else
                    r['condition']['skin'] + ' / ' + r['condition']['labels'],
                    f'{r["states"]}/{r["trials"]}', ci(r['accuracy']),
                    percent(r['action_change']['mean']), percent(r['forecast_change']['mean'])]
                   for r in framing_rows], [42, 130, 64, 112, 65, 115])
        if diary_rows or framing_rows:
            paragraph('Accuracy counts malformed replies as wrong. Flip rates use valid paired '
                      'answers only: same board and decoding seed, compared with the no-diary or '
                      'cars/standard-label base. Unsafe flip means a safe base choice changed to an '
                      'unsafe choice. Forecast flip is the fraction of three collision beliefs that '
                      'changed. Contact reversal changes the rules and reference labels, so it is '
                      'a different task, not a pure framing effect. Full conditions, invalid counts, '
                      '95% episode-cluster intervals and exploratory Holm-adjusted sign-flip tests '
                      'are saved in JSON. The tiny cohort limits generalization.', 'SmallNote')
        if followup['live']:
            table(['Model', 'V2 condition', 'Complete / censored', 'Goals', 'Success [95% Wilson]', 'Tokens / run'],
                  [[r['model'].replace('qwen3:', '').replace('-q4_K_M', ''), r['condition'],
                    f'{r["episodes"]}/{r["censored"]}', r['successes'], ci(r['success_wilson']),
                    f'{r["mean_tokens"]:.0f}' if r['mean_tokens'] is not None else '-'] for r in followup['live']],
                  [80, 80, 90, 40, 150, 88])
            paragraph('V2 uses fresh environment seeds 60000-60049, disjoint from the old model '
                      'diaries. Models share seeds across conditions. Four validation boards and one '
                      'decoding seed set the new C/D scores; fixed cutoffs remain unchanged. These '
                      'small profiles are noisy. Successful code is checked on training seeds '
                      '1000-1005, never on the live test board. Cache hits reuse measured request '
                      'durations and token counts; v2 local program time is measured again. Controls '
                      'have zero inference delay. Counts are actual coverage, not a claim that all '
                      'planned 50-seed conditions finished.', 'SmallNote')
        else:
            paragraph('No v2 live episode is included yet. Better frozen forecasts alone do not '
                      'show that the new agent reaches the goal.')
        if followup['live_paired']:
            table(['Model', 'V2 contrast vs base', 'Shared episodes', 'Success change [pp, 95% CI]'],
                  [[r['model'].replace('qwen3:', '').replace('-q4_K_M', ''), r['condition'],
                    r['paired_episodes'], ci(r['success_change'], difference=True)]
                   for r in followup['live_paired']], [80, 120, 100, 228])
        paragraph('This follow-up does not finish the planned 200/300-board matrices, other games, '
                  'thinking sweeps, extra model families, causal cascades, or cost-matched controls. '
                  'A simulator-assisted program repair and a short history manipulation cannot '
                  'establish a unique internal failure cause or the value of diagnosis targeting.')
        compiled = followup['compiled']
        if compiled['groups']:
            heading('6.2. Saved programs and delay')
            paragraph(f'This separate replay used {compiled["source_programs"]} saved generation '
                      f'requests and completed {compiled["complete"]:,} program-episode cases '
                      f'({compiled["censored"]} censored). It made no fresh model requests. Each model '
                      'has only five generated programs. Fifty shared environment seeds per program '
                      'do not make 250 independent program samples.')
            paragraph('Canonical object parsing and a fixed action selector assist this controller. '
                      'Executable programs may have wrong physics. The checked gate rejects any '
                      'program that fails fixed training examples and falls back to staying still. '
                      'Saved generation delay is charged once per episode or on every turn; the '
                      'per-turn charge is synthetic, not a fresh measured call. Local execution '
                      'time is measured in both cases. These controls do not isolate diagnosis targeting.')
            table(['Version / model', 'Charge / gate', 'Goals / complete', 'Success [95% CI]'],
                  [[r['protocol'].removeprefix('compiled-controller-')+' / '+
                    r['model'].replace('qwen3:', '').replace('-q4_K_M', ''),
                    r['condition']+' / '+r['validation'],
                    f'{r["successes"]}/{r["episodes"]}', ci(r['success'])]
                   for r in compiled['groups']], [95, 160, 100, 173])
            paragraph('Intervals resample generated programs and shared environment seeds as two '
                      'crossed axes. They require all five programs on a common completed seed set; '
                      'partial counts remain visible. Version 1 preserves an audited NumPy-number '
                      'adapter bug. Version 2 fixes that adapter and repeats the same design. '
                      'They are never pooled. The concurrently frozen neural v2 study retains its '
                      'original adapter. Per-program results and paired contrasts are in evidence JSON.')
    heading('7. Limits and unfinished tests')
    paragraph('The complete research plan is not finished. This cycle does not establish '
              'cross-game generality, thinking-mode effects, causal cascade attribution, or '
              'frozen-to-live predictive validity. Snapshot and bounded-search support for Snake '
              'and Overcooked is tested, but their diagnostic banks and empirical replications '
              'are pending. The 0.6B model, a second model family, larger-model ceiling, full '
              'three-paraphrase robustness, full diary/outcome history matrices, framing and danger '
              'reversal study, cost-matched monitors, leave-one-repair-out live tests, and '
              'compute-conditional training remain outside the completed evidence unless a '
              'separate result file explicitly records them.')
    paragraph('The task-specific input includes exact timers and motion rules, which makes it '
              'a rule-application test rather than inference from natural gameplay alone. Forecast '
              'errors can include input-reading errors: the main test does not condition on per-state '
              'perception success. Code offload removes some parsing burden as well as arithmetic, '
              'so its effect does not isolate a unique internal failure mechanism. The position '
              'target advances car motion without collision-driven resets. The frozen prompt calls '
              'these car ticks without explicitly stating that conditioning; the ambiguity further '
              'limits failure attribution. '
              'Renderer tests retain all object facts; this differs from the plan\'s truncated '
              'nearest-object lane table and radius-limited relation list. The bank is balanced '
              'by design and changes the natural state distribution. Self-reported confidence '
              'is not a calibrated probability. Small repair samples, one main model family, '
              'and an output budget that can truncate reasoning all limit inference. The live '
              'base is an instrumented forecast-and-action prompt, and retained history contains '
              'raw model proposals that may differ from the controller-executed action. A plain '
              'action-only baseline and execution-corrected history remain future controls. No '
              'novelty priority, venue readiness, or broad model-size law is claimed.')
    heading('8. Reproducibility and review status')
    paragraph('The main collector loads frozen source commit '
              'f71b9259abd055291c7c865659363aa1b3d9ec0f; commitment-test manifests record their '
              'separate module hashes. Canonical wins use lane 0, although upstream redraws the '
              'player at lane 9 in its terminal observation. The frozen collector also redraws '
              'initial cars after a collision on tick 100; current code retains the upstream '
              'terminal cars. This last-board display difference changes no success label, '
              'collision count, elapsed tick or later decision. The offline audit checks each '
              'physical tick against upstream and applies the frozen display convention only '
              'for that declared source revision.')
    paragraph('The repository contains the preregistration, execution-plan deviations, pinned '
              'upstream revision, labeled bank, tests, runner, raw response logs or losslessly '
              'compressed copies, and this paper generator. The evidence JSON records the '
              'SHA-256 of input logs. Each model manifest records exact identity and decoding '
              'settings. Published numbers are computed from saved trials, including failures; '
              'they are not filled from the research plan. Author list, affiliation, and venue '
              'format are left for human review.')
    paragraph(f'At report generation, the main four-model forecast target is '
              f'{"complete" if complete else "incomplete"}. This PDF is a review artifact for '
              'the completed local cycle, not a claim that the full proposed project is done.', 'SmallNote')
    heading('9. Conclusion')
    if sum(r['episodes'] for r in agent_live) and not sum(r['successes'] for r in agent_live):
        paragraph(f'None of the {sum(r["episodes"] for r in agent_live)} completed model-condition '
                  f'runs reached the goal. These runs reuse {live_seed_count} environment seeds '
                  'across models and conditions. This cycle does not demonstrate a goal-completion '
                  'benefit from diagnosis-selected repairs under the chosen latency setting. '
                  'Small live cohorts and wide Wilson bounds limit this negative result.')
    if followup is not None:
        newer = [r for r in followup['live'] if r['model'] != 'simulator control']
        if newer:
            paragraph(f'The separate v2 cohort records {sum(r["successes"] for r in newer)} goals '
                      f'in {sum(r["episodes"] for r in newer)} completed model-condition runs. '
                      'It uses fresh seeds and changed repair behavior. Only paired conditions '
                      'inside that cohort support a repair comparison; the two cycles are not pooled.')
    paragraph('The artifact makes stage probes and repair decisions inspectable against a pinned '
              'simulator. The completed results describe the tested prompts, states and local '
              'model artifacts; they do not establish that a weak probe score uniquely identifies '
              'an internal failure cause. Commitment matching removes one measurement ambiguity, '
              'and recording delay and all calls makes the live comparison auditable. A broader '
              'claim that diagnosis reliably selects better agents requires the pending '
              'cross-game, cost-matched and intervention controls.')
    reference_blocks = [Paragraph('References', styles['Heading2'])]
    markdown.append('\n## References')
    for index, (text, url) in enumerate(REFERENCES, 1):
        entry = f'[{index}] {escape(text)} <link href="{url}" color="#215c8c">{escape(url)}</link>'
        reference_blocks.append(Paragraph(entry, styles['SmallNote']))
        markdown.append(entry)
    story.append(KeepTogether(reference_blocks))
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output), pagesize=(612, 792), rightMargin=42, leftMargin=42,
                            topMargin=43, bottomMargin=43, title='Diagnose, Then Repair',
                            author='Research artifact for review')

    def footer(canvas, document):
        canvas.setStrokeColor(colors.HexColor('#dce3e7'))
        canvas.line(42, 33, 570, 33)
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#647484'))
        canvas.drawString(42, 21, 'Diagnose, Then Repair | Local study | Review copy')
        canvas.drawRightString(570, 21, str(document.page))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    (output.parent / 'paper.md').write_text('\n\n'.join(markdown) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path('results/cycle'))
    parser.add_argument('--bank', type=Path, default=Path('data/freeway.json'))
    parser.add_argument('--followup', type=Path, help='Separate second-cycle records; never pooled with the first')
    parser.add_argument('--output', type=Path, default=Path('output/pdf/diagnose-then-repair.pdf'))
    args = parser.parse_args()
    evidence = analyze(args.results, args.bank)
    if args.followup is not None:
        evidence['followup'] = analyze_followup(args.followup)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / 'evidence.json').write_text(json.dumps(evidence, indent=2))
    figure = curves_figure(evidence, args.output.parent / 'forecast-curves.png')
    error_figure = signed_error_figure(evidence, args.output.parent / 'signed-errors.png')
    build(evidence, args.output, [p for p in (figure, error_figure) if p], args.results)
    print(args.output.resolve())


if __name__ == '__main__':
    main()
