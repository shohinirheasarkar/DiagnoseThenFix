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
        data['first_error_events'] = first_error_events(test, bank)
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
                                           'unsafe_choice_own_feasible': interval(eligible, 'contradiction'),
                                           'trials': len(group), 'invalid': sum(not r['score']['valid'] for r in group),
                                           'states': len({r['state_id'] for r in group}),
                                           'upper_bound': name in ('oracle', 'shield')})
        output['models'][profile['model']] = data
        for path in folder.glob('live-*.jsonl*'):
            group = rows(path)
            complete = [r for r in group if not r.get('censored')]
            values = [{'state_id': r['seed'], 'success': r['success']} for r in complete]
            output['live'].append({'model': profile['model'], 'condition': path.name.removesuffix('.gz').removesuffix('.jsonl')[5:],
                                  'episodes': len(complete), 'censored': len(group)-len(complete),
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
                              'success': bootstrap([{'state_id': r['seed'], 'success': r['success']} for r in group], 'success'),
                              'success_wilson': wilson(sum(r['success'] for r in group),len(group)),
                              'mean_tokens': 0, 'mean_calls': 0,
                              'mean_collisions': statistics.mean(r['collisions'] for r in group) if group else None})
    return output


def curves_figure(evidence, output):
    models = list(evidence['models'])
    if not models:
        return None
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'font.family': 'DejaVu Sans'})
    figure, axes = plt.subplots(2, len(models), figsize=(3.05*len(models), 5.1), squeeze=False,
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
            if row == 1:
                ax.set_xlabel('Forecast horizon (car ticks)')
            if col == 0:
                ax.set_ylabel(('Exact head position' if row == 0 else 'Collision prediction') + '\nAccuracy')
    axes[0][0].legend(loc='lower left', fontsize=7)
    figure.savefig(output, dpi=180, bbox_inches='tight')
    plt.close(figure)
    return output


def percent(value):
    return 'unavailable' if value is None else f'{100*value:.1f}%'


def ci(value):
    if value['mean'] is None:
        return 'unavailable'
    return f'{percent(value["mean"])} [{100*value["low"]:.1f}, {100*value["high"]:.1f}]'


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
        story.extend((t, Spacer(1, 10)))
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
    title = 'Diagnose, Then Repair'
    paragraph(title, 'PaperTitle')
    paragraph('A simulator-checked study of small language-model agents in real-time Freeway', 'Heading2')
    paragraph('Research paper for review | November 12, 2026 | Time-capped local study', 'SmallNote')
    heading('Abstract')
    paragraph(f'A correct action in a changing world requires a readable observation, an adequate forecast, '
              f'and a decision that follows that forecast. We implement a stage-oriented diagnostic suite '
              f'with replayable states, six renderers, simulator-checked labels, and targeted inference-time '
              f'repairs. This report analyzes {total:,} held-out forecast trials from {len(models)} installed '
              f'Qwen3 model artifacts collected on a 16 GB Apple M3 laptop. The situation bank contains '
              f'500 states with disjoint source episodes across splits. We compare forecast accuracy at '
              f'horizons 1-10 against constant-velocity extrapolation, and separately evaluate consistency '
              f'repairs and measured-delay live play where completed. {finding}Missing or malformed answers count '
              f'as incorrect. This is an exploratory, single-game artifact; the full multi-game, '
              f'thinking-mode, and training-based study remains incomplete. We make no first-of-its-kind '
              f'or causal root-cause claim from this cycle.')
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
              'Intervals are percentile 95% bootstrap intervals over source episodes, with 2,000 '
              'resamples; sampling repeats do not become independent boards.')
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
    paragraph('We compare direct answers with rule reminders, scratchpad tables, chained tables, '
              'model-written code, and a supplied deterministic forecast. Code is interpreted in '
              'a bounded arithmetic subset with no imports, attributes, or arbitrary Python '
              'execution. Invalid programs count as wrong. The supplied forecast and simulator '
              'shield are upper-bound conditions, excluded from model-only claims.')
    body = []
    for name, m in models.items():
        for r in m['repairs']:
            delta = r['difference_from_base']
            body.append([name.replace('qwen3:', '').replace('-q4_K_M', ''), r['name'],
                         f'{r["states"]} / {r["trials"]}', ci(r['accuracy']), ci(delta)])
    if body:
        table(['Model', 'Position repair', 'Boards / trials', 'Accuracy [95% CI]',
               'Paired change [pp]'], body, [60, 85, 75, 173, 135])
        paragraph('Repair intervals are exploratory and unadjusted; no significance claim is made. '
                  'The paired change column is candidate minus base in percentage points. The '
                  'small repair bank limits external validity. Intervals condition on the five '
                  'decoding seeds; cached programs are reused across states rather than counted '
                  'as fresh program samples. Base and repaired trials are paired '
                  'on the same source episodes, prompts and sampling seeds where available.', 'SmallNote')
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
                        r['condition'] + (' (UB)' if r['upper_bound'] else ''), str(r['trials']),
                        ci(r['accuracy']), percent(r['forecast_accuracy_valid_only']['mean']),
                        ci(r['inconsistency_valid_only']), str(r['invalid'])]
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
        table(['Model tag', 'Condition', 'Trials', 'Action accuracy [95% CI]',
               'Forecast accuracy', 'Inconsistency, valid records', 'Invalid'], commitment_rows,
              [50, 75, 34, 115, 91, 123, 40])
        paragraph('UB marks oracle information or a simulator shield. Bound selection uses model '
                  'beliefs only; zero inconsistency is a construction property and does not imply '
                  'correct dynamics. Invalid outputs count against primary accuracy; inconsistency '
                  'and mean per-action forecast accuracy condition on valid action/forecast records. '
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
              'History is explicitly unprofiled in this cycle. The all-repair and random '
              'conditions are controls, not assumed improvements.')
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
              'cap are censored. Simulator baselines have zero inference delay and therefore '
              'provide reference controls rather than latency-matched competitors.')
    if evidence['live']:
        body = [[r['model'].replace('qwen3:', '').replace('-q4_K_M', ''), r['condition'],
                 str(r['episodes']), ci(r['success_wilson']),
                 '-' if r['mean_tokens'] is None else f'{r["mean_tokens"]:.0f}',
                 '-' if r['mean_collisions'] is None else f'{r["mean_collisions"]:.1f}']
                for r in evidence['live']]
        table(['Model', 'Condition', 'Episodes', 'Success [Wilson 95% CI]', 'Generated tokens / ep.', 'Collisions / ep.'],
              body, [90, 60, 50, 148, 90, 90])
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
                    str(r['paired_episodes']), ci(r['success_change']),
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
    heading('7. Limits and unfinished tests')
    paragraph('The complete research plan is not finished. This cycle does not establish '
              'cross-game generality, thinking-mode effects, causal cascade attribution, or '
              'frozen-to-live predictive validity. Snapshot and bounded-search support for Snake '
              'and Overcooked is tested, but their diagnostic banks and empirical replications '
              'are pending. The 0.6B model, a second model family, larger-model ceiling, full '
              'three-paraphrase robustness, diary/outcome history study, framing and danger '
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
    paragraph('The artifact makes stage probes and repair decisions inspectable against a pinned '
              'simulator. The completed results describe the tested prompts, states and local '
              'model artifacts; they do not establish that a weak probe score uniquely identifies '
              'an internal failure cause. Commitment matching removes one measurement ambiguity, '
              'and recording delay and all calls makes the live comparison auditable. A broader '
              'claim that diagnosis reliably selects better agents requires the pending '
              'cross-game, cost-matched and intervention controls.')
    heading('References')
    for index, (text, url) in enumerate(REFERENCES, 1):
        paragraph(f'[{index}] {escape(text)} <link href="{url}" color="#215c8c">{escape(url)}</link>', 'SmallNote')
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
    parser.add_argument('--output', type=Path, default=Path('output/pdf/diagnose-then-repair.pdf'))
    args = parser.parse_args()
    evidence = analyze(args.results, args.bank)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / 'evidence.json').write_text(json.dumps(evidence, indent=2))
    figure = curves_figure(evidence, args.output.parent / 'forecast-curves.png')
    build(evidence, args.output, [figure] if figure else [], args.results)
    print(args.output.resolve())


if __name__ == '__main__':
    main()
