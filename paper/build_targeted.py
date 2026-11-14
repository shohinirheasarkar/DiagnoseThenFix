"""This makes a separate paper from checked repair-study results, keeping old studies untouched."""

import argparse
from collections import Counter, defaultdict
import statistics
import hashlib
import gzip
from html import escape
import json
import math
from pathlib import Path

TITLE = 'Does Diagnostic Targeting Improve Forecast Repair?'
REFERENCES = [
    ('Wen et al. (2025). Real-Time Reasoning Agents in Evolving Environments.',
     'https://arxiv.org/abs/2511.04898'),
    ('Lehrach et al. (2025). Code World Models for General Game Playing.',
     'https://arxiv.org/abs/2510.04542'),
    ('Lou et al. (2026). AutoHarness: improving LLM agents by automatically '
     'synthesizing a code harness.', 'https://arxiv.org/abs/2603.03329'),
    ('Orvalho et al. (2025). Counterexample Guided Program Repair Using Zero-Shot '
     'Learning and MaxSAT-based Fault Localization. AAAI.',
     'https://arxiv.org/abs/2502.07786'),
    ('Zhu et al. (2025). Where LLM Agents Fail and How They can Learn From Failures.',
     'https://arxiv.org/abs/2509.25370'),
]


def percentage(value):
    if value is None:
        return 'unavailable'
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError('Expected a numeric proportion')
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Proportion outside [0, 1]')
    return f'{100 * value:.1f}%'


def signed_points(value):
    if value is None:
        return 'unavailable'
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError('Expected a numeric contrast')
    if not math.isfinite(value) or not -1 <= value <= 1:
        raise ValueError('Contrast outside [-1, 1]')
    return f'{100 * value:+.1f} pp'


def load_json(path):
    def reject_constant(value):
        raise ValueError(f'Non-finite JSON constant: {value}')
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def input_receipt(paths):
    return {str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in paths}


def safe_output(path):
    path = Path(path).resolve()
    if path.name == 'diagnose-then-repair.pdf':
        raise ValueError('The previous study paper must not be overwritten')
    if path.suffix.lower() != '.pdf':
        raise ValueError('Output must be a PDF')
    return path


def paragraph(text, style):
    from reportlab.platypus import Paragraph
    return Paragraph(escape(str(text)), style)


def page_footer(canvas, document):
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.setFillColorRGB(0.35, 0.35, 0.35)
    canvas.drawString(document.leftMargin, 30, 'Diagnosis-guided forecast repair')
    canvas.drawRightString(576, 30, str(document.page))
    canvas.restoreState()


def read_rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def collect_evidence(summary, root):
    if summary.get('audit', {}).get('passed') is not True:
        raise ValueError('A passed independent audit is required')
    required = ('forecast_contrasts', 'board_macro_contrasts', 'live_contrasts', 'protocol')
    if any(key not in summary for key in required):
        raise ValueError('Incomplete audited summary')
    paths = {key: root / summary[key] for key in
             ('forecast_file', 'live_file', 'candidates_file')}
    forecasts = defaultdict(lambda: [0, 0])
    goals, timing = defaultdict(list), defaultdict(list)
    counters, costs = Counter(), defaultdict(lambda: Counter())
    selections, feedback = Counter(), {}
    for row in read_rows(paths['forecast_file']):
        key = (row['model'], row['program_id'], row['branch'])
        if type(row['correct']) is not bool:
            raise ValueError('Forecast correctness must be Boolean')
        forecasts[key][0] += int(row['correct'])
        forecasts[key][1] += 1
        counters['forecast_cases'] += 1
        counters['invalid_forecasts'] += not row['valid']
    trace_shards = defaultdict(dict)
    identity = ('model', 'program_id', 'branch', 'regime', 'env_seed')
    metrics = ('goals', 'collisions', 'final_lane', 'censored')
    live_keys = set()
    for row in read_rows(paths['live_file']):
        key = tuple(row[field] for field in identity)
        if key in live_keys:
            raise ValueError('Duplicate live metric key')
        live_keys.add(key)
        trace_shards[row['trace_file']][key] = row
        goals[key[:4]].append(row['goals'])
        counters['episodes'] += 1
        counters['censored_episodes'] += bool(row['censored'])
    for filename, expected in trace_shards.items():
        seen = set()
        for row in read_rows(root / filename):
            key = tuple(row[field] for field in identity)
            if key not in expected or key in seen:
                raise ValueError('Unexpected or duplicate trace key')
            if any(row[field] != expected[key][field] for field in metrics):
                raise ValueError('Trace and flat metrics disagree')
            seen.add(key)
            for frame in row['record']['trace']:
                response = frame['response']
                timing[(row['model'], row['regime'])].append(response['cpu_seconds'])
                counters['delay_mismatches'] += (response['actual_delay_ticks'] !=
                                                 response['planned_delay_ticks'])
                counters['decisions'] += 1
                counters['fallback_decisions'] += not response['program_executable']
        if seen != set(expected):
            raise ValueError('Trace shard is missing live metric keys')
        paths[f'trace_shard:{filename}'] = root / filename
    for row in read_rows(paths['candidates_file']):
        key = (row['model'], row['branch'])
        response = row['response']
        costs[key]['requests'] += 1
        costs[key]['seconds'] += response['wall_seconds']
        if not response['cache_hit']:
            costs[key]['fresh_seconds'] += response['wall_seconds']
        costs[key]['cache_hits'] += bool(response['cache_hit'])
        costs[key]['output_tokens'] += response['raw'].get('eval_count', 0)
        costs[key]['input_tokens'] += response['raw'].get('prompt_eval_count', 0)
        costs[key]['invalid'] += row['error'] is not None
        if row['round']:
            feedback[(row['model'], row['program_id'], row['round'], row['branch'])] = {
                item['id'] for item in row['feedback']['examples']}
            if row['branch'] == 'targeted':
                selections[row['feedback']['target_category'] or 'no failures'] += 1
    overlaps = []
    for (model, pid, round_, branch), ids in feedback.items():
        if branch != 'targeted':
            continue
        other = feedback.get((model, pid, round_, 'uniform'))
        if other is None:
            raise ValueError('Missing paired feedback branch')
        union = ids | other
        if union:
            overlaps.append(len(ids & other) / len(union))
    if not forecasts or not goals:
        raise ValueError('Actual forecast and live records are required')
    return dict(paths=paths, forecasts={k: [total / count] for k, (total, count) in forecasts.items()}, goals=dict(goals),
                timing=dict(timing), counters=dict(counters), costs=dict(costs),
                selections=dict(selections), overlaps=overlaps)


def short_model(name):
    return name.replace('qwen3:', '').replace('-q4_K_M', '')


def make_figures(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models = sorted({k[0] for k in data['forecasts']})
    figures = []
    for field, suffix, label in [('forecasts', 'forecast', 'Forecast accuracy'),
                                 ('goals', 'goals', 'Goal rate (native delay)')]:
        fig, axes = plt.subplots(1, len(models), figsize=(7.2, 2.8), squeeze=False)
        for model, ax in zip(models, axes[0]):
            rows = {k[1:]: statistics.mean(v) for k, v in data[field].items()
                    if k[0] == model and (field == 'forecasts' or k[3] == 'native')}
            ids = sorted({k[0] for k in rows})
            for pid in ids:
                values = [rows[(pid, b) if field == 'forecasts' else (pid, b, 'native')]
                          for b in ('original', 'uniform', 'targeted')]
                ax.plot(range(3), values, color='#8c9daf', alpha=.32, linewidth=.8)
            means = [statistics.mean([rows[(pid, b) if field == 'forecasts'
                                            else (pid, b, 'native')] for pid in ids])
                     for b in ('original', 'uniform', 'targeted')]
            ax.plot(range(3), means, 'o-', color='#125b76', linewidth=2)
            ax.set(xticks=range(3), xticklabels=['Original', 'Uniform', 'Targeted'],
                   ylim=(-.03, 1.03), title=f'{short_model(model)}: {len(ids)} programs',
                   ylabel=label)
            ax.grid(axis='y', alpha=.15)
        fig.tight_layout()
        path = output / f'targeted-{suffix}.png'
        fig.savefig(path, dpi=180)
        plt.close(fig)
        figures.append(path)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    labels, medians = [], []
    for (model, regime), values in sorted(data['timing'].items()):
        labels.append(f'{short_model(model)}\n{regime}')
        medians.append(statistics.median(values) * 1000)
    axes[0].bar(range(len(labels)), medians, color='#317f93')
    axes[0].set(xticks=range(len(labels)), xticklabels=labels, ylabel='Median execution (ms)')
    axes[0].tick_params(axis='x', labelsize=7)
    for model in models:
        values = []
        for regime in ('zero', 'native', 'two_ticks'):
            cells = [statistics.mean(v) for k, v in data['goals'].items()
                     if k[0] == model and k[2] == 'targeted' and k[3] == regime]
            values.append(statistics.mean(cells))
        axes[1].plot(range(3), values, 'o-', label=short_model(model))
    axes[1].set(xticks=range(3), xticklabels=['Zero', 'Native', 'Two ticks'],
                ylabel='Targeted goal rate', ylim=(-.03, 1.03))
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = output / 'targeted-latency.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append(path)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    labels = sorted(data['selections'])
    axes[0].barh(labels, [data['selections'][k] for k in labels], color='#317f93')
    axes[0].set_xlabel('Targeted repair rounds')
    axes[0].tick_params(axis='y', labelsize=8)
    if data['overlaps']:
        axes[1].hist(data['overlaps'], bins=[i / 10 for i in range(12)], color='#b97937')
    axes[1].set(xlabel='Paired feedback Jaccard overlap', ylabel='Repair rounds', xlim=(0, 1.05))
    fig.tight_layout()
    path = output / 'targeted-feedback.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append(path)
    return figures


def render(summary_path, output):
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Image, PageBreak, SimpleDocTemplate, Spacer, Table, TableStyle
    summary_path, output = Path(summary_path), safe_output(output)
    summary = load_json(summary_path)
    data = collect_evidence(summary, summary_path.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    figures = make_figures(data, output.parent)
    styles = getSampleStyleSheet()
    styles['Normal'].fontSize = 10
    styles['Normal'].leading = 14
    story = []

    def text(value, style='Normal'):
        story.extend([paragraph(value, styles[style]), Spacer(1, 7)])

    def table(rows, widths):
        item = Table([[paragraph(x, styles['Normal']) for x in row] for row in rows],
                     colWidths=widths, repeatRows=1, hAlign='LEFT')
        item.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8f1f4')),
                                  ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                  ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
                                  ('LINEBELOW', (0, 0), (-1, 0), .5, colors.grey)]))
        story.extend([item, Spacer(1, 10)])

    def figure(index, caption):
        story.extend([Image(str(figures[index]), width=504, height=196), Spacer(1, 5)])
        text(caption)

    text(TITLE, 'Title')
    text('A paired Freeway experiment with executable forecasts', 'Heading2')
    text('Abstract', 'Heading2')
    primary = [r for r in summary['forecast_contrasts'] if r['primary']]
    results = '; '.join(f"{short_model(r['model'])}: {signed_points(r['mean'])} "
                        f"(95% interval {signed_points(r['low'])} to {signed_points(r['high'])})"
                        for r in primary)
    text('Does choosing feedback from a diagnosed weak category improve code repair more '
         'than uniformly sampling true failures? We compare both strategies from shared '
         'initial programs using two local Qwen model sizes. Both receive at most six '
         'examples per round and two repair rounds. The primary held-out forecast '
         f'contrast, targeted minus uniform, is {results}. We separately evaluate gameplay '
         'through one fixed human-written controller. These results concern feedback '
         'selection for one component; they do not establish a general diagnostic agent.')
    text('1. Why this comparison?', 'Heading2')
    text('Executable game models, environment feedback, and program repair are established '
         'ideas [2-5]. Real-time environments also make decision delay consequential [1]. '
         'We isolate the extra value of category-based feedback selection, keeping the '
         'initial program, prompt structure, repair limits, and controller fixed. A '
         'forecast improvement is not assumed to improve gameplay.')
    text('2. Design', 'Heading2')
    text('Every initial program has three branches: original, uniformly selected failure '
         'feedback, and category-targeted failure feedback. Repairs always use the final '
         'round; unsuccessful or invalid programs remain in the analysis. The two arms '
         "share each round's sampling seed and counterbalance execution order. Each "
         'prompt contains the current program and current feedback, without accumulated chat.')
    text('Training, development validation, and test episodes are separate. Targeting ranks '
         'predefined behavioral categories by training error rate. It prioritizes failures '
         'from the worst category and fills remaining slots from other failures. Both arms '
         'receive fewer than six examples when too few failures exist. With no failures, '
         'the prompt supplies no failing example. Actual costs are reported separately.')
    story.append(PageBreak())
    text('3. Held-out forecast results', 'Heading1')
    figure(0, 'Figure 1. Each thin line is one initial-program family; the thick line is the '
           'mean. All three branches share test cases. Thousands of cases do not count '
           'as thousands of independent model generations.')
    rows = [['Model', 'Targeted - uniform', '95% interval', 'Holm p']]
    for r in primary:
        rows.append([short_model(r['model']), signed_points(r['mean']),
                     f"{signed_points(r['low'])} to {signed_points(r['high'])}",
                     f"{r['p_value_holm']:.4f}"])
    table(rows, [90, 125, 190, 99])
    text('The primary score weights forecast cases equally and counts invalid forecasts '
         'as wrong. Bootstrap intervals preserve program pairing. Sign-flip tests use '
         'Holm adjustment over the two primary model comparisons. These are comparisons '
         'of sampled programs under this protocol, not independent draws of every forecast.')
    rows = [['Model', 'Board-macro sensitivity', '95% interval']]
    for r in summary['board_macro_contrasts']:
        if r['contrast'] == 'targeted-uniform':
            rows.append([short_model(r['model']), signed_points(r['mean']),
                         f"{signed_points(r['low'])} to {signed_points(r['high'])}"])
    table(rows, [90, 180, 234])
    text(f"The archive contains {data['counters']['forecast_cases']:,} forecast records, "
         f"including {data['counters']['invalid_forecasts']:,} invalid forecasts. "
         'Board-macro scoring gives each board equal weight and is a sensitivity check.')
    story.append(PageBreak())
    text('4. Does repair transfer to gameplay?', 'Heading1')
    figure(1, 'Figure 2. Goal rates under native execution delay. Lines pair branches '
           'from the same initial program; environment seeds are shared.')
    rows = [['Model / delay', 'Targeted - uniform', '95% interval']]
    for r in summary['live_contrasts']:
        if r['contrast'] == 'targeted-uniform':
            rows.append([f"{short_model(r['model'])} / {r['regime']}",
                         signed_points(r['mean']),
                         f"{signed_points(r['low'])} to {signed_points(r['high'])}"])
    table(rows, [155, 120, 229])
    text('The controller compares three decision cycles, including the waiting interval '
         'before each action. Only the forecast component is model-generated. Goal '
         'intervals preserve program pairing and shared environment seeds. This is a '
         'closed-loop rollout: changing delay changes future states and decisions.')
    text(f"Recorded episodes: {data['counters']['episodes']:,}; censored: "
         f"{data['counters']['censored_episodes']:,}. Fallback was used on "
         f"{data['counters']['fallback_decisions']:,} of "
         f"{data['counters']['decisions']:,} recorded decisions. A zero contrast with "
         'a wide interval is inconclusive; zero observed successes do not prove zero '
         'population success probability.')
    story.append(PageBreak())
    text('5. Delay and preparation cost', 'Heading1')
    figure(2, 'Figure 3. Measured execution time and targeted-branch goal rates across '
           'delay conditions. Injected delays are controlled stress tests, not measured '
           'language-model inference latency.')
    text('Generation and repair happen offline. Their cost is not charged anew at each '
         'game decision. Native execution is measured while task workloads are isolated. '
         'One game tick is one second; ceiling rounding charges any positive subsecond '
         'execution one tick. The controller plans one native wait tick, so executions '
         f'exceeding one second are a planning-delay mismatch. Recorded mismatches: '
         f"{data['counters']['delay_mismatches']:,}.")
    rows = [['Model / branch', 'Calls', 'Input / output tokens', 'Fresh / logical seconds', 'Cache hits']]
    for (model, branch), cost in sorted(data['costs'].items()):
        rows.append([f'{short_model(model)} / {branch}', str(cost['requests']),
                     f"{cost['input_tokens']:,} / {cost['output_tokens']:,}",
                     f"{cost['fresh_seconds']:.1f} / {cost['seconds']:.1f}", str(cost['cache_hits'])])
    table(rows, [140, 45, 145, 85, 89])
    text('Original-generation cost is shared by both repair alternatives and listed once. '
         'Repair rows contain the two additional rounds. Cached response wall times are '
         'saved generation charges, not new computation; cache counts make this visible. '
         'Equal token caps are not equal actual token use.')
    story.append(PageBreak())
    text('6. What feedback was selected?', 'Heading1')
    figure(3, 'Figure 4. Targeted categories and paired feedback-set overlap. Jaccard '
           'overlap is intersection divided by union. Empty unions are unavailable and '
           'excluded from the histogram, not scored as zero.')
    text('These mechanism checks show how often the interventions actually supplied '
         'different examples. Later repair rounds can have different candidate failures '
         'because the first repairs differ. Category labels summarize externally observed '
         'errors; they do not identify an internal reasoning mechanism.')
    text('7. Discussion and limits', 'Heading1')
    text('This experiment separates targeted feedback from the general benefit of repair. '
         'A forecast gain without a gameplay gain is a transfer limitation under this '
         'controller. Similar outcomes with wide uncertainty do not establish equivalence. '
         'A negative result is limited to this selection rule, feedback budget, interface, '
         'and models. The supplied rules and correct examples are human/simulator assistance.')
    text('Only Freeway and two sizes in one model family are tested. The generated motion '
         'function is simpler than a full world model. The fixed controller can limit or '
         'mask forecast improvements. Tick rounding creates a coarse latency threshold. '
         'The initial programs, not episodes, determine how much independent evidence '
         'we have about generation and repair variability. Earlier studies are development '
         'context and are not pooled into this experiment.')
    text('8. Reproducibility', 'Heading1')
    text('The release includes prompts, candidates, correct feedback examples, split '
         'identities, raw responses, per-case outcomes, replayable trajectories, analysis '
         'code, and an evidence manifest. The independent audit must pass before this '
         'builder accepts the summary. Protocol deviations and assisted-reference results '
         'must accompany the final evidence package.')
    story.append(PageBreak())
    text('References', 'Heading1')
    for index, (title, url) in enumerate(REFERENCES, 1):
        text(f'[{index}] {title} {url}')
    text('Evidence receipt', 'Heading1')
    receipt = input_receipt([summary_path, *data['paths'].values()])
    for path, sha in receipt.items():
        if path in {str(Path(value).resolve()) for key, value in data['paths'].items()
                    if not key.startswith('trace_shard:')} | {str(summary_path.resolve())}:
            text(f'{Path(path).name}: {sha}', 'Code')
    text(f'All {len(receipt)} input hashes, including trace shards, are saved in the '
         'companion evidence JSON file.')
    text('This paper describes the new targeted-repair study only. Previous paper and '
         'evidence archives remain separate. A passed audit checks records and scoring; '
         'it is not a guarantee of scientific novelty or publication acceptance.')
    SimpleDocTemplate(str(output), pagesize=(612, 792), rightMargin=54, leftMargin=54,
                      topMargin=45, bottomMargin=48, title=TITLE).build(
                          story, onFirstPage=page_footer, onLaterPages=page_footer)
    output.with_suffix('.evidence.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', required=True, type=Path)
    parser.add_argument('--output', type=Path,
                        default=Path('output/pdf/diagnosis-guided-repair.pdf'))
    args = parser.parse_args()
    render(args.summary, args.output)


if __name__ == '__main__':
    main()
