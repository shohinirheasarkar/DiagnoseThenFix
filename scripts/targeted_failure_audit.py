"""This explains saved repair failures without changing scores or running any repaired program."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnose.model import canonical, read_jsonl
from diagnose.repairs import ArithmeticProgram
from scripts.check_targeted import index_rows
from scripts.summarize_targeted import file_hash


def raw_content(row):
    return row['response']['raw'].get('message', {}).get('content')


def inspect_payload(row):
    content = raw_content(row)
    try:
        value = json.loads(content)
    except (TypeError, ValueError):
        value = None
    keys = sorted(value) if isinstance(value, dict) else None
    candidate = value.get('candidate') if isinstance(value, dict) else None
    alternative = isinstance(candidate, str) and 'code' not in value
    grammar_accepted, grammar_error = False, None
    if alternative:
        try:
            ArithmeticProgram(candidate, extended=True, native_inputs=True)
            grammar_accepted = True
        except (TypeError, ValueError, SyntaxError) as exc:
            grammar_error = str(exc)
    return {'json_keys': keys, 'strict_source_available': row['source'] is not None,
            'strict_error': row['error'], 'alternative_candidate_string': alternative,
            'alternative_static_grammar_accepted': grammar_accepted,
            'alternative_static_grammar_error': grammar_error,
            'alternative_executed': False}


def compare(left, right):
    both = isinstance(left['source'], str) and isinstance(right['source'], str)
    return {'same_available_source': both and left['source'] == right['source'],
            'both_sources_unavailable': left['source'] is None and right['source'] is None,
            'same_raw_content': raw_content(left) == raw_content(right),
            'same_request_key': left['response']['key'] == right['response']['key']}


def analyze(rows, design):
    indexed = index_rows(rows, ('model', 'program_id', 'branch', 'round'))
    expected = {(m, p, b, r) for m in design['models'] for p in design['program_ids']
                for b, r in [('original', 0), ('uniform', 1), ('targeted', 1),
                             ('uniform', 2), ('targeted', 2)]}
    if set(indexed) != expected:
        raise ValueError('Need the complete frozen candidate cohort')
    final, pairs, feedback_pairs, request_counts, initial = [], [], [], [], []
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['model'], row['branch'])].append(row)
        if row['round'] == (0 if row['branch'] == 'original' else 2):
            final.append({k: row[k] for k in ('model', 'program_id', 'branch')} |
                         inspect_payload(row))
    for (model, branch), group in sorted(grouped.items()):
        request_counts.append({'model': model, 'branch': branch, 'logical_requests': len(group),
                               'cache_hits': sum(r['response']['cache_hit'] for r in group),
                               'distinct_request_keys': len({r['response']['key'] for r in group}),
                               'distinct_raw_contents': len({raw_content(r) for r in group})})
    for model in design['models']:
        starts = [indexed[(model, p, 'original', 0)] for p in design['program_ids']]
        initial.append({'model': model, 'programs': len(starts),
                        'distinct_prompt_texts': len({canonical(r['response']['identity']['request']
                                                               ['messages']) for r in starts}),
                        'distinct_sampling_seeds': len({r['response']['identity']['request']
                                                       ['options']['seed'] for r in starts}),
                        'distinct_request_keys': len({r['response']['key'] for r in starts}),
                        'distinct_raw_contents': len({raw_content(r) for r in starts}),
                        'distinct_available_sources': len({r['source'] for r in starts
                                                           if r['source'] is not None})})
        for pid in design['program_ids']:
            for left, right in [('original', 'uniform'), ('original', 'targeted'),
                                ('uniform', 'targeted')]:
                a = indexed[(model, pid, left, 0 if left == 'original' else 2)]
                b = indexed[(model, pid, right, 2)]
                pairs.append({'model': model, 'program_id': pid, 'left': left, 'right': right,
                              **compare(a, b)})
            for round_ in (1, 2):
                a, b = (indexed[(model, pid, branch, round_)] for branch in ('uniform', 'targeted'))
                ax, bx = a['feedback']['examples'], b['feedback']['examples']
                aset, bset = {r['id'] for r in ax}, {r['id'] for r in bx}
                feedback_pairs.append({'model': model, 'program_id': pid, 'round': round_,
                                       'same_examples_in_order': canonical(ax) == canonical(bx),
                                       'same_example_ids': aset == bset,
                                       'uniform_examples': len(ax), 'targeted_examples': len(bx),
                                       'same_prompt': canonical(a['response']['identity']['request']
                                                                ['messages']) == canonical(
                                            b['response']['identity']['request']['messages']),
                                       **compare(a, b)})
    arm_summary = []
    for model in design['models']:
        for branch in ('original', 'uniform', 'targeted'):
            group = [r for r in final if r['model'] == model and r['branch'] == branch]
            arm_summary.append({'model': model, 'branch': branch, 'programs': len(group),
                                'strict_valid': sum(r['strict_error'] is None for r in group),
                                'strict_source_available': sum(r['strict_source_available'] for r in group),
                                'json_key_sets': dict(Counter(canonical(r['json_keys']) for r in group)),
                                'strict_errors': dict(Counter(r['strict_error'] or 'none' for r in group)),
                                'alternative_candidate_strings': sum(r['alternative_candidate_string']
                                                                     for r in group),
                                'alternative_static_grammar_accepted': sum(
                                    r['alternative_static_grammar_accepted'] for r in group)})
    return {'post_hoc': True, 'primary_results_changed': False,
            'new_model_calls': 0, 'alternative_program_executions': 0,
            'scope': 'saved response structure, exact text identity, feedback and cache provenance',
            'interpretation': 'Static grammar acceptance is not correctness or gameplay evidence. '
                              'Identical outputs or unavailable programs limit the intervention; '
                              'zero outcome contrasts do not establish diagnosis is ineffective.',
            'arm_summary': arm_summary, 'initial_diversity': initial,
            'request_counts': request_counts, 'final_pairs': pairs,
            'feedback_pairs': feedback_pairs, 'final_payloads': final}


def build(root):
    root = Path(root)
    audit = json.loads((root / 'audit.json').read_text())
    if audit.get('passed') is not True or audit.get('scope') != 'full':
        raise ValueError('A full passed independent audit is required')
    names = ('manifest.json', 'candidates.jsonl')
    for name in names:
        if audit['input_sha256'].get(name) != file_hash(root / name):
            raise ValueError('Candidate evidence changed after audit')
    manifest = json.loads((root / 'manifest.json').read_text())
    design = manifest['design']
    if (manifest['status'] != 'frozen' or hashlib.sha256(canonical(design).encode()).hexdigest()
            != manifest['design_sha256']):
        raise ValueError('Frozen design mismatch')
    return analyze(read_jsonl(root / 'candidates.jsonl'), design) | {
        'input_sha256': {name: file_hash(root / name) for name in (*names, 'audit.json')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = build(args.root)
    output = args.output or args.root / 'failure-audit.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'Saved post-hoc failure audit: {output}')


if __name__ == '__main__':
    main()
