"""These checks stop missing or invalid measurements from turning into paper claims."""

import json

import pytest

from paper.build_targeted import input_receipt, load_json, percentage, safe_output, signed_points


def test_missing_measurements_are_not_zero():
    assert percentage(None) == 'unavailable'
    assert signed_points(None) == 'unavailable'
    assert percentage(0) == '0.0%'
    assert signed_points(-0.125) == '-12.5 pp'


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -0.01, 1.01, True, '0.5'])
def test_invalid_accuracy_rejected(value):
    with pytest.raises(ValueError):
        percentage(value)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1.01, 1.01, True, '0.5'])
def test_invalid_contrast_rejected(value):
    with pytest.raises(ValueError):
        signed_points(value)


def test_nonfinite_input_rejected(tmp_path):
    path = tmp_path / 'summary.json'
    path.write_text('{"accuracy": NaN}')
    with pytest.raises(ValueError, match='Non-finite'):
        load_json(path)


def test_receipt_changes_with_evidence(tmp_path):
    path = tmp_path / 'summary.json'
    path.write_text(json.dumps({'n': 1}))
    first = input_receipt([path])
    path.write_text(json.dumps({'n': 2}))
    assert first != input_receipt([path])


def test_old_paper_is_protected(tmp_path):
    with pytest.raises(ValueError, match='previous study'):
        safe_output(tmp_path / 'diagnose-then-repair.pdf')
    with pytest.raises(ValueError, match='PDF'):
        safe_output(tmp_path / 'new.txt')
    assert safe_output(tmp_path / 'new.pdf') == tmp_path / 'new.pdf'


def test_unapproved_evidence_cannot_render(tmp_path):
    from paper.build_targeted import collect_evidence
    with pytest.raises(ValueError, match='audit'):
        collect_evidence({'audit': {'passed': False}}, tmp_path)


def test_compressed_records_costs_and_feedback(tmp_path):
    import gzip
    from paper.build_targeted import collect_evidence
    common = {'model': 'test-model', 'program_id': 0}
    forecast = []
    live = []
    candidates = []
    trace = {'trace': [{'response': {'cpu_seconds': .01, 'program_executable': True,
                                    'planned_delay_ticks': 1, 'actual_delay_ticks': 1}}]}
    for branch in ('original', 'uniform', 'targeted'):
        forecast.extend([{**common, 'branch': branch, 'correct': correct, 'valid': True}
                         for correct in (True, False)])
        live.append({**common, 'branch': branch, 'regime': 'native', 'goals': 0,
                     'censored': False, 'trace_file': 'trace.jsonl.gz', 'env_seed': 3,
                     'collisions': 0, 'final_lane': 1})
        candidates.append({**common, 'branch': branch, 'round': int(branch != 'original'),
                           'error': None, 'feedback': {'target_category': 'wrap',
                                                      'examples': [{'id': 'a'}]},
                           'response': {'wall_seconds': 10, 'cache_hit': branch == 'targeted',
                                        'raw': {'eval_count': 20, 'prompt_eval_count': 30}}})
    with gzip.open(tmp_path / 'trace.jsonl.gz', 'wt') as handle:
        handle.writelines(json.dumps({**row, 'record': trace}) + '\n' for row in live)
    for filename, rows in [('forecast.jsonl.gz', forecast), ('live.jsonl', live),
                           ('candidates.jsonl', candidates)]:
        opener = gzip.open if filename.endswith('.gz') else open
        with opener(tmp_path / filename, 'wt') as handle:
            handle.writelines(json.dumps(row) + '\n' for row in rows)
    summary = {'audit': {'passed': True, 'scope': 'full'}, 'forecast_contrasts': [],
               'board_macro_contrasts': [], 'live_contrasts': [], 'protocol': {},
               'forecast_file': 'forecast.jsonl.gz', 'live_file': 'live.jsonl',
               'candidates_file': 'candidates.jsonl'}
    import hashlib
    summary['audit']['input_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in tmp_path.iterdir()}
    result = collect_evidence(summary, tmp_path)
    assert result['forecasts'][('test-model', 0, 'targeted')] == [.5]
    assert result['costs'][('test-model', 'targeted')]['seconds'] == 10
    assert result['costs'][('test-model', 'targeted')]['fresh_seconds'] == 0
    assert result['costs'][('test-model', 'uniform')]['fresh_seconds'] == 10
    assert result['overlaps'] == [1.]
    assert result['counters']['delay_mismatches'] == 0

    assert result['counters']['decisions'] == 3
    assert len([k for k in result['paths'] if k.startswith('trace_shard:')]) == 1
    live[0]['goals'] = 1
    (tmp_path / 'live.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in live))
    with pytest.raises(ValueError, match='changed after audit'):
        collect_evidence(summary, tmp_path)
    summary['audit']['input_sha256']['live.jsonl'] = hashlib.sha256(
        (tmp_path / 'live.jsonl').read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='disagree'):
        collect_evidence(summary, tmp_path)


def test_overlap_disclosure_comes_from_evidence():
    from paper.build_targeted import input_overlap_text
    assert '3 of 4' in input_overlap_text({'input_coverage': {'test_cases': 4,
                                                            'seen_test_cases': 3}})
    with pytest.raises(ValueError, match='coverage'):
        input_overlap_text({'input_coverage': {'test_cases': 0, 'seen_test_cases': 0}})
