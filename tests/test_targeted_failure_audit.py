"""These checks keep post-hoc format inspection separate from the experiment's strict scores."""
from copy import deepcopy
import json

import pytest

from scripts.targeted_failure_audit import analyze, compare, inspect_payload

SOURCE = 'def forecast(x, timer, p, k):\n    return x'


def row(branch='original', round_=0, alternative=False):
    content = json.dumps({'candidate' if alternative else 'code': SOURCE, 'confidence': .5})
    return dict(model='m', program_id=0, branch=branch, round=round_,
                source=None if alternative else SOURCE,
                error='malformed: missing fields' if alternative else None,
                feedback={} if round_ == 0 else {'examples': []},
                response={'raw': {'message': {'content': content}}, 'key': f'{branch}-{round_}',
                          'cache_hit': False, 'identity': {'request': {
                              'messages': [{'role': 'user', 'content': 'prompt'}],
                              'options': {'seed': round_}}}})


def test_alternative_key_is_only_static_inspection(monkeypatch):
    from diagnose.repairs import ArithmeticProgram

    def forbidden(*args):
        raise AssertionError('No alternative program may execute')

    monkeypatch.setattr(ArithmeticProgram, '__call__', forbidden)
    value = row(alternative=True)
    before = deepcopy(value)
    result = inspect_payload(value)
    assert result['alternative_candidate_string']
    assert result['alternative_static_grammar_accepted']
    assert not result['alternative_executed']
    assert not result['strict_source_available']
    assert result['strict_error'] == 'malformed: missing fields'
    assert value == before


def test_missing_sources_are_not_identical_programs():
    result = compare(row(alternative=True), row(alternative=True))
    assert result['both_sources_unavailable']
    assert not result['same_available_source']
    assert result['same_raw_content']


def test_invalid_alternative_grammar_remains_invalid():
    value = row(alternative=True)
    value['response']['raw']['message']['content'] = json.dumps({'candidate': 'import os'})
    result = inspect_payload(value)
    assert result['alternative_candidate_string']
    assert not result['alternative_static_grammar_accepted']
    assert result['alternative_static_grammar_error']


def test_complete_report_preserves_invalid_primary_and_counts_cache():
    rows = [row('original', 0), row('uniform', 1), row('targeted', 1),
            row('uniform', 2, True), row('targeted', 2, True)]
    rows[-1]['response']['cache_hit'] = True
    report = analyze(rows, {'models': ['m'], 'program_ids': [0]})
    assert not report['primary_results_changed']
    assert report['alternative_program_executions'] == 0
    assert report['new_model_calls'] == 0
    assert len(report['feedback_pairs']) == 2
    assert len(report['final_pairs']) == 3
    targeted = next(r for r in report['arm_summary'] if r['branch'] == 'targeted')
    assert targeted['strict_valid'] == 0
    assert targeted['alternative_static_grammar_accepted'] == 1
    requests = next(r for r in report['request_counts'] if r['branch'] == 'targeted')
    assert requests['cache_hits'] == 1
    with pytest.raises(ValueError, match='cohort'):
        analyze(rows[:-1], {'models': ['m'], 'program_ids': [0]})


def test_malformed_json_is_not_salvaged():
    value = row()
    value['response']['raw']['message']['content'] = '```python bad'
    result = inspect_payload(value)
    assert result['json_keys'] is None
    assert not result['alternative_candidate_string']
