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
