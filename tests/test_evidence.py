"""These tests make the evidence checker reject changed scores and impossible episode histories."""
from copy import deepcopy
import json

import pytest

from diagnose.live import baseline, episode
from diagnose.model import LocalModel
from diagnose.runner import run
from diagnose.world import initial, label
from scripts.check_evidence import check, verify_live
from test_study import FakeOllama


def test_evidence_audit_detects_score_tampering(tmp_path):
    s = initial(1000)
    bank = [{'id': s.key, 'split': 'test', 'state': s.to_dict(), 'oracle': label(s)}]
    bank_path = tmp_path / 'bank.json'
    bank_path.write_text(json.dumps({'situations': bank}))
    model = LocalModel('test', tmp_path / 'cache', transport=FakeOllama())
    root = tmp_path / 'run'
    result = run(bank, model, root, stages=('B',), horizons=(1,), seeds=(0,))
    assert check(root, bank_path)['forecast_trials'] == 2
    result[0]['score']['correct'] = not result[0]['score']['correct']
    (root / 'responses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in result))
    with pytest.raises(ValueError, match='Forecast score mismatch'):
        check(root, bank_path)


def test_physics_audit_detects_changed_live_outcomes():
    record = episode(baseline('advance'), seed=50000)
    verify_live(record)
    changed = deepcopy(record)
    changed['success'] = not changed['success']
    with pytest.raises(ValueError, match='success label'):
        verify_live(changed)
    changed = deepcopy(record)
    changed['trace'][0]['after']['lane'] = 0
    with pytest.raises(ValueError, match='after-state'):
        verify_live(changed)


def test_physics_audit_checks_delay_and_cost_totals():
    record = episode(baseline('stay'), seed=50000, max_decisions=2)
    changed = deepcopy(record)
    changed['trace'][0]['delay_ticks'] = 1
    with pytest.raises(ValueError, match='delay mismatch'):
        verify_live(changed)
    changed = deepcopy(record)
    changed['tokens'] = 1
    with pytest.raises(ValueError, match='cost totals'):
        verify_live(changed)
