"""These checks keep paper statistics tied to an unchanged, complete audit receipt."""
import gzip
import json

import pytest

from diagnose.model import canonical
from scripts.check_targeted import digest
from scripts.summarize_targeted import file_hash, summarize


def make_archive(root):
    design = {'models': ['a', 'b'], 'program_ids': [0, 1], 'live_seeds': [11, 12],
              'regimes': ['zero']}
    objects = {'manifest.json': {'design': design, 'design_sha256': digest(design),
                                 'status': 'frozen'},
               'train.json': [{'id': 'train', 'input': [1, 0, 1, 1]}],
               'validation.json': [],
               'test.json': [{'id': 'test', 'input': [1, 0, 1, 1]}]}
    for name, value in objects.items():
        (root / name).write_text(json.dumps(value))
    (root / 'candidates.jsonl').write_text('{}\n')
    (root / 'trace.jsonl.gz').write_bytes(b'audited-trace-fixture')
    forecasts, live = [], []
    for model in design['models']:
        for pid in design['program_ids']:
            for branch in ('original', 'uniform', 'targeted'):
                forecasts.append(dict(model=model, program_id=pid, branch=branch,
                                      state_id='test', board_id='board', horizon=1,
                                      correct=branch == 'targeted', valid=True))
                for seed in design['live_seeds']:
                    live.append(dict(model=model, program_id=pid, branch=branch,
                                     regime='zero', env_seed=seed, goals=int(branch == 'targeted'),
                                     collisions=0, final_lane=0 if branch == 'targeted' else 9,
                                     trace_file='trace.jsonl.gz'))
    with gzip.open(root / 'forecast.jsonl.gz', 'wt') as handle:
        for row in forecasts:
            handle.write(canonical(row) + '\n')
    (root / 'live.jsonl').write_text(''.join(canonical(r) + '\n' for r in live))
    receipt = {'passed': True, 'scope': 'full',
               'counts': {'forecasts': len(forecasts), 'live': len(live)},
               'input_sha256': {p.name: file_hash(p) for p in root.iterdir()}}
    audit = root / 'audit.json'
    audit.write_text(json.dumps(receipt))
    return audit


def test_summary_preserves_holm_family_and_descriptive_coverage(tmp_path):
    audit = make_archive(tmp_path)
    result = summarize(tmp_path, audit, replicates=100)
    primary = [r for r in result['forecast_contrasts'] if r['primary']]
    assert len(primary) == 2
    assert all(r['holm_family_size'] == 2 and r['p_value_holm'] == 1 for r in primary)
    assert result['input_coverage']['seen_test_cases'] == 1
    assert all(r['metric'] == 'goals' for r in result['live_contrasts'])
    assert all('p_value' not in r for r in result['secondary_live_contrasts'])
    assert result['audit_receipt_sha256'] == file_hash(audit)


@pytest.mark.parametrize('damage', ['changed', 'partial', 'omission', 'trace_omission', 'counts'])
def test_summary_rejects_unverified_evidence(tmp_path, damage):
    audit = make_archive(tmp_path)
    receipt = json.loads(audit.read_text())
    if damage == 'changed':
        (tmp_path / 'candidates.jsonl').write_text('changed')
    elif damage == 'partial':
        receipt['scope'] = 'collection'
    elif damage == 'omission':
        del receipt['input_sha256']['test.json']
    elif damage == 'trace_omission':
        del receipt['input_sha256']['trace.jsonl.gz']
    else:
        receipt['counts']['live'] += 1
    audit.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        summarize(tmp_path, audit, replicates=100)
