"""These checks keep paper counts tied to saved trials and missing studies visibly missing."""
import json
from pathlib import Path

from pypdf import PdfReader

from paper.build import analyze, build, linear_answer
from diagnose.world import Car, State


def test_linear_baseline_is_declared_modulo_extrapolation():
    s = State((Car(8, 1, 0, 1, 1),))
    bank = {'x': {'state': s.to_dict()}}
    record = {'state_id': 'x', 'task': {'k': 1, 'kind': 'position'}}
    assert linear_answer(record, bank) == 0
    assert s.cars[0].tick().x == 9


def test_empty_run_builds_an_honest_review_artifact(tmp_path):
    root = tmp_path / 'results'
    root.mkdir()
    evidence = analyze(root, Path('data/freeway.json'))
    assert not evidence['models'] and not evidence['live']
    output = tmp_path / 'review.pdf'
    build(evidence, output, [], root)
    text = '\n'.join(p.extract_text() for p in PdfReader(output).pages)
    assert '0 held-out forecast trials' in text
    assert 'No live episode completed' in text
    assert 'main four-model forecast target is incomplete' in text
    assert 'complete research plan is not finished' in text


def test_analysis_counts_invalid_replies_as_incorrect(tmp_path):
    root = tmp_path / 'results'
    phase = root / 'qwen' / 'B-test'
    phase.mkdir(parents=True)
    bank = json.loads(Path('data/freeway.json').read_text())
    situation = next(r for r in bank['situations'] if r['split'] == 'test')
    profile = {'model': 'test-model', 'best_representation': 'coordinates'}
    (phase.parent / 'profile.json').write_text(json.dumps(profile))
    common = {'state_id': situation['id'], 'source_seed': situation['state']['source_seed'],
              'task': {'kind': 'position', 'k': 1, 'seed': 0, 'truth': situation['oracle']['future_x'][0][0]}}
    trials = [{**common, 'score': {'valid': True, 'correct': True, 'confidence': 1}},
              {**common, 'score': {'valid': False, 'correct': False, 'error': 'truncated'}}]
    (phase / 'responses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in trials))
    evidence = analyze(root, Path('data/freeway.json'))
    model = evidence['models']['test-model']
    assert model['trials'] == 2 and model['invalid'] == 1
    assert model['position_accuracy']['mean'] == .5
    assert model['source_episodes'] == 1


def test_live_boundary_intervals_do_not_claim_population_certainty():
    from paper.build import wilson
    assert wilson(0, 50)['high'] > .07
    assert wilson(50, 50)['low'] < .93
    assert wilson(25, 50)['low'] < .5 < wilson(25, 50)['high']
    assert wilson(0, 0)['mean'] is None
