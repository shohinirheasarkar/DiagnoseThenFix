"""These checks keep diary changes paired and make every history outcome replayable."""
import json

import pytest

from diagnose.context import (HISTORIES, HISTORY_REPAIRS, conditions, diary, history_text,
                              real_diary, run, score, task)
from diagnose.live import baseline, episode
from diagnose.world import State, initial, label, step


def board(seed=1000):
    state = initial(seed)
    return {'id': state.key, 'state': state.to_dict(), 'oracle': label(state), 'split': 'test'}


def response(action='U', collision=None, confidence=.5):
    return {'error': None, 'raw': {'message': {'content': json.dumps({
        'action': action, 'collision': collision or {'U': False, 'S': False, 'D': False},
        'confidence': confidence})}}}


@pytest.mark.parametrize('kind', HISTORIES[:-1])
@pytest.mark.parametrize('length', [0, 3, 6, 12])
def test_generated_outcomes_are_actual_simulator_events(kind, length):
    state = initial(1000)
    rows = diary(state, kind, length, 2)
    assert len(rows) == length
    assert rows == diary(state, kind, length, 2)
    for row in rows:
        following, event = step(State.from_dict(row['before']), row['action'])
        assert following.to_dict() == row['after']
        assert row['events'] == [event]
        assert row['before']['source_seed'] != state.source_seed


def test_real_history_uses_executed_flags_and_observed_events():
    e = episode(baseline('stay'), seed=1000, max_decisions=12)
    rows = real_diary([e])
    selected = diary(initial(1001), 'own', 6, real=rows)
    assert selected == rows[-6:]
    assert all(r['source'] == 'saved model episode' for r in selected)
    with pytest.raises(ValueError):
        diary(initial(1001), 'own', 13, real=rows)
    assert diary(initial(1001), 'own', 0, real=rows) == []


@pytest.mark.parametrize('repair', HISTORY_REPAIRS)
def test_repairs_preserve_board_and_truth(repair):
    row = board()
    base = task(row, 'D', 'coordinates', 0, length=12)
    query = task(row, 'D', 'coordinates', 0, length=12, repair=repair)
    assert query['truth'] == base['truth']
    assert query['actual_collision'] == base['actual_collision']
    assert query['prompt'].endswith(base['prompt'].split('Read the board carefully.')[1])
    if repair == 'no_history':
        assert 'Previous proposed' not in query['prompt'] and not query['shown_actions']
    if repair == 'window':
        assert query['prompt'].count('Previous proposed') == 2
    if repair == 'balanced':
        assert query['prompt'].count('Balanced synthetic') == 12
        assert 'Observed events' not in query['prompt']
        assert all(query['shown_actions'].count(a) == 4 for a in ('U', 'S', 'D'))
    if repair == 'delta':
        assert 'Previous proposed' not in query['prompt']
        assert 'Earlier episode head-column changes' in query['prompt']


def test_outcome_annotation_does_not_invent_current_board_feedback():
    rows = diary(initial(1000), 'up', 12)
    text = history_text(rows, outcomes=True)
    assert 'separate episodes' in text and 'Observed events' in text
    assert 'Oracle-optimal' not in text
    assert 'Oracle-optimal' in history_text(rows, 'outcomes')


@pytest.mark.parametrize('stage', ['D', 'E'])
def test_conditions_have_unique_prompts_except_explicit_no_history_controls(stage):
    values = list(conditions(stage, full=True))
    assert len(values) == len({json.dumps(v, sort_keys=True) for v in values})
    assert len(values) == (42 if stage == 'D' else 17)


@pytest.mark.parametrize('action', ['UP', None, 3])
def test_invalid_actions_count_as_incorrect(action):
    result = score(task(board(), 'D', 'coordinates', 0), response(action))
    assert not result['valid'] and not result['correct']


@pytest.mark.parametrize('confidence', [-1, 2, True, 'high'])
def test_invalid_confidence_is_not_a_success(confidence):
    assert not score(task(board(), 'E', 'coordinates', 0), response(confidence=confidence))['valid']


def test_label_and_skin_changes_keep_physics_but_reversal_changes_it():
    row = board()
    base = task(row, 'E', 'coordinates', 0)
    changed = task(row, 'E', 'coordinates', 0, labels='neutral', skin='clouds')
    assert changed['truth'] == base['truth']
    assert score(changed, response('A'))['action'] == 'U'
    reversed_task = task(row, 'E', 'coordinates', 0, reversal=True)
    assert 'taking priority' in reversed_task['prompt']
    assert reversed_task['actual_collision']['U'] <= base['actual_collision']['U']


def test_resume_reuses_trials_and_rejects_design_changes(tmp_path):
    class Model:
        name, digest, version = 'test', 'hash', 'v1'
        calls = 0
        def ask(self, prompt, **kwargs):
            self.calls += 1
            return response('A' if 'A means U' in prompt else
                            'FORWARD' if 'FORWARD means U' in prompt else 'U')
    model = Model()
    row = board()
    rows = run([row], model, tmp_path, 'E', 'coordinates', seeds=(0,), limit=1)
    assert len(rows) == 17 and model.calls == 17
    assert run([row], model, tmp_path, 'E', 'coordinates', seeds=(0,), limit=1) == rows
    assert model.calls == 17
    with pytest.raises(ValueError, match='Different study design'):
        run([row], model, tmp_path, 'E', 'ascii', seeds=(0,), limit=1)


def test_deadline_stops_before_a_model_call(tmp_path):
    class Forbidden:
        name, digest, version = 'test', 'hash', 'v1'
        def ask(self, *args, **kwargs):
            raise AssertionError('expired run')
    assert run([board()], Forbidden(), tmp_path, 'E', 'coordinates', stop_at=0) == []


def test_paired_summary_excludes_invalid_flips_but_counts_invalid_accuracy():
    from diagnose.context import summarize
    row = board()
    records = []
    for kind, length, action in [('stay', 0, 'U'), ('stay', 12, 'S'), ('up', 12, 'bad')]:
        query = task(row, 'D', 'coordinates', 0, kind=kind, length=length)
        raw = response(action)
        records.append({'model': 'test', 'state_id': row['id'], 'source_seed': 1000,
                        'task': query, 'response': raw, 'score': score(query, raw)})
    output = summarize(records, replicates=50)
    invalid = next(r for r in output if r['condition']['kind'] == 'up')
    changed = next(r for r in output if r['condition']['length'] == 12 and r['condition']['kind'] == 'stay')
    assert invalid['paired_valid'] == 0 and invalid['paired_all'] == 1
    assert invalid['action_change']['mean'] is None and invalid['accuracy']['mean'] == 0
    assert changed['action_change']['mean'] == 1
    assert changed['p_holm'] >= changed['p_sign_flip']


def test_cluster_sign_flip_is_exact_for_small_samples():
    from diagnose.context import sign_test
    assert sign_test([0, 0]) == 1
    assert sign_test([1, 1, 1]) == .25
    assert sign_test([1, -1]) == 1
    assert sign_test([1]*20, draws=100) == 1/101


def test_evidence_verifier_rejects_changed_truth_score_or_diary():
    import copy
    from diagnose.context import verify
    row = board()
    query = task(row, 'D', 'coordinates', 0, length=3)
    raw = response()
    record = json.loads(json.dumps({'task': query, 'response': raw, 'score': score(query, raw)}))
    assert verify(record, row)
    for field in ('truth', 'score', 'diary'):
        bad = copy.deepcopy(record)
        if field == 'truth':
            bad['task']['truth']['safe'] = []
        elif field == 'score':
            bad['score']['correct'] = not bad['score']['correct']
        else:
            bad['task']['diary'][0]['events'] = ['goal']
        with pytest.raises(ValueError):
            verify(bad, row)
