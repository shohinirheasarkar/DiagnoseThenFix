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


def test_descriptive_shapes_recover_known_curves():
    import math
    import pytest
    from paper.build import shape_fits
    assert shape_fits({1: .5, 2: .4}) is None
    line = shape_fits({k: 1-.05*k for k in range(1, 11)})
    assert line['linear']['rmse'] < 1e-12
    assert line['linear']['slope'] == pytest.approx(-.05)
    exponential = shape_fits({k: .9*math.exp(-.2*k) for k in range(1, 11)})
    assert exponential['exponential']['rmse'] < 1e-12
    assert exponential['exponential']['rate'] == pytest.approx(-.2)
    cliff = shape_fits({k: .9 if k <= 4 else .2 for k in range(1, 11)})
    assert cliff['cliff']['rmse'] < 1e-12
    assert cliff['cliff']['last_before'] == 4


def test_paired_live_estimates_exclude_unshared_and_censored_episodes(tmp_path):
    root = tmp_path / 'results'
    folder = root / 'model'
    folder.mkdir(parents=True)
    (folder / 'profile.json').write_text(json.dumps({'model': 'test-model', 'best_representation': 'coordinates'}))
    def row(seed, success, tokens, censored=False):
        return {'seed': seed, 'success': success, 'tokens': tokens, 'censored': censored,
                'calls': 1, 'collisions': 0, 'trace': []}
    base = [row(1, False, 10), row(2, True, 20), row(3, False, 10)]
    selected = [row(1, True, 15), row(2, False, 50, True), row(4, True, 100)]
    base[0]['trace'] = [{'response': {'error': 'program failed'}}]
    base[1]['trace'] = [{'response': {'error': None}}]
    selected[1]['trace'] = [{'response': {'error': 'censored failure'}}]
    for name, records in (('base', base), ('selected', selected)):
        (folder / f'live-{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    result = analyze(root, Path('data/freeway.json'))
    paired = result['live_paired'][0]
    assert paired['paired_episodes'] == 1
    assert paired['success_change']['mean'] == 1
    assert paired['token_change']['mean'] == 5
    assert next(r for r in result['live'] if r['condition'] == 'selected')['censored'] == 1
    assert next(r for r in result['live'] if r['condition'] == 'selected')['error_decisions'] == 0
    baseline = next(r for r in result['live'] if r['condition'] == 'base')
    assert baseline['error_decisions'] == 1 and baseline['decisions'] == 2
    assert baseline['successes'] == 1
    assert next(r for r in result['live'] if r['condition'] == 'selected')['successes'] == 2
    assert next(r for r in result['live'] if r['condition'] == 'selected')['episode_seeds'] == [1, 4]


def test_error_onset_uses_delayed_wrap_and_keeps_partial_sequences_separate():
    from paper.build import first_error_events, motion_events
    s = State((Car(8, 1, 0, 1, 1),))
    assert motion_events(s, 3) == ['timer_move', 'wrap', 'timer_move']
    fast = State((Car(8, 1, 0, .25, 4),))
    assert motion_events(fast, 2) == ['fast_step', 'wrap']
    waiting = State((Car(0, 1, 2, 3, 1),))
    assert motion_events(waiting, 3) == ['timer_wait', 'timer_wait', 'timer_move']
    bank = {'s': {'state': s.to_dict()}}
    def trial(seed, k, correct):
        return {'state_id': 's', 'task': {'kind': 'position', 'seed': seed, 'k': k},
                'score': {'correct': correct}}
    records = [trial(0, k, k != 2) for k in range(1, 11)]
    records += [trial(1, k, True) for k in range(1, 11)]
    records += [trial(2, 1, False)]
    result = first_error_events(records, bank)
    assert result['complete_sequences'] == 2 and result['incomplete_sequences'] == 1
    assert result['all_correct'] == 1
    assert result['first_error_event']['wrap'] == 1
    assert result['first_error_after_any_wrap'] == 1


def test_followup_keeps_forecast_cohorts_and_invalid_denominators_separate(tmp_path):
    from paper.build import analyze_followup
    def record(repair, correct, valid, key):
        return {'model': 'qwen3:1.7b', 'state_id': 's', 'source_seed': 10,
                'task': {'stage': 'B', 'kind': 'position', 'repair': repair, 'seed': 0, 'k': 1},
                'response': {'key': key}, 'score': {'correct': correct, 'valid': valid}}
    for name, values in [('grammar', [record('base', False, True, 'a'),
                                      record('code_v2', True, True, 'b'),
                                      {**record('code_v2', False, False, 'c'),
                                       'state_id': 'other', 'source_seed': 11}]),
                          ('B-chained-calls', [record('base', True, True, 'd'),
                                               record('chained_calls', False, True, 'e')])]:
        folder = tmp_path/name
        folder.mkdir()
        (folder/'responses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in values))
    evidence = analyze_followup(tmp_path)
    code = next(r for r in evidence['forecasts'] if r['repair'] == 'code_v2')
    chain = next(r for r in evidence['forecasts'] if r['repair'] == 'chained_calls')
    assert code['trials'] == 2 and code['invalid'] == 1 and code['accuracy']['mean'] == .5
    assert code['paired_trials'] == 1 and code['difference_from_base']['mean'] == 1
    assert chain['difference_from_base']['mean'] == -1
    assert code['unique_requests'] == 2
    assert len(evidence['input_sha256']) == 2


def test_followup_never_pools_v1_or_duplicate_completed_seeds(tmp_path):
    import pytest
    from paper.build import analyze_followup
    folder = tmp_path/'model'
    folder.mkdir()
    (folder/'profile.json').write_text('{"model":"test-model"}')
    record = {'seed': 60000, 'success': False, 'censored': False, 'trace': [],
              'tokens': 0, 'calls': 0}
    path = folder/'live-base.jsonl'
    path.write_text(json.dumps(record)+'\n')
    with pytest.raises(ValueError, match='never pool'):
        analyze_followup(tmp_path)
    record['policy_revision'] = 'v2'
    path.write_text((json.dumps(record)+'\n')*2)
    with pytest.raises(ValueError, match='Duplicate'):
        analyze_followup(tmp_path)


def test_followup_pairs_only_shared_completed_live_runs(tmp_path):
    from paper.build import analyze_followup
    folder = tmp_path/'model'
    folder.mkdir()
    (folder/'profile.json').write_text('{"model":"test-model"}')
    def row(seed, success=False, censored=False):
        return {'seed': seed, 'success': success, 'censored': censored, 'trace': [],
                'tokens': 10, 'calls': 1, 'policy_revision': 'v2'}
    for condition, records in [('base', [row(60000), row(60001)]),
                               ('selected', [row(60000, True), row(60001, censored=True)])]:
        (folder/f'live-{condition}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    evidence = analyze_followup(tmp_path)
    paired = evidence['live_paired'][0]
    assert paired['paired_episodes'] == 1 and paired['success_change']['mean'] == 1
    selected = next(r for r in evidence['live'] if r['condition'] == 'selected')
    assert selected['episodes'] == selected['censored'] == 1


def test_followup_report_marks_empty_cells_without_rewriting_first_cycle(tmp_path):
    from paper.build import analyze_followup
    root = tmp_path/'empty'
    root.mkdir()
    evidence = analyze(root, Path('data/freeway.json'))
    evidence['followup'] = analyze_followup(root)
    output = tmp_path/'followup.pdf'
    build(evidence, output, [], root)
    text = '\n'.join(p.extract_text() for p in PdfReader(output).pages)
    assert '0 held-out forecast trials' in text
    assert 'separate overnight follow-up' in text
    assert '0 held-out diary/framing trials' in text
    assert 'No v2 live episode is included yet' in text
    assert 'never pooled with the first cycle' in text


def test_followup_context_table_uses_test_boards_only(tmp_path):
    from paper.build import analyze_followup
    folder = tmp_path/'D-test'
    folder.mkdir()
    def record(length, split, action, safe):
        return {'model': 'test-model', 'state_id': 's', 'source_seed': 10, 'split': split,
                'task': {'stage': 'D', 'kind': 'stay', 'length': length, 'repair': 'base',
                         'outcomes': False, 'seed': 0},
                'score': {'valid': True, 'correct': safe, 'safe': safe, 'action': action,
                          'collision': {'U': False, 'S': False, 'D': True},
                          'history_match': action == 'S' if length else None}}
    values = [record(0, 'test', 'U', True), record(12, 'test', 'S', False),
              record(12, 'validation', 'U', True)]
    (folder/'responses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in values))
    result = analyze_followup(tmp_path)
    assert result['context_trials'] == 2
    changed = next(r for r in result['context'] if r['condition']['length'] == 12)
    assert changed['harmful_flip']['mean'] == 1 and changed['accuracy']['mean'] == 0
    (tmp_path/'no-first-cycle').mkdir()
    evidence = analyze(tmp_path/'no-first-cycle', Path('data/freeway.json'))
    evidence['followup'] = result
    output = tmp_path/'context.pdf'
    build(evidence, output, [], tmp_path)
    text = '\n'.join(p.extract_text() for p in PdfReader(output).pages)
    assert '2 held-out diary/framing trials' in text and 'Unsafe flip' in text
    assert '100.0%' in text


def test_crossed_interval_keeps_shared_environment_dependence():
    from paper.build import crossed_interval
    result = crossed_interval({p: {10: 0, 11: 1} for p in range(5)})
    assert result['mean'] == .5 and result['low'] == 0 and result['high'] == 1
    assert crossed_interval({p: {10: 1} for p in range(4)})['mean'] is None
    assert crossed_interval({p: {10: 1} for p in range(1, 6)})['mean'] is None


def test_compiled_analysis_separates_versions_and_rejects_changed_sources(tmp_path):
    from copy import deepcopy
    import pytest
    from paper.build import analyze_compiled
    from test_compiled import generation
    folder = tmp_path/'compiled'
    folder.mkdir()
    records = []
    for version in (1, 2):
        for p in range(5):
            records.append({'protocol': f'compiled-controller-v{version}', 'model': 'test-model',
                            'program_seed': p, 'condition': 'amortized', 'validation': 'executable',
                            'generation_response': generation(p), 'seed': 60000, 'success': version == 2,
                            'censored': False, 'tokens': 42, 'calls': 1, 'collisions': 0,
                            'longest_stay_loop': 0, 'trace': []})
    path = folder/'episodes.jsonl'
    def save():
        path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    save()
    result = analyze_compiled(tmp_path)
    assert result['source_programs'] == 5 and result['complete'] == 10
    assert [g['success']['mean'] for g in result['groups']] == [0, 1]
    duplicate = deepcopy(records[0])
    records.append(duplicate)
    save()
    with pytest.raises(ValueError, match='Duplicate'):
        analyze_compiled(tmp_path)
    duplicate['seed'] += 1
    duplicate['generation_response']['key'] = 'changed'
    save()
    with pytest.raises(ValueError, match='source program changed'):
        analyze_compiled(tmp_path)
    records.pop()
    records[0]['trace'] = [{'response': {'fresh_requests': 1}}]
    save()
    with pytest.raises(ValueError, match='fresh requests'):
        analyze_compiled(tmp_path)


def test_compiled_paper_table_has_protocol_and_assistance_caveats(tmp_path):
    from paper.build import analyze_followup
    evidence = analyze(tmp_path, Path('data/freeway.json'))
    evidence['followup'] = analyze_followup(tmp_path)
    evidence['followup']['compiled'] = {'source_programs': 5, 'complete': 250, 'censored': 0,
        'groups': [{'protocol': 'compiled-controller-v2', 'model': 'test-model',
                    'condition': 'amortized', 'validation': 'executable', 'successes': 3,
                    'episodes': 250, 'success': {'mean': .012, 'low': 0, 'high': .04}}]}
    output = tmp_path/'compiled.pdf'
    build(evidence, output, [], tmp_path)
    text = '\n'.join(p.extract_text() for p in PdfReader(output).pages)
    assert 'Saved programs and delay' in text and '3/250' in text
    assert 'synthetic' in text and 'crossed axes' in text
    assert 'never pooled' in text and 'five generated programs' in text


def test_heldout_shapes_never_fit_the_test_episode():
    from paper.build import heldout_shape_fits
    records = [{'source_seed': episode, 'task': {'kind': 'position', 'k': k},
                'score': {'correct': episode != 2}}
               for episode in range(3) for k in range(1, 5)]
    result = heldout_shape_fits(records)
    heldout = next(f for f in result['folds'] if f['source_episode'] == 2)
    assert heldout['training_episodes'] == [0, 1]
    assert heldout['scores']['flat']['predictions'] == dict.fromkeys(range(1, 5), 1.)
    assert heldout['scores']['flat']['curve_mse'] == 1
    for fold in result['folds']:
        assert fold['source_episode'] not in fold['training_episodes']
        for score in fold['scores'].values():
            assert all(0 <= value <= 1 for value in score['predictions'].values())
    assert set(result['contrasts_vs_flat']) == {'linear', 'exponential', 'cliff'}
    assert heldout_shape_fits(records[:4])['summary'] == {}


def test_heldout_shapes_reject_missing_horizons_without_filling_them():
    from paper.build import heldout_shape_fits
    records = [{'source_seed': episode, 'task': {'kind': 'position', 'k': k},
                'score': {'correct': k < 3}}
               for episode in range(3) for k in range(1, 5) if not (episode == 2 and k == 4)]
    result = heldout_shape_fits(records)
    assert {f['source_episode'] for f in result['folds']} == {0, 1}
    for kind, score in result['summary'].items():
        assert score['curve_mse']['mean'] >= 0 and kind in ('flat', 'linear', 'exponential', 'cliff')


def test_signed_error_plot_uses_valid_numeric_replies_and_keeps_horizon_gaps(tmp_path):
    from paper.build import signed_error_figure
    bank = json.loads(Path('data/freeway.json').read_text())
    situation = next(r for r in bank['situations'] if r['split'] == 'test')
    folder = tmp_path/'model'
    phase = folder/'B-test'
    phase.mkdir(parents=True)
    (folder/'profile.json').write_text(json.dumps({'model': 'test-model'}))
    common = {'state_id': situation['id'], 'source_seed': situation['state']['source_seed'],
              'task': {'kind': 'position', 'k': 1, 'seed': 0, 'truth': 4}}
    trials = [{**common, 'score': {'valid': True, 'correct': False, 'confidence': 1, 'signed_error': 2}},
              {**common, 'score': {'valid': False, 'correct': False, 'error': 'truncated'}}]
    (phase/'responses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in trials))
    evidence = analyze(tmp_path, Path('data/freeway.json'))
    curve = evidence['models']['test-model']['position_signed_error_curve']
    assert curve[0]['error']['mean'] == 2 and curve[0]['error']['observations'] == 1
    assert curve[1]['error']['mean'] is None
    output = tmp_path/'signed.png'
    assert signed_error_figure(evidence, output) == output and output.stat().st_size > 1000
    assert signed_error_figure({'models': {}}, output) is None


def test_paper_keeps_each_plot_with_its_caption(tmp_path):
    import matplotlib.pyplot as plt
    figure, ax = plt.subplots(figsize=(7, 4))
    ax.plot([0, 1], [0, 1])
    png = tmp_path/'plot.png'
    figure.savefig(png)
    plt.close(figure)
    evidence = analyze(tmp_path, Path('data/freeway.json'))
    output = tmp_path/'captions.pdf'
    build(evidence, output, [png, png], tmp_path)
    pages = PdfReader(output).pages
    for caption in ('Figure 1.', 'Figure 2.'):
        page = next(p for p in pages if caption in p.extract_text())
        assert len(page.images) > 0


def test_related_work_marks_existing_code_policies_and_conversion_limits(tmp_path):
    from paper.build import REFERENCES
    assert len({url for _, url in REFERENCES}) == len(REFERENCES)
    evidence = analyze(tmp_path, Path('data/freeway.json'))
    output = tmp_path/'related.pdf'
    build(evidence, output, [], tmp_path)
    text = '\n'.join(p.extract_text() for p in PdfReader(output).pages)
    for phrase in ('already evaluates code-as-policy', 'AutoHarness', 'XiangqiBench',
                   'Seeing Without Understanding', 'not an exhaustive priority search'):
        assert phrase in text


def test_live_details_distinguish_program_availability_from_execution_and_missing_data():
    from paper.build import live_details
    def frame(repairs, validated, seconds):
        return {'response': {'repairs': repairs, 'program_validated': validated, 'seconds': seconds}}
    records = [{'trace': [frame(['code'], True, 1), frame(['code'], False, 3), frame([], True, 2)],
                'collisions': 2, 'final_lane': 8, 'longest_stay_loop': 4},
               {'trace': [frame(['code'], False, 2)], 'collisions': 0, 'final_lane': 2,
                'longest_stay_loop': 2}]
    result = live_details(records)
    assert result['code_enabled_decisions'] == 3 and result['available_program_decisions'] == 1
    assert result['mean_decision_seconds'] == 2 and result['mean_final_lane'] == 5
    assert result['mean_collisions'] == 1 and result['mean_stay_loop'] == 3
    assert live_details([{'trace': []}])['mean_collisions'] is None
    assert live_details([])['mean_decision_seconds'] is None


def test_followup_pdf_explains_unavailable_code_repairs(tmp_path):
    from paper.build import analyze_followup
    folder = tmp_path/'model'
    folder.mkdir()
    (folder/'profile.json').write_text('{"model":"test-model"}')
    record = {'policy_revision': 'v2', 'seed': 60000, 'success': False, 'censored': False,
              'calls': 1, 'tokens': 10, 'collisions': 0, 'final_lane': 9, 'longest_stay_loop': 1,
              'trace': [{'response': {'repairs': ['code'], 'program_validated': False,
                                       'seconds': 3, 'error': 'training failure'}}]}
    (folder/'live-selected.jsonl').write_text(json.dumps(record)+'\n')
    (tmp_path/'empty').mkdir()
    evidence = analyze(tmp_path/'empty', Path('data/freeway.json'))
    evidence['followup'] = analyze_followup(tmp_path)
    output = tmp_path/'availability.pdf'
    build(evidence, output, [], tmp_path)
    text = ' '.join(' '.join(p.extract_text() for p in PdfReader(output).pages).split())
    assert 'Code available / enabled' in text and '0/1' in text
    assert 'cannot test transfer of a successful code repair' in text
    assert 'excludes implicit waiting' in text
