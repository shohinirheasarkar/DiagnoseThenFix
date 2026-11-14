"""These checks stop broken pairs or repeated predictions from inflating certainty."""
from copy import deepcopy

import pytest

from diagnose.targeted_analysis import forecast_contrasts, live_contrasts


def forecasts():
    return [dict(model=model, program_id=p, branch=b, state_id=s, horizon=k,
                 correct=b == 'targeted' and p > 0, valid=True)
            for model in ('small', 'large') for p in range(4)
            for b in ('original', 'uniform', 'targeted')
            for s in range(3) for k in (1, 2)]


def games():
    return [dict(model='small', program_id=p, branch=b, regime=r, env_seed=s,
                 goals=int(b == 'targeted' and p > 0 and s > 0))
            for p in range(4) for b in ('original', 'uniform', 'targeted')
            for r in ('actual', 'delay') for s in range(5)]


def test_forecasts_keep_models_separate_and_pair_programs():
    result = forecast_contrasts(forecasts(), replicates=200)
    assert len(result) == 6
    primary = [r for r in result if r['primary']]
    assert len(primary) == 2
    assert all(r['mean'] == .75 and r['programs'] == 4 for r in primary)
    assert all(r['observations'] == 24 for r in primary)


def test_repeating_test_cells_does_not_create_independent_programs():
    rows = forecasts()
    extra = [{**r, 'state_id': r['state_id'] + 100} for r in rows]
    a = forecast_contrasts(rows, replicates=200)
    b = forecast_contrasts(rows + extra, replicates=200)
    assert [(r['mean'], r['low'], r['high']) for r in a] == [
        (r['mean'], r['low'], r['high']) for r in b]


@pytest.mark.parametrize('damage', ['duplicate', 'missing_cell', 'missing_branch', 'bad_branch'])
def test_reject_incomplete_or_duplicate_pairs(damage):
    rows = forecasts()
    if damage == 'duplicate':
        rows.append(rows[0])
    elif damage == 'missing_cell':
        rows.pop()
    elif damage == 'missing_branch':
        rows = [r for r in rows if r['branch'] != 'original']
    else:
        rows[0]['branch'] = 'chosen_after_test'
    with pytest.raises(ValueError):
        forecast_contrasts(rows, replicates=100)


def test_invalid_forecasts_count_as_wrong():
    rows = forecasts()
    rows[0].update(valid=False, correct=False)
    assert forecast_contrasts(rows, replicates=100)
    rows[0]['correct'] = True
    with pytest.raises(ValueError, match='invalid'):
        forecast_contrasts(rows, replicates=100)


def test_crossed_bootstrap_and_regime_separation():
    result = live_contrasts(games(), replicates=200)
    assert len(result) == 6
    for row in result:
        assert row['programs'] == 4
        assert row['environment_seeds'] == 5
        assert row['observations'] == 20
        if row['contrast'] == 'targeted-uniform':
            assert row['mean'] == pytest.approx(.6)
            assert row['low'] < row['mean'] < row['high']


def test_all_zero_is_descriptive_not_positive_evidence():
    rows = games()
    for row in rows:
        row['goals'] = 0
    result = live_contrasts(rows, replicates=100)
    assert all(r['mean'] == r['low'] == r['high'] == 0 for r in result)
    assert all('p_value' not in r for r in result)


def test_seed_reproducibility_and_nonmutation():
    rows = games()
    before = deepcopy(rows)
    assert live_contrasts(rows, seed=3, replicates=100) == live_contrasts(
        rows, seed=3, replicates=100)
    assert rows == before


def test_nonfinite_live_metric_rejected():
    rows = games()
    rows[0]['goals'] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        live_contrasts(rows, replicates=100)


def test_expected_cohort_detects_complete_missing_program_and_model():
    rows = forecasts()
    expected = {'small': list(range(4)), 'large': list(range(4))}
    assert forecast_contrasts(rows, expected_programs=expected, replicates=100)
    for subset in ([r for r in rows if r['program_id'] != 3],
                   [r for r in rows if r['model'] != 'large']):
        with pytest.raises(ValueError, match='cohort'):
            forecast_contrasts(subset, expected_programs=expected, replicates=100)


def test_macro_board_weighting():
    rows = forecasts()
    for r in rows:
        r['board_id'] = 'one' if r['state_id'] == 0 else 'two'
        r['correct'] = r['branch'] == 'targeted' and r['state_id'] == 0
    micro = forecast_contrasts(rows, replicates=100)[0]
    macro = forecast_contrasts(rows, replicates=100, board_macro=True)[0]
    assert micro['mean'] == pytest.approx(1 / 3)
    assert macro['mean'] == .5


def test_expected_live_seeds():
    with pytest.raises(ValueError, match='seeds'):
        live_contrasts(games(), expected_seeds=range(6), replicates=100)


def test_entire_missing_forecast_cell_is_detected():
    rows = forecasts()
    with pytest.raises(ValueError, match='cells'):
        forecast_contrasts(rows, expected_cells=[(s, k) for s in range(4) for k in (1, 2)],
                           replicates=100)


def test_entire_missing_live_regime_is_detected():
    with pytest.raises(ValueError, match='regimes'):
        live_contrasts(games(), expected_regimes=['actual', 'delay', 'zero'], replicates=100)


def test_signflip_exact_known_results():
    from diagnose.targeted_analysis import paired_signflip
    assert paired_signflip([0, 0, 0])['p_value'] == 1
    assert paired_signflip([1, 1, 1, 1])['p_value'] == .125
    assert paired_signflip([1, -1])['p_value'] == 1
    assert paired_signflip([1, 1, 1, 1])['permutations'] == 16


def test_signflip_montecarlo_reproducible_and_plus_one():
    from diagnose.targeted_analysis import paired_signflip
    result = paired_signflip([1] * 21, seed=4)
    assert result == paired_signflip([1] * 21, seed=4)
    assert result['p_value'] >= 1 / 10001
    assert result['permutations'] == 10000


def test_holm_only_primary_two_model_family():
    rows = forecast_contrasts(forecasts(), replicates=100)
    for row in rows:
        if row['primary']:
            assert row['holm_family_size'] == 2
            assert row['p_value_holm'] == .5
            assert row['p_value'] == .25
        else:
            assert 'p_value' not in row
    source = forecasts()
    for row in source:
        row['board_id'] = row['state_id']
    sensitivity = forecast_contrasts(source, board_macro=True, replicates=100)
    assert all('p_value' not in row for row in sensitivity)
