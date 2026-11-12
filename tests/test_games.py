"""These tests restore original Snake and Freeway games from plain JSON and check replay identity."""
from dataclasses import replace
import random

import numpy as np
import pytest

from diagnose.games import Snapshot, advance, search, snapshot


@pytest.mark.parametrize('game', ['Freeway-v0', 'Snake-v0'])
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_snapshot_roundtrip_and_original_game_replay(game, seed):
    saved = snapshot(game, seed)
    assert Snapshot.from_dict(saved.to_dict()) == saved
    assert saved == snapshot(game, seed)
    next_state, event = advance(saved, 'S')
    assert next_state == snapshot(game, seed, ['S'])
    assert next_state.actions == ('S',)
    assert isinstance(event['collision'], bool)


def test_replay_does_not_change_global_randomness():
    random.seed(1)
    np.random.seed(1)
    expected_python, expected_numpy = random.getstate(), np.random.get_state()
    snapshot('Snake-v0')
    assert random.getstate() == expected_python
    current = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(expected_numpy, current))


def test_replay_rejects_corruption_and_bad_actions():
    saved = snapshot('Snake-v0')
    with pytest.raises(ValueError, match='Replay mismatch'):
        advance(replace(saved, observation='{}'), 'S')
    with pytest.raises(ValueError):
        advance(saved, 'Z')
    with pytest.raises(ValueError):
        snapshot('Chess-v0')


def test_search_budget_never_gets_reported_as_an_oracle():
    saved = snapshot('Snake-v0')
    bounded = search(saved, 3, node_budget=0)
    assert bounded['safe'] is None and not bounded['exhaustive']
    result = search(saved, 1)
    assert result['exhaustive'] and result['safe']
    assert set(result['optimal']) <= set(result['safe'])


def test_overcooked_restore_includes_scripted_partner_and_rng():
    pytest.importorskip('ipywidgets')
    saved = snapshot('Overcooked-v0', 0)
    for action in ('U', 'R', 'I', 'S', 'L', 'D'):
        saved, event = advance(saved, action)
        assert saved == snapshot(saved.game, saved.seed, saved.actions)
        assert not event['collision']
    assert Snapshot.from_dict(saved.to_dict()) == saved
