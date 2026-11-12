"""These tests compare our replay physics against the original game and stress the oracle."""
from dataclasses import replace
from itertools import product
import random

import pytest

from diagnose.render import FORMATS, render
from diagnose.world import ACTIONS, Car, State, futures, initial, label, oracle, step


@pytest.mark.parametrize('seed', range(20))
def test_differential_upstream(seed):
    from realtimegym.environments.freeway import FreewayEnv
    env = FreewayEnv()
    env.set_seed(seed + 1000)
    env.reset()
    s = initial(seed + 1000)
    rng = random.Random(seed)
    for _ in range(100):
        a = rng.choice(ACTIONS)
        _, done, _, collision = env.step(a)
        s, event = step(s, a)
        assert (event == 'collision') == bool(collision)
        if event != 'goal':
            assert s.lane == env.pos
            assert s.cars == tuple(Car(*c) for c in env.cars if c[3] is not None)
        assert s.turn == env.game_turn
        if done:
            assert event in ('goal', 'timeout', 'collision')
            break


@pytest.mark.parametrize('p', [1, -1, 2, -2, 4, -4, .25, -.25, .5, -.5])
@pytest.mark.parametrize('x', [-4, -1, 0, 4, 8, 9, 12])
def test_timer_and_wrap(p, x):
    c = Car(x, 1, 0, p, 2)
    nxt = c.tick()
    expected = 8 if x < 0 else 0 if x > 8 else x + (int(1/p) if abs(p)<1 else (1 if p>0 else -1))
    assert nxt.x == expected
    assert c.x == x


def test_goal_before_car_update():
    s = State((Car(3, 1, 0, 1, 1),), lane=1)
    nxt, event = step(s, 'U')
    assert event == 'goal' and nxt.cars == s.cars


def test_long_car_occupancy_and_collision_reset():
    s = replace(initial(1000), cars=(Car(5, 8, 1, 2, 3),))
    assert s.cars[0].cells() == (5, 4, 3)
    nxt, event = step(s, 'U')
    assert event == 'collision' and nxt.lane == 9 and nxt.turn == 1
    assert nxt.cars == initial(1000).cars


@pytest.mark.parametrize('seed', range(10))
def test_oracle_against_exhaustive_sequences(seed):
    s = replace(initial(seed + 1000), lane=5)
    safe = set()
    for seq in product(ACTIONS, repeat=4):
        current = s
        for a in seq:
            current, event = step(current, a)
            if event in ('collision', 'timeout'):
                break
            if event == 'goal':
                safe.add(seq[0])
                break
        else:
            safe.add(seq[0])
    assert set(oracle(s, 4)['safe']) == safe


def test_dead_end_is_not_labeled_safe():
    cars = tuple(Car(4, lane, 1, 2, 1) for lane in (4, 5, 6))
    s = State(cars, lane=5)
    truth = label(s, 1)
    assert truth['safe'] == [] and truth['choice'] is None
    assert truth['required_foresight'] is None


def test_roundtrip_and_no_aliasing():
    s = initial(1000)
    data = s.to_dict()
    assert State.from_dict(data) == s
    assert State.from_dict(data).key == s.key
    before = s.to_dict()
    futures(s)
    oracle(s, 4)
    assert s.to_dict() == before


@pytest.mark.parametrize('representation', FORMATS)
def test_render_is_factual_and_does_not_mutate(representation):
    s = initial(1000)
    before = s.to_dict()
    text = render(s, representation)
    assert text and ('timer' in text)
    assert s.to_dict() == before
    assert 'time_to_collision' not in text


@pytest.mark.parametrize('action', ['X', '', None])
def test_bad_actions_fail_loudly(action):
    with pytest.raises(ValueError):
        step(State(()), action)


def test_terminal_and_bad_horizon_rejected():
    with pytest.raises(ValueError):
        step(State((), lane=0), 'S')
    with pytest.raises(ValueError):
        oracle(State(()), 0)


def test_bank_is_reproducible_unique_and_split_by_episode():
    from diagnose.world import build_bank
    rows, report = build_bank(4, max_episodes=2, horizon=2)
    again, report2 = build_bank(4, max_episodes=2, horizon=2)
    assert rows == again and report == report2
    assert len(rows) == len({row['id'] for row in rows})
    splits = {}
    for row in rows:
        source = row['state']['source_seed']
        splits.setdefault(source, set()).add(row['split'])
        assert row['oracle'] == label(State.from_dict(row['state']), 2)
    assert all(len(v) == 1 for v in splits.values())
    assert report['actual'] <= report['requested']


def test_renderer_rejects_unknown_conditions():
    with pytest.raises(ValueError):
        render(State(()), 'picture')
    with pytest.raises(ValueError):
        render(State(()), skin='lava')


def test_validation_and_terminal_oracle():
    with pytest.raises(ValueError):
        State((), lane=10)
    with pytest.raises(ValueError):
        State((Car(0, 1, 0, 0, 1),))
    with pytest.raises(ValueError):
        oracle(State((), lane=0))
    from diagnose.world import build_bank
    with pytest.raises(ValueError):
        build_bank(0)


def test_bank_reserves_each_split_before_filling_categories():
    from diagnose.world import build_bank
    rows, report = build_bank(10, max_episodes=10, horizon=1)
    assert report['actual'] == 10
    assert {r['split'] for r in rows} == {'train', 'validation', 'test'}
    assert report['splits'] == {'train': 6, 'validation': 2, 'test': 2}


@pytest.mark.parametrize('turn', [98, 99])
@pytest.mark.parametrize('action,lane', [('U', 9), ('S', 8), ('D', 7)])
@pytest.mark.parametrize('x,p,length', [(3, 1, 1), (5, -1, 1), (0, .25, 4), (8, -.25, 4)])
def test_collision_reset_matches_upstream_at_terminal_boundary(turn, action, lane, x, p, length):
    from realtimegym.environments.freeway import FreewayEnv
    env = FreewayEnv()
    env.set_seed(1000)
    env.reset()
    car = Car(x, 8, 0, p, length)
    env.cars = [[car.x, car.lane, car.timer, car.speed, car.length]]
    env.pos, env.game_turn, env.reward = lane, turn, 100-turn
    current = State((car,), lane=lane, turn=turn, source_seed=1000)
    _, terminal, _, collision = env.step(action)
    following, event = step(current, action)
    assert collision and event == 'collision'
    assert terminal == (turn == 99)
    assert following.lane == env.pos == 9
    assert following.turn == env.game_turn == turn+1
    assert following.cars == tuple(Car(*c) for c in env.cars if c[3] is not None)
    if terminal:
        assert following.cars == (car.tick(),)
    else:
        assert following.cars == initial(1000).cars
