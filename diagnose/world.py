"""This keeps a replayable copy of a Freeway board and asks the real game what happens next."""
from collections import Counter
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
import json
import random

UPSTREAM = '3d5b3ef37dafee25ca2e8d2e8d71f5fe842eda7b'
ACTIONS = ('U', 'S', 'D')


@dataclass(frozen=True)
class Car:
    x: int
    lane: int
    timer: float
    speed: float
    length: int

    def cells(self):
        direction = -1 if self.speed > 0 else 1
        return tuple(self.x + offset * direction for offset in range(self.length))

    def tick(self):
        if self.x < 0:
            return replace(self, x=8)
        if self.x > 8:
            return replace(self, x=0)
        if abs(self.speed) < 1:
            return replace(self, x=self.x + int(1 / self.speed))
        timer = self.timer - 1
        if timer == -1:
            return replace(self, timer=timer + abs(self.speed),
                           x=self.x + (1 if self.speed > 0 else -1))
        return replace(self, timer=timer)


@dataclass(frozen=True)
class State:
    cars: tuple[Car, ...]
    lane: int = 9
    turn: int = 0
    source_seed: int = 0

    def __post_init__(self):
        if not 0 <= self.lane <= 9 or not 0 <= self.turn <= 100:
            raise ValueError('Invalid player lane or game turn')
        for c in self.cars:
            if (c.speed == 0 or not 1 <= c.lane <= 8 or c.length < 1
                    or c.timer < 0 or c.timer >= abs(c.speed) and abs(c.speed) >= 1):
                raise ValueError('Invalid car')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(cars=tuple(Car(**c) for c in data['cars']), lane=data['lane'],
                   turn=data['turn'], source_seed=data['source_seed'])

    @property
    def key(self):
        data = {'cars': [asdict(c) for c in self.cars], 'lane': self.lane, 'turn': self.turn}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:20]


def initial(seed):
    from realtimegym.environments.freeway import FreewayEnv
    env = FreewayEnv()
    env.set_seed(seed)
    env.reset()
    return State(tuple(Car(*c) for c in env.cars if c[3] is not None), source_seed=seed)


def step(state, action):
    if action not in ACTIONS:
        raise ValueError(f'Unknown action: {action}')
    if state.lane == 0 or state.turn == 100:
        raise ValueError('Cannot step a terminal state')
    lane = max(0, min(9, state.lane + {'U': -1, 'S': 0, 'D': 1}[action]))
    if lane == 0:
        return replace(state, lane=0, turn=state.turn + 1), 'goal'
    cars = tuple(c.tick() for c in state.cars)
    hit = any(c.lane == lane and 4 in c.cells() for c in cars)
    next_state = replace(state, cars=cars, lane=lane, turn=state.turn + 1)
    if hit:
        if next_state.turn == 100:
            return replace(next_state, lane=9), 'collision'
        reset = initial(state.source_seed)
        return replace(reset, turn=next_state.turn), 'collision'
    return next_state, 'timeout' if next_state.turn == 100 else 'neutral'


def futures(state, horizon=10):
    cars = state.cars
    rows = []
    for _ in range(horizon):
        cars = tuple(c.tick() for c in cars)
        rows.append([c.x for c in cars])
    return rows


def repeated_outcome(state, action, horizon):
    current = state
    for _ in range(horizon):
        current, outcome = step(current, action)
        if outcome != 'neutral':
            return outcome
    return 'progress' if current.lane < state.lane else 'neutral'


def oracle(state, horizon=10):
    if horizon < 1:
        raise ValueError('Horizon must be positive')
    if state.lane == 0 or state.turn == 100:
        raise ValueError('Oracle needs a nonterminal state')

    @lru_cache(None)
    def value(s, remaining):
        if s.lane == 0:
            return (1, 1, 0)
        if remaining == 0:
            return (1, 0, -s.lane)
        options = []
        for a in ACTIONS:
            nxt, event = step(s, a)
            if event == 'collision' or event == 'timeout':
                options.append((0, 0, -10))
            elif event == 'goal':
                options.append((1, 1, 0))
            else:
                options.append(value(nxt, remaining - 1))
        return max(options)

    scores = {}
    for a in ACTIONS:
        nxt, event = step(state, a)
        scores[a] = ((0, 0, -10) if event in ('collision', 'timeout') else
                     (1, 1, 0) if event == 'goal' else value(nxt, horizon - 1))
    safe = [a for a in ACTIONS if scores[a][0]]
    best = max(scores.values())
    optimal = [a for a in ACTIONS if scores[a] == best] if safe else []
    return {'safe': safe, 'optimal': optimal, 'choice': optimal[0] if optimal else None,
            'scores': {a: list(v) for a, v in scores.items()}, 'horizon': horizon}


def label(state, horizon=10):
    ref = oracle(state, horizon)
    required = None
    if ref['safe']:
        for depth in range(1, horizon + 1):
            if oracle(state, depth)['choice'] in ref['safe']:
                required = depth
                break
    immediate = {a: step(state, a)[1] for a in ACTIONS}
    category = ('must_retreat' if ref['safe'] == ['D'] else
                'dangerous_advance' if 'U' not in ref['safe'] else 'safe_advance')
    if 'S' in ref['safe'] and immediate['U'] == 'collision' and immediate['S'] == 'neutral':
        waited, _ = step(state, 'S')
        if step(waited, 'U')[1] != 'collision':
            category = 'waiting_helps'
    return {**ref, 'required_foresight': required, 'future_x': futures(state, horizon),
            'repeated_outcomes': {a: repeated_outcome(state, a, horizon) for a in ACTIONS},
            'category': category}


def build_bank(count=500, seed=0, horizon=10, max_episodes=3000, max_per_episode=2):
    if count < 1 or max_per_episode < 1:
        raise ValueError('Count must be positive')
    rng = random.Random(seed)
    categories = ('safe_advance', 'dangerous_advance', 'waiting_helps', 'must_retreat')
    sizes = {'train': int(count * .6), 'validation': int(count * .2)}
    sizes['test'] = count - sum(sizes.values())
    targets = {(split, category): size // 4 + (i < size % 4)
               for split, size in sizes.items() for i, category in enumerate(categories)}
    buckets = {key: [] for key in targets}
    seen = set()
    for episode in range(max_episodes):
        source = seed * 100000 + episode + 1000
        s = initial(source)
        collected = 0
        for _ in range(70):
            if s.key not in seen:
                seen.add(s.key)
                truth = label(s, horizon)
                split = 'train' if episode % 10 < 6 else 'validation' if episode % 10 < 8 else 'test'
                key = (split, truth['category'])
                bucket = buckets[key]
                if len(bucket) < targets[key]:
                    bucket.append({'id': s.key, 'split': split, 'state': s.to_dict(),
                                   'oracle': truth, 'upstream': UPSTREAM})
                    collected += 1
                    if collected >= max_per_episode:
                        break
            s, event = step(s, rng.choice(ACTIONS))
            if event in ('goal', 'timeout'):
                break
        if all(len(v) >= targets[k] for k, v in buckets.items()):
            break
    rows = [row for bucket in buckets.values() for row in bucket]
    rng.shuffle(rows)
    rows = rows[:count]
    return rows, {'requested': count, 'actual': len(rows), 'bank_seed': seed,
                  'horizon': horizon, 'upstream': UPSTREAM,
                  'categories': dict(Counter(r['oracle']['category'] for r in rows)),
                  'splits': dict(Counter(r['split'] for r in rows)),
                  'max_states_per_episode': max_per_episode,
                  'source_episodes': {split: len({r['state']['source_seed'] for r in rows if r['split'] == split})
                                      for split in sizes},
                  'balanced': all(len(v) >= targets[k] for k, v in buckets.items())}
