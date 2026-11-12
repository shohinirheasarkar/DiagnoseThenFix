"""This replays the original games from a seed and action diary, so saved states need no pickle files."""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import random

import numpy as np

from .world import UPSTREAM

GAME_ACTIONS = {'Freeway': ('U', 'S', 'D'), 'Snake': ('U', 'D', 'L', 'R', 'S'),
                'Overcooked': ('U', 'D', 'L', 'R', 'I', 'S')}


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@contextmanager
def seeded(seed):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@dataclass(frozen=True)
class Snapshot:
    game: str
    seed: int
    actions: tuple[str, ...]
    observation: str
    reward: float
    terminal: bool
    upstream: str = UPSTREAM

    @property
    def key(self):
        return hashlib.sha256(self.observation.encode()).hexdigest()[:20]

    def to_dict(self):
        return {'game': self.game, 'seed': self.seed, 'actions': list(self.actions),
                'observation': json.loads(self.observation), 'reward': self.reward,
                'terminal': self.terminal, 'upstream': self.upstream}

    @classmethod
    def from_dict(cls, data):
        if data['upstream'] != UPSTREAM:
            raise ValueError('Simulator revision mismatch')
        return cls(data['game'], data['seed'], tuple(data['actions']),
                   json.dumps(data['observation'], sort_keys=True),
                   data['reward'], data['terminal'], data['upstream'])


def restore(game, seed, actions=()):
    import realtimegym
    base = game.split('-')[0]
    if base not in GAME_ACTIONS:
        raise ValueError(f'Unknown game {game}')
    env, _, _ = realtimegym.make(game, seed=seed, render=False)
    env.reset()
    for action in actions:
        if env.terminal:
            raise ValueError('Action diary continues past a terminal state')
        if action not in GAME_ACTIONS[base]:
            raise ValueError('Unknown action')
        env.step(action)
    return env


def snapshot(game, seed=0, actions=()):
    with seeded(seed):
        env = restore(game, seed, actions)
        return Snapshot(game, seed, tuple(actions),
                        json.dumps(plain(env.observe()), sort_keys=True),
                        float(env.reward), bool(env.terminal))


def advance(saved, action):
    if saved.terminal:
        raise ValueError('Cannot advance a terminal snapshot')
    with seeded(saved.seed):
        env = restore(saved.game, saved.seed, saved.actions)
        actual = json.dumps(plain(env.observe()), sort_keys=True)
        if actual != saved.observation or env.reward != saved.reward:
            raise ValueError('Replay mismatch; simulator or saved state changed')
        if action not in GAME_ACTIONS[saved.game.split('-')[0]]:
            raise ValueError('Unknown action')
        _, done, reward, collision = env.step(action)
        nxt = Snapshot(saved.game, saved.seed, saved.actions + (action,),
                       json.dumps(plain(env.observe()), sort_keys=True), float(reward), bool(done))
        died = saved.game.startswith('Snake') and done and reward < saved.reward
        return nxt, {'collision': bool(collision or died), 'reward_delta': reward - saved.reward,
                     'terminal': bool(done)}


def search(saved, horizon=3, node_budget=10000):
    if horizon < 1 or saved.terminal:
        raise ValueError('Search needs a positive horizon and a nonterminal snapshot')
    actions = GAME_ACTIONS[saved.game.split('-')[0]]
    scores, visited, nodes = {}, {}, 0
    truncated = False

    def value(current, depth):
        nonlocal nodes, truncated
        key = (current.observation, current.reward, current.terminal, depth)
        if key in visited:
            return visited[key]
        if depth == 0 or current.terminal:
            return (1, current.reward - saved.reward)
        if nodes >= node_budget:
            truncated = True
            return None
        nodes += 1
        options = []
        for action in actions:
            nxt, event = advance(current, action)
            if event['collision']:
                options.append((0, nxt.reward - saved.reward))
            else:
                score = value(nxt, depth - 1)
                if score is not None:
                    options.append(score)
        result = max(options) if options else None
        visited[key] = result
        return result

    for action in actions:
        nxt, event = advance(saved, action)
        scores[action] = (0, nxt.reward - saved.reward) if event['collision'] else value(nxt, horizon-1)
    if truncated:
        return {'exhaustive': False, 'safe': None, 'optimal': None, 'scores': scores,
                'horizon': horizon, 'nodes': nodes}
    safe = [a for a, score in scores.items() if score[0]]
    best = max(scores.values())
    return {'exhaustive': True, 'safe': safe,
            'optimal': [a for a, score in scores.items() if score == best] if safe else [],
            'scores': scores, 'horizon': horizon, 'nodes': nodes}
