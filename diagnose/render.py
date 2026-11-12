"""These six ways of showing a board all read the same car facts, without peeking ahead."""
import json

FORMATS = ('ascii', 'coordinates', 'prose', 'lanes', 'relations', 'json')


def render(state, representation='coordinates', skin='cars'):
    if representation not in FORMATS:
        raise ValueError(representation)
    if skin not in ('cars', 'logs', 'clouds', 'X'):
        raise ValueError(skin)
    heading = f'9 columns (0-8); lanes 0-9, goal 0; player ({state.lane},4); tick {state.turn}.'
    facts = [f'{skin} {i}: lane={c.lane}, head={c.x}, timer={c.timer}, '
             f'speed_parameter={c.speed}, length={c.length}' for i, c in enumerate(state.cars)]
    if representation == 'json':
        return json.dumps({'width': 9, 'goal_lane': 0, 'player_column': 4,
                           'object_label': skin, **state.to_dict()}, sort_keys=True)
    if representation == 'coordinates':
        body = '\n'.join(facts)
    elif representation == 'ascii':
        grid = [['.' for _ in range(9)] for _ in range(10)]
        for c in state.cars:
            for x in c.cells():
                if 0 <= x < 9:
                    grid[c.lane][x] = 'X'
        grid[state.lane][4] = 'P' if grid[state.lane][4] == '.' else '!'
        body = '  0 1 2 3 4 5 6 7 8\n' + '\n'.join(
            f'{lane} ' + ' '.join(row) for lane, row in enumerate(grid))
        body += '\nX=occupied, P=player, !=both. Motion metadata:\n' + '\n'.join(facts)
    elif representation == 'prose':
        body = '\n'.join(f'Object {i} is in lane {c.lane}, with head at column {c.x}, '
                         f'timer {c.timer}, speed parameter {c.speed}, and length {c.length}.'
                         for i, c in enumerate(state.cars))
    elif representation == 'lanes':
        body = 'lane | object | head minus player column | timer | speed parameter | length\n'
        body += '\n'.join(f'{c.lane} | {i} | {c.x - 4} | {c.timer} | {c.speed} | {c.length}'
                          for i, c in enumerate(state.cars))
    else:
        body = '\n'.join(f'{skin} {i}: {c.lane - state.lane} lanes below player, '
                         f'head {c.x - 4} columns right, timer {c.timer}, '
                         f'speed parameter {c.speed}, length {c.length}'
                         for i, c in enumerate(state.cars))
    return heading + '\n' + body


RULES = '''Game rules: U moves one lane toward 0, D toward 9, S stays. Column stays 4.
The player moves first. Reaching lane 0 wins before objects update.
Each object is [head x, lane, timer, speed_parameter p, length].
If x<0 set x=8; if x>8 set x=0; either wrap uses the whole tick.
Otherwise if abs(p)>=1: decrease timer by 1; only if timer becomes -1,
increase timer by abs(p) and move head by sign(p).
If abs(p)<1: move head by int(1/p). No same-tick modulo wrap.
An object occupies x, x-sign(p), ... for its length. After objects move, contact
with column 4 in the player's lane causes a collision and resets the board.
Only occupied cells at tick endpoints collide; no swept collision rule.
At tick 100 the game ends. Prefer safe progress to the goal; don't wait forever.'''
