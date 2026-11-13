"""These small fixes use the model's own forecast, clean its history, or subtract its default bias."""
import ast
import math
import operator
import random

from .world import ACTIONS


def bound_action(collision):
    if set(collision) != set(ACTIONS) or any(type(v) is not bool for v in collision.values()):
        raise ValueError('Expected one boolean collision belief per action')
    safe = [a for a in ACTIONS if not collision[a]]
    return safe[0] if safe else None


def retry_once(action, collision, retry):
    if action not in ACTIONS:
        raise ValueError('Invalid proposed action')
    if not collision[action]:
        return action, False
    new_action = retry()
    if new_action not in ACTIONS or collision[new_action]:
        return bound_action(collision), True
    return new_action, True


def calibrate(scores, null_scores):
    if set(scores) != set(ACTIONS) or set(null_scores) != set(ACTIONS):
        raise ValueError('Calibration needs complete action log probabilities')
    adjusted = {a: scores[a] - null_scores[a] for a in ACTIONS}
    if any(not math.isfinite(v) for v in adjusted.values()):
        raise ValueError('Nonfinite score')
    return max(ACTIONS, key=adjusted.get), adjusted


def choose_repairs(profile, thresholds=None):
    if profile.get('split') != 'validation':
        raise ValueError('Fit the repair profile on validation only')
    thresholds = thresholds or {'A': .8, 'B': .8, 'C': .9, 'D': .9, 'E': .9}
    names = {'A': 'structured', 'B': 'code', 'C': 'bound', 'D': 'no_history', 'E': 'calibrate'}
    return {names[stage] for stage, cutoff in thresholds.items()
            if stage in profile and profile[stage] < cutoff}


def ablations(selected, seed=0):
    all_repairs = {'structured', 'code', 'bound', 'no_history', 'calibrate'}
    selected = set(selected)
    if not selected <= all_repairs:
        raise ValueError('Unknown repair')
    return {'base': set(), 'selected': selected, 'all': all_repairs,
            'random': set(random.Random(seed).sample(sorted(all_repairs), len(selected))),
            **{f'without_{r}': selected - {r} for r in sorted(selected)}}


def monitor(stays=0, collision=None, action=None, forecast_error=None, perception_ok=True):
    triggers = set()
    if not perception_ok:
        triggers.add('structured')
    if forecast_error is not None and forecast_error > 2:
        triggers.add('code')
    if collision is not None:
        if action is not None and collision.get(action, False):
            triggers.add('bound')
        if stays >= 3 and not collision.get('U', True):
            triggers.update(('no_history', 'calibrate'))
    return triggers


class ArithmeticProgram:
    def __init__(self, source, extended=False):
        tree = ast.parse(source)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise ValueError('Expected one forecast(x, timer, p, k) function')
        self.function = tree.body[0]
        self.extended = extended
        if ([a.arg for a in self.function.args.args] != ['x', 'timer', 'p', 'k']
                or self.function.decorator_list or self.function.args.defaults
                or self.function.args.vararg or self.function.args.kwarg
                or self.function.args.kwonlyargs or self.function.args.posonlyargs
                or self.function.returns or self.function.type_comment
                or any(a.annotation for a in self.function.args.args)):
            raise ValueError('Expected four plain arguments')
        allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Assign, ast.AugAssign,
                   ast.For, ast.If, ast.Return, ast.Expr, ast.Name, ast.Constant, ast.BinOp,
                   ast.UnaryOp, ast.Compare, ast.BoolOp, ast.Call, ast.Load, ast.Store,
                   ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
                   ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq,
                   ast.Lt, ast.LtE, ast.Gt, ast.GtE)
        if extended:
            allowed += (ast.IfExp,)
        if any(not isinstance(n, allowed) for n in ast.walk(tree)):
            raise ValueError('Only bounded arithmetic, assignments, if, and for are allowed')
        if len(list(ast.walk(tree))) > 500:
            raise ValueError('Program too large')

    def __call__(self, x, timer, p, k):
        if not 0 <= k <= 10:
            raise ValueError('Forecast horizon must be 0-10')
        env = dict(x=x, timer=timer, p=p, k=k)
        fuel = [3000]
        binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                  ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
        compare = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
                   ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}

        def burn():
            fuel[0] -= 1
            if fuel[0] < 0:
                raise ValueError('Arithmetic budget exceeded')

        def bounded(value):
            if type(value) in (float, int) and (not math.isfinite(value) or abs(value) > 1000000):
                raise ValueError('Numeric budget exceeded')
            return value

        def expression(node):
            burn()
            if isinstance(node, ast.Constant) and type(node.value) in (int, float, bool):
                return bounded(node.value)
            if isinstance(node, ast.Name):
                return env[node.id]
            if isinstance(node, ast.BinOp):
                return bounded(binary[type(node.op)](expression(node.left), expression(node.right)))
            if isinstance(node, ast.UnaryOp):
                return {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Not: operator.not_}[
                    type(node.op)](expression(node.operand))
            if isinstance(node, ast.IfExp) and self.extended:
                return expression(node.body if expression(node.test) else node.orelse)
            if isinstance(node, ast.BoolOp):
                if isinstance(node.op, ast.And):
                    return all(expression(v) for v in node.values)
                return any(expression(v) for v in node.values)
            if isinstance(node, ast.Compare):
                left = expression(node.left)
                for op, right in zip(node.ops, node.comparators):
                    other = expression(right)
                    if not compare[type(op)](left, other):
                        return False
                    left = other
                return True
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
                args = [expression(a) for a in node.args]
                if node.func.id == 'range':
                    result = range(*args)
                    if len(result) > 100:
                        raise ValueError('Loop too large')
                    return result
                if node.func.id in ('abs', 'int', 'min', 'max'):
                    return bounded({'abs': abs, 'int': int, 'min': min, 'max': max}[node.func.id](*args))
                if node.func.id == 'sign' and self.extended and len(args) == 1:
                    return (args[0] > 0) - (args[0] < 0)
            raise ValueError('Unsupported arithmetic expression')

        class Returned(Exception):
            pass

        def statements(nodes):
            for node in nodes:
                burn()
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    env[node.targets[0].id] = expression(node.value)
                elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                    env[node.target.id] = bounded(binary[type(node.op)](env[node.target.id], expression(node.value)))
                elif isinstance(node, ast.If):
                    statements(node.body if expression(node.test) else node.orelse)
                elif isinstance(node, ast.For) and isinstance(node.target, ast.Name) and not node.orelse:
                    values = expression(node.iter)
                    if not isinstance(values, range):
                        raise ValueError('Only bounded range loops are allowed')
                    for value in values:
                        env[node.target.id] = value
                        statements(node.body)
                elif isinstance(node, ast.Return):
                    raise Returned(expression(node.value))
                else:
                    raise ValueError('Unsupported statement')
        try:
            statements(self.function.body)
        except Returned as result:
            value = result.args[0]
            if type(value) not in (int, float) or int(value) != value:
                raise ValueError('Forecast must return an integer cell')
            return int(value)
        raise ValueError('Forecast did not return a value')
