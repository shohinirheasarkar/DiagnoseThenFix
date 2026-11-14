"""This compares matched repairs, counting each starting program as one experiment."""
import math
from collections import defaultdict

import numpy as np

from .analysis import holm

BRANCHES = ('original', 'uniform', 'targeted')
CONTRASTS = (('uniform', 'targeted'), ('original', 'uniform'), ('original', 'targeted'))


def _interval(values, estimate, programs, observations):
    low, high = np.quantile(values, [.025, .975])
    return {'mean': float(estimate), 'low': float(low), 'high': float(high),
            'programs': programs, 'observations': observations}


def _check_options(replicates):
    if not isinstance(replicates, int) or replicates < 100:
        raise ValueError('Use at least 100 bootstrap replicates')


def _grid(rows, cells, metric):
    grouped = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        branch = row['branch']
        if branch not in BRANCHES:
            raise ValueError(f'Unknown branch: {branch}')
        value = float(row[metric])
        if not math.isfinite(value):
            raise ValueError('Metrics must be finite; retain failures as scored outcomes')
        key = tuple(row[field] for field in cells)
        target = grouped[row['program_id']][branch]
        if key in target:
            raise ValueError('Duplicate observation')
        target[key] = value
    if not grouped:
        raise ValueError('No observations')
    reference = None
    for branches in grouped.values():
        if set(branches) != set(BRANCHES):
            raise ValueError('Every program requires all three branches')
        for values in branches.values():
            keys = set(values)
            if reference is None:
                reference = keys
            if keys != reference:
                raise ValueError('Unbalanced cells: incomplete pairs must be reported separately')
    programs = sorted(grouped, key=str)
    keys = sorted(reference, key=str)
    return np.array([[[grouped[p][b][k] for k in keys] for b in BRANCHES]
                     for p in programs]), programs, keys



def paired_signflip(differences, seed=0, draws=10000):
    """Two-sided paired sign-flip test assumes exchangeability under the null."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Need finite paired differences')
    if not isinstance(draws, int) or draws < 100:
        raise ValueError('Need at least 100 Monte Carlo draws')
    observed = abs(values.sum())
    threshold = observed - 1e-12
    n = len(values)
    if n <= 20:
        extreme, total = 0, 2 ** n
        for start in range(0, total, 8192):
            ids = np.arange(start, min(start + 8192, total), dtype=np.uint64)
            signs = ((ids[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(float) * 2 - 1
            extreme += int((np.abs((signs * values).sum(axis=1)) >= threshold).sum())
        return {'p_value': extreme / total, 'test': 'exact two-sided paired sign-flip',
                'permutations': total}
    rng = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, draws, 8192):
        signs = rng.integers(0, 2, size=(min(8192, draws - start), n)) * 2 - 1
        extreme += int((np.abs((signs * values).sum(axis=1)) >= threshold).sum())
    return {'p_value': (extreme + 1) / (draws + 1),
            'test': 'Monte Carlo two-sided paired sign-flip; plus-one correction',
            'permutations': draws}

def forecast_contrasts(rows, seed=0, replicates=5000, expected_programs=None,
                       board_macro=False, expected_cells=None):
    """Intervals describe program variability conditional on this fixed test bank."""
    _check_options(replicates)
    strata = defaultdict(list)
    for row in rows:
        if row['correct'] not in (True, False) or row['valid'] not in (True, False):
            raise ValueError('Forecast correctness and validity must be binary')
        if row['correct'] and not row['valid']:
            raise ValueError('An invalid forecast cannot be correct')
        strata[row['model']].append(row)
    observed_models = {row['model'] for records in strata.values() for row in records}
    if expected_programs is not None and observed_models != set(expected_programs):
        raise ValueError('Observed models differ from the frozen expected cohort')
    output = []
    rng = np.random.default_rng(seed)
    for model, records in sorted(strata.items()):
        grid, programs, cells = _grid(records, ('state_id', 'horizon'), 'correct')
        if expected_programs is not None and set(programs) != set(expected_programs[model]):
            raise ValueError('Observed programs differ from the frozen expected cohort')
        if expected_cells is not None and set(cells) != {tuple(c) for c in expected_cells}:
            raise ValueError('Observed forecast cells differ from the frozen expected cohort')
        means = grid.mean(axis=2)
        if board_macro:
            board_for_cell = {}
            for row in records:
                cell = (row['state_id'], row['horizon'])
                board = row['board_id']
                if cell in board_for_cell and board_for_cell[cell] != board:
                    raise ValueError('A forecast cell cannot change board across candidates')
                board_for_cell[cell] = board
            boards = sorted(set(board_for_cell.values()), key=str)
            means = np.stack([grid[:, :, [i for i, cell in enumerate(cells)
                                         if board_for_cell[cell] == board]].mean(axis=2)
                              for board in boards]).mean(axis=0)
        draws = rng.integers(len(programs), size=(replicates, len(programs)))
        for left, right in CONTRASTS:
            differences = means[:, BRANCHES.index(right)] - means[:, BRANCHES.index(left)]
            output.append({'model': model, 'contrast': f'{right}-{left}',
                           'primary': left == 'uniform' and not board_macro, 'test_cells': len(cells),
                           'weighting': 'equal-board' if board_macro else 'equal-case',
                           'uncertainty': 'paired program bootstrap; fixed held-out bank',
                           **_interval(differences[draws].mean(axis=1), differences.mean(),
                                       len(programs), len(programs) * len(cells))})
            if left == 'uniform' and not board_macro:
                output[-1].update(paired_signflip(differences, seed=seed))
    primary = [r for r in output if r['primary'] and not board_macro]
    for row, adjusted in zip(primary, holm([r['p_value'] for r in primary])):
        row['p_value_holm'] = adjusted
        row['holm_family_size'] = len(primary)
    return output


def live_contrasts(rows, metric='goals', seed=0, replicates=5000,
                   expected_programs=None, expected_seeds=None, expected_regimes=None):
    """Resample programs and game seeds independently, retaining paired branches."""
    _check_options(replicates)
    strata = defaultdict(list)
    for row in rows:
        strata[(row['model'], row['regime'])].append(row)
    observed_models = {row['model'] for records in strata.values() for row in records}
    if expected_programs is not None and observed_models != set(expected_programs):
        raise ValueError('Observed models differ from the frozen expected cohort')
    if expected_regimes is not None:
        expected = {(m, r) for m in observed_models for r in expected_regimes}
        if set(strata) != expected:
            raise ValueError('Observed live regimes differ from the frozen expected cohort')
    output = []
    rng = np.random.default_rng(seed)
    for (model, regime), records in sorted(strata.items()):
        grid, programs, cells = _grid(records, ('env_seed',), metric)
        if expected_programs is not None and set(programs) != set(expected_programs[model]):
            raise ValueError('Observed programs differ from the frozen expected cohort')
        if expected_seeds is not None and {c[0] for c in cells} != set(expected_seeds):
            raise ValueError('Observed game seeds differ from the frozen expected cohort')
        pdraws = rng.integers(len(programs), size=(replicates, len(programs)))
        edraws = rng.integers(len(cells), size=(replicates, len(cells)))
        for left, right in CONTRASTS:
            differences = grid[:, BRANCHES.index(right), :] - grid[:, BRANCHES.index(left), :]
            samples = np.array([differences[p[:, None], e[None, :]].mean()
                                for p, e in zip(pdraws, edraws)])
            output.append({'model': model, 'regime': regime, 'metric': metric,
                           'contrast': f'{right}-{left}', 'environment_seeds': len(cells),
                           'uncertainty': 'crossed program and environment bootstrap',
                           **_interval(samples, differences.mean(), len(programs),
                                       len(programs) * len(cells))})
    return output
