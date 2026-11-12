"""This summarizes repeated trials without pretending that repeats are independent boards."""
from collections import Counter, defaultdict
import math
import random
import statistics


def bootstrap(rows, metric='correct', cluster='state_id', seed=0, replicates=2000):
    groups = defaultdict(list)
    for row in rows:
        if metric in row:
            groups[row[cluster]].append(float(row[metric]))
    means = [statistics.mean(v) for v in groups.values()]
    if not means:
        return {'mean': None, 'low': None, 'high': None, 'clusters': 0, 'observations': 0}
    rng = random.Random(seed)
    samples = sorted(statistics.mean(rng.choices(means, k=len(means)))
                     for _ in range(replicates))
    return {'mean': statistics.mean(means), 'low': samples[int(.025 * (replicates - 1))],
            'high': samples[int(.975 * (replicates - 1))], 'clusters': len(means),
            'observations': sum(map(len, groups.values()))}


def paired_bootstrap(left, right, metric='correct', seed=0, replicates=2000):
    def means(rows):
        groups = defaultdict(list)
        for row in rows:
            if metric in row:
                groups[row['state_id']].append(float(row[metric]))
        return {k: statistics.mean(v) for k, v in groups.items()}
    a, b = means(left), means(right)
    shared = sorted(a.keys() & b.keys())
    return bootstrap([{'state_id': k, 'difference': b[k] - a[k]} for k in shared],
                     'difference', seed=seed, replicates=replicates)


def ece(rows, bins=10):
    usable = [r for r in rows if r.get('valid') and 'confidence' in r]
    if not usable:
        return None
    groups = defaultdict(list)
    for row in usable:
        groups[min(bins-1, int(row['confidence'] * bins))].append(row)
    return sum(len(g) / len(usable) * abs(statistics.mean(r['confidence'] for r in g)
                                        - statistics.mean(r['correct'] for r in g))
               for g in groups.values())


def available_foresight(curve, threshold=.8):
    f = 0
    for k in range(1, 11):
        if k not in curve or curve[k] < threshold:
            break
        f = k
    return f


def holm(pvalues):
    if any(not 0 <= p <= 1 or not math.isfinite(p) for p in pvalues):
        raise ValueError('Invalid p value')
    ordered = sorted(enumerate(pvalues), key=lambda x: x[1])
    result = [0.] * len(ordered)
    previous = 0.
    for rank, (index, p) in enumerate(ordered):
        previous = max(previous, min(1., (len(ordered) - rank) * p))
        result[index] = previous
    return result


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        t = row['task']
        key = (row['model'], row['thinking'], t['stage'], t['representation'], t['kind'],
               t['k'], t['repair'], t['history_kind'], t['history_length'],
               t['skin'], t['labels'])
        groups[key].append({**row['score'], 'state_id': row.get('source_seed', row['state_id']),
                            'wall_seconds': row['response']['wall_seconds']})
    output = []
    for key, values in sorted(groups.items(), key=lambda x: str(x[0])):
        fields = ('model', 'thinking', 'stage', 'representation', 'kind', 'k', 'repair',
                  'history_kind', 'history_length', 'skin', 'labels')
        is_null = key[2] == 'E' and key[4] == 'null'
        output.append({**dict(zip(fields, key)), 'accuracy': None if is_null else bootstrap(values),
                       'invalid': sum(not v['valid'] for v in values),
                       'ece_valid_only': None if is_null else ece(values),
                       'actions': dict(Counter(v['action'] for v in values if 'action' in v)),
                       'mean_wall_seconds': statistics.mean(v['wall_seconds'] for v in values),
                       'contradiction': bootstrap(values, 'contradiction'),
                       'safety': bootstrap(values, 'safe')})
    return output
