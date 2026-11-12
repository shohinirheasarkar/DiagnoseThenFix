"""This runs frozen-board studies, remembers completed work, and records exactly what was measured."""
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import subprocess

from .analysis import summarize
from .model import append_jsonl, canonical, parse_reply, read_jsonl
from .probes import inventory, make_task, score
from .world import State


def source_fingerprint():
    files = sorted(Path(__file__).parent.glob('*.py'))
    return hashlib.sha256(b''.join(p.name.encode() + p.read_bytes() for p in files)).hexdigest()


def run(bank, model, output, stages=('A', 'B', 'C'), representations=('coordinates',),
        seeds=(0,), thinking=False, horizons=(1, 5, 10), paraphrases=(0,),
        repairs=('base',), split='test', limit=10, max_tokens=512,
        history_kinds=('none',), history_lengths=(0,), skins=('cars',), labels=('standard',)):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selected = sorted((r for r in bank if r['split'] == split), key=lambda r: r['id'])[:limit]
    if not selected:
        raise ValueError(f'No bank states in split {split}')
    design = {'source': source_fingerprint(), 'model': model.name, 'digest': model.digest,
              'version': model.version, 'model_details': model.info.get('details'),
              'stages': stages, 'representations': representations, 'seeds': seeds,
              'thinking': thinking, 'horizons': horizons, 'paraphrases': paraphrases,
              'repairs': repairs, 'split': split, 'state_ids': [r['id'] for r in selected],
              'bank_sha256': hashlib.sha256(canonical(bank).encode()).hexdigest(),
              'max_tokens': max_tokens, 'history_kinds': history_kinds,
              'history_lengths': history_lengths, 'skins': skins, 'labels': labels,
              'python': platform.python_version()}
    design = json.loads(canonical(design))
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())['design'] != design:
            raise ValueError('Output folder belongs to a different design; choose a new folder')
    else:
        git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
        manifest_path.write_text(json.dumps({'design': design, 'git_revision': git.stdout.strip(),
                                            'created_at': datetime.now(timezone.utc).isoformat()},
                                           indent=2))
    results_path = output / 'responses.jsonl'
    done = {r['trial_id'] for r in read_jsonl(results_path)}
    for row, stage, rep, seed, paraphrase, repair in itertools.product(
            selected, stages, representations, seeds, paraphrases, repairs):
        state = State.from_dict(row['state'])
        kinds = inventory()[stage]
        ks = horizons if stage in ('B', 'C', 'D', 'E') else (1,)
        diaries = list(itertools.product(history_kinds, history_lengths)) if stage == 'D' else [('none', 0)]
        frames = list(itertools.product(skins, labels)) if stage == 'E' else [('cars', 'standard')]
        for kind, k, (diary, length), (skin, label_set) in itertools.product(kinds, ks, diaries, frames):
            task = make_task(state, stage, rep, k, seed, paraphrase, kind=kind, repair=repair,
                             history_kind=diary, history_length=length, skin=skin, labels=label_set)
            trial = hashlib.sha256(canonical({'design': design, 'id': row['id'], 'task': task}).encode()).hexdigest()
            if trial in done:
                continue
            response = model.ask(task['prompt'], seed, thinking, max_tokens, schema='json')
            required = ('answer', 'confidence') if stage in ('A', 'B') else ('action', 'collision', 'confidence')
            value, error = parse_reply(response, required)
            result = {'trial_id': trial, 'state_id': row['id'], 'source_seed': state.source_seed,
                      'split': split, 'model': model.name, 'digest': model.digest,
                      'thinking': thinking, 'task': task, 'response': response,
                      'parsed': value, 'score': score(task, value, error, state, row['oracle'])}
            append_jsonl(results_path, result)
            done.add(trial)
            print(f'{model.name} {stage} {kind} k={k} {row["id"]} '
                  f'correct={result["score"]["correct"]}', flush=True)
    results = read_jsonl(results_path)
    (output / 'summary.json').write_text(json.dumps(summarize(results), indent=2))
    return results
