"""This runs one local research queue, saves every completed result, and stops before the review deadline."""
import argparse
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('DIAGNOSE_SOURCE_ROOT', ROOT)).resolve()
sys.path.insert(0, str(SOURCE))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def deadline_model(model, stop):
    transport = model.transport

    def send(route, payload):
        unload = (route == '/api/generate' and payload and payload.get('keep_alive') == 0
                  and not payload.get('prompt') and not payload.get('messages'))
        if route in ('/api/chat', '/api/generate') and not unload and time.time() >= stop:
            raise TimeoutError('Research deadline reached before request dispatch')
        return transport(route, payload)

    model.transport = send
    return model


def profile(base, diaries, commitments):
    if base.get('split') != 'validation' or any(
            r.get('split') != 'validation' for r in [*diaries, *commitments]):
        raise ValueError('Repair selection needs validation records only')
    result = dict(base)
    baseline = {(r['state_id'], r['task']['seed']): r['score'] for r in diaries
                if r['task']['length'] == 0}
    groups = {}
    for r in diaries:
        t, s = r['task'], r['score']
        if not t['length'] or t['repair'] != 'base':
            continue
        b = baseline.get((r['state_id'], t['seed']))
        if b is None:
            continue
        stable = bool(s['valid'] and b['valid'] and
                      not (s['action'] != b['action'] and b['safe'] and not s['safe']))
        groups.setdefault((t['kind'], t['length'], t['outcomes']), []).append(stable)
    result['D'] = min((statistics.mean(v) for v in groups.values()), default=0.)
    c = [r['score'] for r in commitments if r['condition'] == 'own']
    result['C'] = statistics.mean(s['valid'] and not s.get('inconsistent', True) for s in c) if c else 0.
    result.update(history_not_profiled=False, split='validation',
                  followup_profile={'D': 'worst validation diary stability; invalid pairs score zero',
                                    'C': 'matched own-forecast consistency; invalid replies score zero',
                                    'diary_trials': len(diaries), 'commitment_trials': len(commitments),
                                    'thresholds': 'unchanged fixed A/B .8 and C/D/E .9',
                                    'exploratory': True})
    return result


def main():
    from diagnose.commitment import run as commitments
    from diagnose.composed import ComposedPolicy
    from diagnose.context import real_diary, run as context, summarize
    from diagnose.live import baseline, episode
    from diagnose.model import LocalModel, append_jsonl, read_jsonl
    from diagnose.runner import run as probes, source_fingerprint
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('results/overnight'))
    parser.add_argument('--context-stop', default='2026-11-13T10:50:00+00:00')
    parser.add_argument('--stop', default='2026-11-13T15:05:00+00:00')
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    stop = datetime.fromisoformat(args.stop).timestamp()
    context_stop = min(stop, datetime.fromisoformat(args.context_stop).timestamp())
    if stop <= time.time() or stop <= context_stop:
        parser.error('Need a future collection deadline after the context deadline')
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    previous = json.loads((output/'cycle.json').read_text()) if (output/'cycle.json').exists() else {}
    fingerprint = source_fingerprint()
    if previous and previous['source'] != fingerprint:
        raise ValueError('Worker source changed; use a new output directory')
    lock = output / 'worker.lock'
    with lock.open('x') as handle:
        handle.write(str(os.getpid()))
    state = {'started_at': previous.get('started_at', datetime.now(timezone.utc).isoformat()),
             'resumed_at': datetime.now(timezone.utc).isoformat(), 'status': 'running',
             'source': fingerprint, 'source_commit': (SOURCE/'SOURCE_REVISION').read_text().strip(),
             'context_stop': args.context_stop, 'collection_stop': args.stop,
             'full_research_plan_complete': False, 'policy_revision': 'v2'}
    collector_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    state.update(collector_sha256=collector_hash, request_deadline_enforced=True,
                 collector_sha256_history=list(dict.fromkeys([
                     *previous.get('collector_sha256_history', []),
                     *([previous['collector_sha256']] if 'collector_sha256' in previous else []),
                     collector_hash])),
                 source_commit_history=list(dict.fromkeys([
                     *previous.get('source_commit_history', []),
                     *([previous['source_commit']] if previous else []), state['source_commit']])),
                 queue_revision='protect-context-stage-coverage',
                 phase_history=previous.get('phase_history', []),
                 phase_minutes={'D': 35, 'E': 20, 'B-chain': 5})
    save(output/'cycle.json', state)
    models = ('qwen3:1.7b-q4_K_M', 'qwen3:4b-q4_K_M')
    current_model = None
    try:
        if args.wait_pid:
            while time.time() < context_stop:
                try:
                    os.kill(args.wait_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(10)
            else:
                raise RuntimeError('Previous collector did not stop; refusing concurrent models')
        bank = json.loads((ROOT/'data/freeway.json').read_text())['situations']
        for tag in models:
            if time.time() >= context_stop:
                break
            current_model = deadline_model(LocalModel(tag, cache=ROOT/'.cache/model', timeout=90), stop)
            folder = output/tag.replace(':', '-')
            previous = ROOT/'results/cycle'/tag.replace(':', '-')
            old_profile = json.loads((previous/'profile.json').read_text())
            old_revision = json.loads((previous.parent/'cycle.json').read_text())['source_commit']
            real = real_diary(read_jsonl(previous/'live-base.jsonl'), source_revision=old_revision)
            kwargs = dict(representation=old_profile['best_representation'], stop_at=context_stop)
            d = context(bank, current_model, folder/'D-validation', 'D', real=real,
                        limit=4, seeds=(0,), split='validation', **kwargs)
            c = commitments(bank, current_model, folder/'C-validation', limit=4, seeds=(0,),
                            split='validation', max_tokens=256, **kwargs)
            save(folder/'profile.json', profile(old_profile, d, c))
            current_model.transport('/api/generate', {'model': tag, 'keep_alive': 0})
            current_model = None
        for stage in ('D', 'E', 'B-chain'):
            for tag in models:
                if time.time() >= context_stop:
                    break
                folder = output/tag.replace(':', '-')
                if not (folder/'profile.json').exists():
                    continue
                frozen_profile = json.loads((folder/'profile.json').read_text())
                current_model = deadline_model(LocalModel(tag, cache=ROOT/'.cache/model', timeout=90), stop)
                representation = frozen_profile['best_representation']
                phase_stop = min(context_stop, time.time()+60*state['phase_minutes'][stage])
                phase = {'stage': stage, 'model': tag, 'started_at': datetime.now(timezone.utc).isoformat(),
                         'stop_at': datetime.fromtimestamp(phase_stop, timezone.utc).isoformat()}
                state['phase_history'].append(phase)
                state['phase'] = f'{stage}:{tag}'
                save(output/'cycle.json', state)
                if stage == 'B-chain':
                    rows = probes(bank, current_model, folder/'B-chained-calls', stages=('B',),
                           representations=(representation,), horizons=(1, 3, 5),
                           repairs=('base', 'chained_calls'), seeds=(0, 1), limit=2,
                           stop_at=phase_stop, max_tokens=256)
                else:
                    previous = ROOT/'results/cycle'/tag.replace(':', '-')
                    old_revision = json.loads((previous.parent/'cycle.json').read_text())['source_commit']
                    real = real_diary(read_jsonl(previous/'live-base.jsonl'), source_revision=old_revision) if stage == 'D' else ()
                    rows = context(bank, current_model, folder/f'{stage}-test', stage,
                                   representation, real=real, limit=4, stop_at=phase_stop)
                    save(folder/f'{stage}-test'/'summary.json', summarize(rows))
                phase.update(ended_at=datetime.now(timezone.utc).isoformat(), saved_trials=len(rows))
                save(output/'cycle.json', state)
                current_model.transport('/api/generate', {'model': tag, 'keep_alive': 0})
                current_model = None
        for name in ('random', 'stay', 'advance', 'planner', 'oracle'):
            path = output/f'baseline-{name}.jsonl'
            completed = {r['seed'] for r in read_jsonl(path)}
            for seed in range(60000, 60050):
                if seed not in completed and time.time() < stop:
                    append_jsonl(path, {'condition': name, **episode(baseline(name, seed), seed)})
        state['phase'] = 'live'
        save(output/'cycle.json', state)
        for tag in models:
            if time.time() >= stop:
                break
            folder = output/tag.replace(':', '-')
            if not (folder/'profile.json').exists():
                continue
            frozen_profile = json.loads((folder/'profile.json').read_text())
            current_model = deadline_model(LocalModel(tag, cache=ROOT/'.cache/model', timeout=90), stop)
            save(folder/'live-design.json', {'source': source_fingerprint(), 'model': tag,
                  'digest': current_model.digest, 'version': current_model.version,
                  'profile': frozen_profile, 'tick_seconds': 1., 'seeds': [60000, 60049],
                  'max_tokens': 512, 'policy_revision': 'v2', 'conditions': ['base', 'selected', 'all']})
            completed = {c: {r['seed'] for r in read_jsonl(folder/f'live-{c}.jsonl')
                             if not r['censored']} for c in ('base', 'selected', 'all')}
            for seed in range(60000, 60050):
                for condition in ('base', 'selected', 'all'):
                    if seed in completed[condition] or time.time() >= stop:
                        continue
                    policy = ComposedPolicy(current_model, frozen_profile, condition,
                                            seed=seed, max_tokens=512, revision='v2')
                    record = episode(policy, seed, tick_seconds=1., stop_at=stop)
                    append_jsonl(folder/f'live-{condition}.jsonl', {'condition': condition,
                                 'policy_revision': 'v2', **record})
                    print(f'LIVE {tag} {condition} {seed} success={record["success"]} censored={record["censored"]}', flush=True)
                if time.time() >= stop:
                    break
            current_model.transport('/api/generate', {'model': tag, 'keep_alive': 0})
            current_model = None
        state.update(status='collection_finished', ended_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save(output/'cycle.json', state)
        try:
            if current_model is not None:
                current_model.transport('/api/generate', {'model': current_model.name, 'keep_alive': 0})
        finally:
            lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
