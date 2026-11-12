"""This gives the study a few plain commands: make boards, run probes, and summarize results."""
import argparse
import json
from pathlib import Path

from .analysis import summarize
from .model import LocalModel, append_jsonl, read_jsonl
from .runner import run
from .world import build_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    bank = commands.add_parser('bank')
    bank.add_argument('--count', type=int, default=500)
    bank.add_argument('--seed', type=int, default=0)
    bank.add_argument('--horizon', type=int, default=10)
    bank.add_argument('--max-episodes', type=int, default=3000)
    bank.add_argument('--output', type=Path, default=Path('data/freeway.json'))
    probe = commands.add_parser('probe')
    probe.add_argument('--bank', type=Path, default=Path('data/freeway.json'))
    probe.add_argument('--model', required=True)
    probe.add_argument('--output', type=Path, required=True)
    probe.add_argument('--stages', nargs='+', choices=list('ABCDE'), default=['A', 'B', 'C'])
    probe.add_argument('--representations', nargs='+', default=['coordinates'])
    probe.add_argument('--seeds', nargs='+', type=int, default=[0])
    probe.add_argument('--horizons', nargs='+', type=int, default=[1, 5, 10])
    probe.add_argument('--paraphrases', nargs='+', type=int, default=[0])
    probe.add_argument('--repairs', nargs='+', default=['base'])
    probe.add_argument('--thinking', action='store_true')
    probe.add_argument('--split', choices=['train', 'validation', 'test'], default='test')
    probe.add_argument('--limit', type=int, default=10)
    probe.add_argument('--max-tokens', type=int, default=512)
    probe.add_argument('--history-kinds', nargs='+', default=['none'])
    probe.add_argument('--history-lengths', nargs='+', type=int, default=[0])
    probe.add_argument('--skins', nargs='+', default=['cars'])
    probe.add_argument('--labels', nargs='+', default=['standard'])
    live = commands.add_parser('live')
    live.add_argument('--model')
    live.add_argument('--baseline', choices=['random', 'stay', 'advance', 'planner', 'oracle'])
    live.add_argument('--repair', default='base')
    live.add_argument('--episodes', type=int, default=50)
    live.add_argument('--seed-start', type=int, default=50000)
    live.add_argument('--tick-seconds', type=float, default=1.)
    live.add_argument('--thinking', action='store_true')
    live.add_argument('--output', type=Path, required=True)
    summary = commands.add_parser('summarize')
    summary.add_argument('responses', type=Path)
    args = parser.parse_args()
    if args.command == 'bank':
        rows, report = build_bank(args.count, args.seed, args.horizon, args.max_episodes)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text('{"manifest": ' + json.dumps(report, sort_keys=True) +
                               ', "situations": [\n' +
                               ',\n'.join(json.dumps(r, sort_keys=True) for r in rows) + '\n]}\n')
        print(json.dumps(report, indent=2))
    elif args.command == 'probe':
        data = json.loads(args.bank.read_text())
        model = LocalModel(args.model)
        run(data['situations'], model, args.output, args.stages, args.representations,
            args.seeds, args.thinking, args.horizons, args.paraphrases, args.repairs,
            args.split, args.limit, args.max_tokens, args.history_kinds,
            args.history_lengths, args.skins, args.labels)
    elif args.command == 'live':
        from .live import baseline, episode, model_policy
        if bool(args.model) == bool(args.baseline):
            parser.error('Choose exactly one --model or --baseline')
        model = LocalModel(args.model) if args.model else None
        completed = {r['seed']: r for r in read_jsonl(args.output)}
        condition = {'model': args.model, 'baseline': args.baseline, 'repair': args.repair,
                     'thinking': args.thinking, 'tick_seconds': args.tick_seconds,
                     'digest': model.digest if model else None}
        if any(row['condition'] != condition for row in completed.values()):
            raise ValueError('Live output belongs to another condition')
        for seed in range(args.seed_start, args.seed_start + args.episodes):
            if seed in completed:
                continue
            policy = (model_policy(model, seed, args.thinking, repair=args.repair) if model
                      else baseline(args.baseline, seed))
            result = episode(policy, seed, args.tick_seconds)
            append_jsonl(args.output, {'condition': condition, **result})
            print(f'episode {seed}: success={result["success"]} ticks={result["ticks"]}', flush=True)
    else:
        print(json.dumps(summarize(read_jsonl(args.responses)), indent=2))


if __name__ == '__main__':
    main()
