"""This runs the declared local batch one model at a time, keeping every completed trial on disk."""
import argparse
import json
from pathlib import Path

from diagnose.model import LocalModel
from diagnose.runner import run

MODELS = ['qwen3:1.7b-q4_K_M', 'qwen3:4b-q4_K_M', 'qwen3:8b-q4_K_M', 'qwen3:14b-q4_K_M']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Run a larger Freeway matrix; remaining plan items are separate')
    parser.add_argument('--models', nargs='+', default=MODELS)
    parser.add_argument('--output', type=Path, default=Path('results/local-pilot'))
    args = parser.parse_args()
    bank = json.loads(Path('data/freeway.json').read_text())['situations']
    seeds = tuple(range(5)) if args.full else (0,)
    paraphrases = (0, 1, 2) if args.full else (0,)
    for name in args.models:
        model = LocalModel(name)
        for thinking in ((False, True) if args.full else (False,)):
            tag = name.replace(':', '-').replace('/', '-') + ('-think' if thinking else '-direct')
            root = args.output / tag
            common = dict(seeds=seeds, paraphrases=paraphrases, thinking=thinking,
                          max_tokens=2048 if thinking else 512)
            for split in ('validation', 'test'):
                run(bank, model, root / f'A-{split}', stages=('A',),
                    representations=('ascii', 'coordinates', 'prose', 'lanes', 'relations', 'json'),
                    limit=100 if args.full else 2, split=split, **common)
            for stage in ('B', 'C', 'D', 'E'):
                options = {}
                if stage == 'D':
                    options = {'history_kinds': ('none', 'stay', 'up', 'down', 'random', 'oracle'),
                               'history_lengths': (0, 3, 6, 12)}
                elif stage == 'E':
                    options = {'skins': ('cars', 'logs', 'clouds', 'X'),
                               'labels': ('standard', 'neutral', 'words')}
                run(bank, model, root / stage, stages=(stage,),
                    horizons=tuple(range(1, 11)) if stage == 'B' else (3,),
                    limit=100 if args.full else 4 if stage == 'B' else 2,
                    split='test', **options, **common)
            run(bank, model, root / 'B-repairs', stages=('B',), horizons=(1, 5, 10),
                repairs=('base', 'rule', 'scratchpad', 'chained', 'code', 'tool'),
                limit=100 if args.full else 2, **common)
            run(bank, model, root / 'C-repairs', stages=('C',), horizons=(3,),
                repairs=('generic', 'forecast_then_act', 'bound', 'retry', 'oracle_shield'),
                limit=100 if args.full else 2, **common)
        model.transport('/api/generate', {'model': name, 'keep_alive': 0})


if __name__ == '__main__':
    main()
