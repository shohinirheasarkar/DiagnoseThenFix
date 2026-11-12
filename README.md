# Diagnose, then fix

Where do small language models get stuck in real-time games? We separate reading
the board, forecasting motion, choosing an action, copying history, and following
priors, then test fixes against simulator labels.

This eight-hour cycle completed 8,000 held-out Freeway forecast trials across
four local Qwen3 artifacts. [The review paper](output/pdf/diagnose-then-repair.pdf)
reports the results, failures, and partial live cohorts.

The full multi-game study remains unfinished; [the execution plan](execution_plan.md)
records the scope and deviations. Snake and Overcooked have tested snapshot/replay
adapters.

## Run locally

Use Python 3.12 and a running Ollama server with your chosen model installed.
The upstream checkout needs editable installation because its wheel omits assets.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[test,games,paper]'
git clone https://github.com/SALT-NLP/RealtimeGym.git .cache/realtimegym
git -C .cache/realtimegym checkout 3d5b3ef37dafee25ca2e8d2e8d71f5fe842eda7b
pip install --no-deps -e .cache/realtimegym
python -m pytest
```

```sh
diagnose probe --model qwen3:1.7b-q4_K_M --output results/my-run --stages B --limit 5
python -m diagnose.commitment --model qwen3:1.7b-q4_K_M --output results/my-commitments
```

## Check the paper and data

These commands use [saved evidence](results/review), with no model calls:

```sh
python scripts/check_evidence.py results/review
python paper/build.py --results results/review --output output/pdf/diagnose-then-repair.pdf
```

Requests include exact model hashes and prompts; completed trials resume without
being repeated. Missing models trigger an error, never an automatic download.
The finite-horizon oracle labels safe continuations, not indefinite safety.
Commitment tests explicitly match the chosen action to its forecast horizon;
older one-tick/repeated-action logs are only a risk-disagreement proxy.

See [the preregistration](preregistration.md). Built on
[RealtimeGym](https://github.com/SALT-NLP/RealtimeGym) and
[Wen et al.](https://arxiv.org/abs/2511.04898). Stage diagnosis itself is not new.
