# Diagnose, then fix

Where do small language models get stuck in real-time games? We separate reading
the board, forecasting motion, choosing an action, copying history, and following
priors, then test fixes against simulator labels.

The first eight-hour cycle completed 8,000 held-out Freeway forecast trials across
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
[The source comparison](related_work.md) explains what earlier work already covers.

## Overnight follow-up

[The plan checklist](requirements.md) shows what still needs work.
[Closed follow-up evidence](results/review-overnight) is saved separately;
neural live collection is still running, and the PDF update follows at the cap.
The follow-up adds paired diary/story tests and an opt-in v2 agent that checks generated code,
retries once, and keeps its own forecast when a repair fails. Old results stay
separate. To rescore a local context run without calling a model:

```sh
python -m diagnose.context --output results/overnight/qwen3-1.7b-q4_K_M/D-test --check
```

Saved programs can also drive an offline controller. `python -m diagnose.compiled
--help` shows how to replay them. This tests code reuse and delay charges with no
new model calls; it stays separate from the live-agent study.
