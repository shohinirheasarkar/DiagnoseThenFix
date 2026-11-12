# Diagnose, then fix

Where do small language models get stuck in real-time games? We separate reading
the board, predicting motion, choosing an action, copying history, and following
priors. Then we test whether fixing the diagnosed stage helps.

## Run locally

Python 3.11+:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test,paper]'
pip install --no-deps 'git+https://github.com/SALT-NLP/RealtimeGym.git@3d5b3ef37dafee25ca2e8d2e8d71f5fe842eda7b'
pytest
```

We use the actual Freeway timer, long-car, collision, and wrap rules. The compact
replay engine is checked against upstream trajectories. The oracle's safe set
means a collision-free continuation exists through its stated horizon; it isn't
an unlimited-horizon safety guarantee. Empty safe sets stay empty.

The six renderers share one immutable state. All retain object identities, motion
parameters and timer phase. Our lane table and relative list retain every object
so perception comparisons don't silently change available information.

See [the preregistration](preregistration.md) for hypotheses, splits, controls,
and what counts as evidence. Model collection has not started. Full multi-game
experiments and the paper are still in progress.

Built on [RealtimeGym](https://github.com/SALT-NLP/RealtimeGym) and
[Wen et al.](https://arxiv.org/abs/2511.04898). The research framing also builds on
[Okcu et al.](https://arxiv.org/abs/2610.17444); stage diagnosis itself is not new.
