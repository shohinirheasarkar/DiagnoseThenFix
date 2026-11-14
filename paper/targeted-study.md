# When Repair Breaks the Interface

## Abstract

We tested whether choosing failure examples by diagnostic category helps more than
uniformly choosing true failures. Forty initial Freeway forecast programs received
two repair rounds in each arm. The completed study contains 200 logical requests,
1,458,000 forecast records, and 36,000 game episodes. Neither model had an observed
targeted-versus-uniform advantage. This is not a clean null result: all 40 final 4B
repair responses failed the required output interface. The 1.7B arms had identical
code in all 17 pairs where code remained available. The experiment exposes a failed
intervention and does not establish whether diagnosis helps under a reliable interface.

## Method

We used installed Qwen3 1.7B and 4B quantized models, with 20 initial sampling seeds
per size. Each initial program produced original, uniform-feedback, and targeted-
feedback branches. Each repair received at most six true failures per round, two
rounds, the same prompt format and paired sampling seed, and a 1,024-token cap.
Temperature was 0.7; thinking was off. Branch order was counterbalanced. Prompts
contained the current program and feedback without accumulated chat history.

Targeting prioritized the category with highest training error, then filled remaining
slots from other failures. Categories were boundary reset, fractional motion, and
left/right timer movement. With no failures, revision still proceeded without failing
examples. The final round was always scored; no best-round or test-based selection
was allowed. The strict output interface required a JSON `code` field.

The test held out 100 boards and used horizons 1–10. Board keys did not overlap
training, but 11,810 of 12,150 primitive test inputs (97.2%) occurred in training.
This tests familiar motion inputs in new boards, not unseen dynamics. Invalid
programs scored wrong. The primary contrast was targeted minus uniform accuracy,
with paired-program bootstrap intervals and two Holm-adjusted sign-flip tests.
Board-macro accuracy was descriptive sensitivity analysis.

Each candidate played 100 shared game seeds under zero, native, and two-tick delay
using one fixed human-written controller. The controller forecasts three decision
cycles including waiting before each action. Native execution uses ceiling rounding
at one second per tick. Offline generation costs are separate. Live intervals account
for shared programs and environment seeds. Earlier studies remain separate.

## Results

| Model | Original accuracy | Uniform repair | Targeted repair | Executable final programs per repair arm |
| --- | ---: | ---: | ---: | ---: |
| 1.7B | 43.776% | 38.115% | 38.115% | 17/20 |
| 4B | 100% | 0% | 0% | 0/20 |

Both primary contrasts were exactly zero, with zero-width empirical intervals and
Holm-adjusted p=1. This reflects identical paired outcomes, not population equivalence.

All 4B first-round feedback sets were empty because the original programs passed
training. The original 20 programs contained only one distinct source string. The
fixed revision policy then produced JSON with `candidate` instead of required `code`.
All 40 final 4B repair responses were rejected. Their alternative strings passed static
grammar checks, but were not executed or rescored. This post-hoc inspection does not
recover their performance or change the frozen results.

For 1.7B, executable programs decreased from 19 original programs to 17 per repair
arm. All 17 available final code pairs were identical; the other three pairs were
unavailable on both sides. Across the study, 29 of 200 logical requests were cache
hits, leaving 171 fresh requests. Cached duplicates are not fresh independent samples.

Gameplay also had zero targeted-versus-uniform contrasts. Repair reduced native
goal rates relative to originals by 1.15 percentage points for 1.7B and 20 points for
4B. This secondary result is limited by the controller: a ten-seed development check
with correct forecasts reached 8/10 goals at zero delay and only 1/10 under each
delayed regime. The planned matched 100-seed assisted test control was not collected
before the research cutoff. No new experiments were started afterward.

## Interpretation

The semantic targeting intervention was not realized as different executable final
programs. A zero effect therefore cannot establish that diagnostic targeting is
ineffective. Mandatory revision can also damage an already-correct program when
format compliance fails. Early stopping, retaining the original, and accepting other
JSON keys would be different policies; they were not tested here.

The result supports checking that a repair becomes executable and differs from its
control before interpreting its downstream effect. It does not justify a general
claim about diagnosis, larger models, other games, or unseen dynamics. The large
number of simulator records does not replace diverse independent program samples.

## Reproducibility and prior work

The release preserves the frozen protocol, raw responses, programs, feedback,
per-case scores, compressed trajectories, source hashes, complete audit, and a
separate static failure audit. The PDF builder verifies evidence hashes and generates
figures and numerical tables directly from those files. See
[the primary-source comparison](../targeted_related_work.md): executable world models,
environment-guided repair, fault localization, and diagnostic feedback all predate
this study. This paper reports a controlled experiment and its failed intervention,
not a new general repair framework or a completed original multi-game plan.
