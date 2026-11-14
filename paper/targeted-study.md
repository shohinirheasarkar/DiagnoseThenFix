# Does Diagnostic Targeting Improve Forecast Repair?

**Protocol draft; results pending.** This document is separate from the two earlier
studies. Final sample sizes and implementation details must match the frozen protocol.

## Abstract

A failed test tells a language model that its code is wrong. Does choosing that test
using a failure diagnosis help more than ordinary feedback? We compare two feedback
selection rules for executable Freeway forecasts. Each initial program has an
unrepaired branch, an ordinary-feedback branch, and a diagnosis-targeted branch.
The repair branches receive equal numbers of correct failure examples and equal
generation limits. We measure held-out forecasts and closed-loop gameplay using one
fixed controller. Results and conclusions will be inserted only after audit.

## Question and scope

The primary question is the extra value of targeting feedback. This is not a test
of a complete five-stage diagnostic agent. Code synthesis and counterexample-guided
repair are established methods. Our comparison isolates one choice inside that
workflow and checks its downstream consequences.

## Method

We use the locally installed Qwen 1.7B and 4B artifacts; exact identifiers, digests,
sampling settings, and software versions belong in the frozen manifest. The target
is 20 independently sampled initial programs per model, with a predefined fallback
to 12 based on runtime before test outcomes are inspected.

Every initial program is retained. Its two repair branches each receive at most two
repair rounds. Ordinary feedback samples true failures uniformly. Targeted feedback
samples true failures from the weakest predefined behavioral category. Category
rules, tie breaking, sampling, and behavior when too few failures exist must be
fixed before collection. Each round uses at most six failure examples, fewer when insufficient failures
exist. The input template and generation caps are shared. The chosen examples differ.
Both branches use the same sampling seed for a given paired round; branch order is
counterbalanced. Each prompt contains the current candidate and current feedback,
without an accumulated conversation history. Actual tokens, elapsed time, and feedback-selection cost
are reported rather than assumed equal.

The final round is evaluated even if it regresses. No best-of-test selection is
allowed. Failed execution remains a failure, not a silently excluded observation.
The frozen implementation specifies how zero-failure branches are treated.

Fresh source episodes define training, validation, and test splits. Feedback uses
training cases only. Validation is for development gates. Test outcomes do not
change prompts, sample sizes, category rules, or the controller. Earlier study data
are development context and are never pooled with this experiment.

## Evaluation

The planned frozen test has 100 held-out boards and horizons 1 through 10. The
primary outcome is exact forecast accuracy, counting invalid programs as wrong.
The primary contrast is targeted minus ordinary repair, paired by initial program.
We also report both repairs against the original program.

Each final candidate plays 100 shared fresh game seeds through the same fixed
human-written controller. We report goals, collisions, progress, invalid execution,
and fallback use. A correct forecast function passed through that controller is an
assisted reference. The controller must pass the validation gate before collection.
The generated function is a component, not an autonomous LLM policy.

Generation and repair are preparation costs. Execution is a separate deployment
cost. We record actual execution time and rerun trajectories under predefined delay
stress tests. The three regimes are zero delay, measured native execution delay, and a fixed
two-tick delay. Each game tick is one second. Native delay uses ceiling rounding,
so a positive subsecond execution still consumes one tick. This coarse time model
is a limitation: it does not resolve small speed differences. Native timing runs
are isolated from competing task workloads. Injected delays are synthetic
conditions, not measurements of LLM latency.

## Analysis

The independent model samples are initial program generations. Thousands of
forecasts and episodes do not create thousands of independent model samples.
Uncertainty must preserve the three-way pairing and account for shared test boards
and environment seeds. Per-model effects are reported; pooled summaries cannot hide
opposite effects. The primary score weights test cases equally. Board-macro accuracy is a
sensitivity analysis. Program-paired bootstrap intervals accompany sign-flip
tests; Holm adjustment covers the two primary per-model tests. Closed-loop
intervals additionally account for shared environment seeds.

Mechanism checks report the diagnosed categories, overlap between selected feedback
sets, repair validity, training-to-test changes, and regression rates. They explain
the intervention without replacing the primary comparison. Missing paired blocks
and timeouts are listed explicitly under the frozen completion policy.

## Results

Pending collection and independent audit. Planned figures: paired program-level
forecast changes; forecast accuracy by horizon; goal rates by delay condition; and
preparation versus execution costs. No synthetic fixture is research evidence.

## Discussion and limits

A forecast gain without a gameplay gain shows a transfer limitation under this
controller. Similar performance with wide intervals is inconclusive. A null result
does not prove that all diagnosis is useless. One game, one function interface, two
sizes in one family, and a human controller restrict the claim. Feedback categories
are externally defined behaviors, not measurements of internal model reasoning.

## Reproducibility

The release must contain the frozen protocol, prompts, initial and repaired programs,
raw model responses, split and seed manifests, per-case scores, replayable episodes,
analysis commands, tests, and source hashes. The final paper must state every
protocol deviation and distinguish planned from completed samples.

## References

See [the primary-source comparison](../targeted_related_work.md). Final bibliography
and claims will be checked against those sources before PDF delivery.
