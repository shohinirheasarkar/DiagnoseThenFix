# Does targeted feedback improve repair?

Status: protocol fixed before new held-out evaluation; the signed manifest records the freeze time and hashes.

This is a fresh, narrower study. Earlier Freeway studies informed its design but
are not pooled with it. It tests diagnostic example selection, not a new code
agent or the full five-stage proposal.

## Time and ownership

Approved start: November 14, 2026, 06:11:19 UTC. Hard stop: 22:11:19 UTC.
Stop new research calls by 16:11:19 UTC. Reserve the last six hours for audits,
writing, tested PRs, PDF inspection and cleanup. Local installed models only.

The coordinator owns collection and integration. Three agents separately own
repair code, independent verification, and the paper. Separate worktrees prevent
overlapping edits. Only one model collector may run.

## Paired experiment

Use Qwen3 1.7B and 4B, thinking off. Target 20 independent initial generation
seeds per model. A timing-only pilot may reduce this to 12 before freezing.
Keep all initial generations, including invalid programs. Each initial program
has three branches: original, uniformly chosen failure feedback, and feedback
chosen from the weakest predefined physics category. Both repair branches get
two rounds; use the final candidate, never the best test result.

Both branches use the same training checker pool, prompt template, maximum
example count, output cap and generation settings. Only example selection
differs. Use distinct failing examples up to the cap, never invented failures or
passing-example padding. If none fail, issue the same preserve-function message.
Actual example counts, tokens and time can differ after repair and are reported.

Categories are defined by simulator mechanics before testing. Rank by error
rate, not failure count. Use a fixed tie order; fill remaining slots from other
failures if the weakest category has too few. Save the full checker results,
selected example IDs, category coverage and overlap between branches.

## Fresh evidence

Create episode-disjoint training, validation, frozen-test and live-test seeds.
Development uses training/validation only. Target 100 test boards, horizons 1–10,
and 100 shared live environment seeds. Never reuse old test evidence as new data.

Primary contrast: targeted minus uniform exact forecast accuracy, averaged per
initial program on the fixed case bank. Invalid outputs count as wrong. Also
report equal-board accuracy, validity and both branches versus original.

Use the same human-written parser and forecast-bound controller for every
program. Report assistance explicitly. Retain invalid programs with a fixed
stay fallback. Test the correct forecast through this controller before freezing.

Run closed-loop episodes under actual execution latency and two predefined
injected-delay conditions. Recompute trajectories, not timestamps. Preparation
cost is separate from decision-time cost. Injected delays are stress tests, not
fresh LLM calls. Actual latency measurements run without competing task workloads.

## Analysis and claims

Bootstrap paired initial programs within model; do not count individual
predictions as independent generations. Live intervals resample programs and
shared environment seeds as crossed axes. Declare primary contrasts and any
multiple-testing correction before test results. Report missing/censored records
and uncertainty; a noisy null is inconclusive. Preserve all intermediate attempts.

Four possible findings are forecast-and-play benefit, forecast-only benefit,
no clear targeting benefit, or failed repair. None establishes all-stage routing,
cross-family generalization or top-venue readiness.

## Freeze checklist

- [x] Timing pilot and fixed sample sizes.
- [x] Exact prompts, category definitions, example cap and decoding settings.
- [x] Fresh split IDs and no overlap with earlier evidence.
- [x] Controller ceiling and latency accounting checked.
- [x] Independent tests of selection, budgets, scoring and record identity.
- [x] Source commit, model digests, case hashes and analysis specification saved.
- [x] Protocol committed before held-out evaluation.

## Fixed settings and development record

The timing pilot used four development-only calls (2.4–8.4 seconds each). Fix
20 initial programs per model, 100 test boards and 100 live seeds. Generation
uses temperature0.7, thinking off, JSON format, max1024 output tokens. Repair
uses at most6 distinct failures and2 rounds. Both branches use seed800000 +
10*program_id + round; even program IDs run uniform first and odd IDs targeted
first within each round. Only the current candidate and current feedback are
shown; no accumulated chat history. A cache hit is reuse, not a new sample.

Training source seeds710000–710019 have snapshots0/3/7; validation720000–720009
and test730000–730099 use snapshot7. Live seeds740000–740099 are separate.
Every car at every horizon1–10 is scored. Snapshots hold the player at the start
while cars advance; this is a controlled diagnostic distribution, not natural
agent visitation. Primary accuracy weights forecast cases equally; equal-board
accuracy is a sensitivity analysis. Seed IDs are distinct from the prior studies.

Category order for ties: boundary_reset, fractional_motion, timer_left,
timer_right. Boundary reset overrides other labels if it occurs in the target
rollout. Categories are operational buckets, not independent causal mechanisms.
An absent category has no error rate. Invalid programs tie across represented
categories; report that subgroup and the chosen categories. Supplied rules
specify the entire transition algorithm: this tests implementation repair, not
learning unknown physics.

The fixed controller predicts three cycles of waiting d ticks and taking the
same candidate action once, then chooses the first safe action U/S/D. It uses
only the candidate forecasts, not oracle future states. Regimes are zero delay,
measured native execution (planned d=1), and two injected ticks (d=2). Native
positive execution rounds up to one whole tick; any planned/actual mismatch
is retained. Generation/preparation are offline and reported separately.
The perfect-forecast development ceiling on seeds900100–900109 was8/10 goals
at zero delay and1/10 at each delayed setting. This weak delayed controller
limits live transfer claims; forecast accuracy remains the primary endpoint.
Legacy incorrect-delay pilot controls stay separately labeled development.

For each model, report the paired targeted-minus-uniform accuracy difference,
5000 paired-program bootstrap intervals and an exact two-sided paired sign-flip
test (symmetry/exchangeability assumption), with Holm correction across the two
model tests. Other contrasts and equal-board sensitivity are descriptive.
Live intervals resample programs and environment seeds as crossed axes.
No significant result is promised. Do not discard failing baseline programs.
Incomplete blocks remain saved but prevent a complete-cohort claim.

A Codex usage limit interrupted work from about06:19 to10:54UTC. No main-study
collection happened during this gap. The approved16:11UTC research cutoff and
22:11UTC hard stop remain unchanged. Existing pilot speed still supports the
fixed sample size. No outcome-driven extensions will be made.
