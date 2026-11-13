# November 12 local cycle

The user capped this cycle at eight hours, including implementation. The full
multiweek plan remains the target; this cycle is an exploratory, time-capped
Freeway study, not completion of every planned experiment. Earlier tiny runs
under results/local-pilot are engineering pilots and are not pooled with this run.

The main queue freezes at its launching commit. Run one local Q4_K_M model at a
time: Qwen3 1.7B, 4B, 8B, 14B. No paid services. Preserve every completed trial.
Stop model collection by 14:40 UTC to leave time for the paper, checks and PRs.

Priority order:

1. Choose a renderer on 10 validation states, six formats, five probes, seed 0.
   Profile forecasts on 8 validation states at k=1,5,10; consistency on 8 states.
   Calibration uses complete first-token A/B/C log probabilities, never imputed
   missing candidates. These are conditional action scores, not unconstrained
   action frequencies. The profile uses fixed preregistered thresholds; it does
   not optimize thresholds on test outcomes. D is explicitly unprofiled.
2. Forecast test: 20 held-out states, horizons 1-10, two question types, seeds 0-4,
   each of four models, thinking off. This is the main result if it completes.
3. Consistency test (10 states) and isolated forecast/consistency repairs (5 states),
   five seeds. The oracle shield uses the bank's reference horizon of 10, while
   repeated-action collision beliefs use the prompt's declared horizon.
4. Live base/selected/all/random/monitor, 50 fresh episode seeds each, for 1.7B
   and 4B. The world advances by ceil(measured inference seconds / 1 second),
   taking S while thinking, then applies the pending action if time remains.
5. Simulator-only baselines: random, stay, advance, shallow planner and oracle,
   50 fresh seeds. Budget-expired partial live episodes are censored, not failures.

The queue can stop before completing these items. Report actual counts, model
hashes, malformed replies, code failures, unavailable calibration and censored
runs. Raw program-generation costs are included when paid, and reused programs
are counted once per episode. No cross-model comparison is complete unless its
paired states and conditions are present.

The 500-state bank has 125 states per category and 300/100/100 splits. At most
two states come from an episode. Summary CIs resample source episodes; a probe
repeat is not a new independent observation. Other games have tested snapshot
adapters, but no completed diagnostic replication. Thinking sweeps, the 0.6B and
second-family models, three-paraphrase robustness, full diary/framing tests,
leave-one-out live comparisons, cost matching, curve-shape confirmation,
counterfactual causal attribution and conditional fine-tuning remain pending
unless separately completed and documented. Do not turn those into paper claims.

## Queue adjustment at 08:36 UTC

The observed 4B throughput threatens the live-play time budget. Preserve all
frozen model code, cases, requests and decoding settings in a source snapshot
of merge commit f71b9259abd055291c7c865659363aa1b3d9ec0f; its fingerprint was
verified against the running manifests. Resume completed trials without repeats.
After all four main forecast runs, prioritize C tests and base/code/tool forecast
contrasts plus base/generic/bound consistency contrasts on 1.7B and 4B. Then run
base/selected/all live conditions, alternating conditions within each shared
seed rather than finishing one whole condition first. Broader repair sweeps
follow only if time remains. Random/monitor empirical comparisons are now
optional, and remain pending if not run. This scheduling change uses throughput,
not test outcomes, to protect the frozen-to-live comparison.

The existing `contradiction` log field is a repeated-action risk-disagreement
proxy: it compares a one-tick chosen action with a forecast about repeating that
action. It is not a logically sufficient belief-action-gap metric. Report it
under that narrower name and leave commitment-matched gap claims open.

## Separate commitment diagnostic

A follow-up diagnostic corrects the temporal mismatch rather than changing the
legacy logs. Each action is explicitly repeated for three ticks; beliefs and
choices refer to that same commitment. It uses twenty held-out states, five
sampling seeds and eight conditions: no injection, own forecast, oracle forecast,
combined answer, generic check, bound selection, one retry, and oracle shield.
Oracle injection and shielding are upper bounds. Record unsafe choices, skipped
believed-safe progress, and failure to abstain separately. Queue this after the
main batch, one local model at a time, stopping by 14:55 UTC. Any partial run keeps
its actual condition counts. This correction is motivated by task semantics,
not by a favorable observed result.

The report also includes descriptive linear/exponential/cliff fits, valid-answer
cell errors, and paired live changes on shared uncensored seeds. In-sample fit
error does not confirm the preregistered curve-shape hypothesis. The extra cliff
parameter is explicitly disclosed. No adjusted significance or causal claim is
made from these exploratory summaries.

## Terminal-state correction during the evidence audit

An audit found that the frozen replay redraws the initial board after a collision
on tick 100. Upstream retains the moved cars on that terminal tick. The current
engine is corrected and tested against upstream for both sides of this boundary,
all actions, and both slow/fast directions. The frozen study keeps its original
source and logs. Success, elapsed ticks, collisions, and all subsequent decisions
are unchanged because no action follows a terminal state. The offline audit uses
this display convention only when cycle.json declares the exact frozen source
commit above; other sources must match the corrected engine. Do not silently
rewrite the raw terminal boards or call them literal upstream observations.

The current harness also censors episodes stopped by a shortened decision budget
and preserves outcomes when an episode starts in a terminal state. The frozen
published queue uses 100 decisions with at least one world tick per decision,
so that budget cannot truncate a fresh episode before the game's 100-tick limit.
Its collected labels do not change. The audit checks censoring and final-lane
summaries against the replayed state as well as checking success and costs.

## Live coverage adjustment at 13:44 UTC

Observed episode throughput would leave the entire remaining live window on
1.7B. Finish its tenth shared seed, then prioritize the already-planned 4B
conditions. The collector resumes with 4B first in the model queue; the four
forecast tests are already complete and are not changed or repeated. The current
interrupted seed is reconstructed with the same frozen policy and cached raw
replies before switching models. Completed records are retained. Conditions,
cases, decoding, seeds and scoring are unchanged; only scheduling changes. Keep
the 50-seed targets and report the actual partial counts. The reason is model
coverage within the fixed time cap, not selection on a favorable live outcome.

## Closed local cycle

The four-model main forecast matrix completed 8,000 held-out trials. Each of the
two live models completed ten shared seeds in base/selected/all: 60 completed
model-condition runs, zero goals, and one additional censored 4B base attempt.
All 250 simulator-control episodes completed. Commitment trials completed at 800/800 for 1.7B (20 boards) and 436/800 for 4B
(11 boards).
Collection stopped at 14:55 UTC. The paper and lossless archive report these
actual cohorts; the full multi-game research plan remains unfinished.

## November 13 follow-up

The user authorized eight more local hours. See [the requirements checklist](requirements.md)
for the full plan audit and the second cycle's fixed deadline. New studies are
exploratory follow-ups. The old arithmetic contract remains the default; an
explicit `code_v2` condition adds bounded conditional expressions and `sign`.
This fixes an interface limitation, not the model's understanding of motion.

### Paired context follow-up

Stage D now has six diary sources, lengths 0/3/6/12, action-only versus
observed-outcome versions, and five repairs. Diaries are explicitly transplanted
from other episodes. This holds the current board fixed without pretending those
old actions happened on that board. Synthetic outcomes are replayed, not guessed.
Real diaries preserve whether the model's proposed action was actually applied.
The outcome repair includes oracle feedback about the source board and is labeled
as assisted feedback. Delta summaries concern the source episode, not a hidden
previous frame of the test board. These choices limit claims about natural live
memory, but make the controlled manipulation honest.

Stage E pairs three label sets, four skins, game naming, a strategy hint, three
paraphrases and a small contact-rule reversal. Reversal changes the scored physics;
ordinary skin and label changes do not. The paired analysis reports invalid replies,
all-trial accuracy differences, valid-pair action/forecast flips, and safe-to-unsafe
flips. Intervals resample source episodes. Cluster sign-flip tests and Holm correction
are exploratory; small cohorts have very limited power.

### Separate-call forecasts and recoverable repairs

`chained_calls` is a new position-only condition: one request per tick, with earlier
answers in later context. The old `chained` condition remains a one-reply table.
Malformed intermediate replies stop the chain and count as invalid. Every request
and its cost is saved; cached prefixes remain logical reused work.

The opt-in composed policy `revision='v2'` accepts the wider arithmetic contract.
It checks each generated program on fixed initial boards from training episode
seeds 1000-1005, at horizons 0-10. It supplies the first failing training example
for one retry. This is simulator-assisted code validation, not test-time oracle
forecasting. Passing those checks does not prove a program correct everywhere.
If code still fails, it is disabled for that episode; the original model forecast
remains available. Missing calibration scores also leave the model action intact.
Failures and every generation/retry/scoring call remain logged. The default v1
policy and the first cycle stay unchanged. Never pool the two policy versions.

A wiring audit then found that code forecasts could be computed but unused when
binding was off. In v2, that branch now asks the model to act on the generated
forecast. Calibration's board-conditioned scores also receive the current forecast;
the masked-board prior receives no state or forecast. This preserves a meaningful
leave-binding-out comparison. The added action request is charged and logged.
The old v1 evidence is not rewritten or presented as having this corrected path.

### Fixed overnight queue

The first follow-up collector uses the merged arithmetic-contract source frozen
at 0667a194f52545e6b48edfb0efe1063979cb06c5. It stops by 08:30 UTC and keeps
partial model cohorts. The next queue waits for it, so models never compete for
RAM. It collects four validation boards (one decoding seed) for diaries and matched
commitment consistency before selecting v2 repairs. D is the worst diary-condition
fraction with valid paired answers and no safe-to-unsafe flip; C is matched
own-forecast consistency, with invalid replies scored zero. Existing fixed cutoffs
stay unchanged. This small follow-up profile is exploratory and noisy.

The context target is four held-out boards, all defined conditions, five decoding
seeds, on both local models. This smaller balanced matrix protects coverage of
both D and E within the measured local throughput. Extra separate-call forecast
checks follow if time remains. Context work stops at 10:50 UTC. Live work then
prioritizes 50 seeds each for base/selected/all on 1.7B, then 4B if time remains.
New live seeds are 60000-60049, disjoint from the earlier live diaries used as
context. Five simulator controls use those same fresh seeds. Collection stops at
15:05 UTC, with the overall 15:39 UTC deadline unchanged. Unfinished cells stay
visible; this schedule does not turn four boards into the planned 300.

A local thread heartbeat (`finish-the-eight-hour-research-cycle`) checks progress
every twenty minutes and continues implementation and paper work. It must pause
itself at the hard stop. The worker is `scripts/overnight.py`; it runs from a fixed
source snapshot, permits one collector per output, rejects changed sources on
resume, and saves actual completed/censored episodes. The heartbeat must inspect
that worker before launching any additional model work. It should be quiet while
collection is healthy and no action is needed.

The context evidence audit now rebuilds the exact prompt, verifies model digest
and sampling seed, replays both synthetic and real diary outcomes (including
thinking delays), and checks oracle feedback. Some old diary terminal frames use
the first cycle's display convention. Their exact old source revision is carried
into the new records, and only that revision permits its terminal-display rule.
The waiting worker was restarted before any follow-up trials to include this
provenance; no collected model results were deleted or changed.

### Paper follow-up support

The review-paper builder now accepts a separate follow-up folder. It keeps grammar
comparisons, separate-call forecasts, held-out diary/framing tests, and v2 live
play apart from the first study. It reports partial denominators and valid-pair
flip rates. Duplicate completed episodes and mixed v1/v2 records are errors.
Validation diagnostics never enter the held-out context table. All conditions and
exploratory adjusted tests stay in evidence JSON; the paper shows compact tables.

### Protecting coverage at 08:35 UTC

Measured context throughput could let diary probes consume the whole context
window. Keep the same four-board/five-seed targets and every completed record, but
cap each model's D phase at 35 minutes, E at 20, and the chained-call pilot at five.
The global 10:50 context cutoff still wins. This is a scheduling change based on
throughput, not answer quality. A phase can be partial. The worker records phase
start/end/cutoff times and its own script hash. Source-commit history is retained;
unchanged package fingerprints and trial manifests let validation resume exactly.
The worker restarts once to apply this budget before any held-out test phase.

### Saved-program latency pilot

All ten saved `code_v2` samples (five per model) fail at least one fixed training
check. Before any controller test, freeze a separate exploratory replay: use all
five samples, never choose them by held-out accuracy. Compare the bounded
interpreter alone with a strict training-check gate. The gate rejects any failing
program and uses S; executable programs can still have wrong physics. Unsafe
Python remains blocked in both conditions. No extra generation or repair is used.

Cross each gate with two delay charges: saved generation time once per episode,
or that same time every turn. The latter is a synthetic repeated-charge control,
not a new measured model call. Both include measured local checking/execution
costs. Replays make zero model-server requests. They use five programs by fifty
fresh environment seeds (60000-60049) per condition per model; repeats of a saved
program are not independent programs. The controller reads canonical object
facts and uses a fixed U/S/D selector over its code forecasts. This removes input
parsing and action readout work; it is not an isolated cognition or targeting test.
It also does not solve delay-aware planning or the full latency utility rule.

Keep this `compiled-controller-v1` cohort separate from v2 live play. Save every
program failure, raw source response, CPU duration and logical generation charge.
Audit both the program's chosen moves and every physical tick against upstream.
The published archive must include the original generation logs for cross-checks.

### Numeric boundary correction after the first replay audit

The first compiled replay completed 2,000 cases. Its independent audit caught a
mismatch: real game states contain NumPy scalar numbers, but JSON-restored states
contain plain Python numbers. The old arithmetic interpreter rejected some
integer-valued NumPy results. This was an input-adapter bug, not proof that a
program's physics was wrong. Keep all `compiled-controller-v1` records unchanged.
Their audit now rebuilds the original typed simulator trajectory before evaluating
programs, and all 2,000 cases pass that exact replay.

The interpreter now converts finite real inputs to plain bounded numbers. Existing
forecast-bank inputs were already plain numbers; original evidence remains checked.
A new `compiled-controller-v2` repeats the same programs, seeds, gates and delay
conditions after this fix. Never pool its results with v1. The frozen neural v2
collector retains its original package; this fix does not silently change that
running study. Core input normalization and the saved-program replay are separate
from a claim about an LLM's internal forecasting skill.

### Reporting the saved-program pilot

The paper keeps both adapter versions and all four gate/delay conditions separate.
Intervals resample five generated programs and shared environment seeds as two
crossed axes. Balanced estimates require all five programs on common completed
seeds; partial counts stay visible. Per-program intervals and paired contrasts
are saved in evidence JSON. The parser and selector are human-provided assistance,
and repeated generation delay is a synthetic charge. No new model call occurs.
The analysis rejects changed source programs, duplicate completed cases, or fresh
requests disguised as offline replay. This remains an exploratory pilot.

### Exploratory held-out curve prediction

The original shape RMSE stays labeled in-sample. A separate post-hoc analysis now
leaves each source episode out before fitting the fixed line/exponential/cliff
families and a flat control. Predictions are clipped to probabilities. Missing
horizons skip a fold rather than being filled. Training uses equally weighted
saved trials; test error averages horizons and then source episodes. Fold membership,
predictions, Brier scores and descriptive episode-bootstrap intervals are saved.
Overlapping training folds limit those intervals. This does not establish the
planned preregistered curve hypothesis or frozen-to-live predictive validity.

### Signed forecast-error plot

The paper now plots coordinate bias at each horizon, using valid numeric position
answers and source-episode bootstrap intervals. Malformed replies still count as
wrong in accuracy, but have no numeric error. Zero mean error can hide opposite
mistakes. This is ordinary x-coordinate error, not circular distance or progress
along travel; wrap can create large jumps. Missing horizons stay blank.

### Archive completeness guard

Packaging now refuses a nonempty destination and detects source files changing
while copied. The audit checks that every JSON evidence file is listed exactly
once in the hash manifest. This prevents an old leftover file from quietly entering
a new paper cohort without its own manifest entry. Use a fresh archive folder.

### Stable failed-repair logs

A disabled code repair was adding the same prefix to its saved failure on every
later turn. New code keeps the original failure and adds that prefix once in the
turn's message. It still makes no further generation attempts, and keeps the same
fallback beliefs and action. The running study's frozen code and raw messages
remain unchanged. Repeated prefixes in those records are a logging issue.

### Primary-source comparison

A focused comparison now covers all reading-list topics from the uploaded plan,
plus RealtimeGym's own code-policy baseline. It separates existing ideas from this
cycle's narrower evidence. Saved code policies, verifier feedback, stage separation,
history bias and static-to-live gaps are not claimed as new. The paper adds the
missing closest-work citations; related_work.md gives direct primary-source links.
This is a focused audit, not an exhaustive proof of priority.

### Closed-cohort evidence snapshot at 11:05 UTC

The follow-up archive now saves the closed frozen probes, both compiled replay
versions, original program-generation logs, and fresh simulator controls. Open
neural live logs are excluded from this interim snapshot. CI replays the new
archive as well as the first study. Final packaging will use a fresh staging
archive and replace this snapshot consistently after collection stops; it must
keep the same closed raw records and add the actual live outcomes.
