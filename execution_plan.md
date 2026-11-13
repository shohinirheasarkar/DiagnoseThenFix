# What ran, and what changed

The uploaded plan is a multiweek study. These are two eight-hour local cycles.
The full plan is still unfinished. [requirements.md](requirements.md) tracks the
missing experiments. [preregistration.md](preregistration.md) records the original
hypotheses; the second cycle is an exploratory follow-up after seeing results.

## Time and access

Everything runs on the local 16 GB Apple M3 Mac. No paid model APIs or new model
downloads. GitHub PRs merge only after all CI checks pass. One model collector
runs at a time, from a frozen source snapshot.

The second cycle runs November 13, 07:39–15:39 UTC (02:39–10:39 EST). New research
calls stop at 15:05. The final window is for replay checks, the paper PDF, PRs and
cleanup. Its heartbeat must stop by 15:39. Closing the laptop can still suspend it.

The collector now checks the fixed 15:05 deadline immediately before each model
request, including nested repair calls. Calls already in flight may finish; model
unloading remains allowed. A blocked call becomes an explicit error record and
never reaches the server. Cached replies may still be reused without a new call.
Collector hashes retain their history; the frozen model-policy package stays fixed.

## Keep the studies separate

| Study | Frozen source | Saved evidence |
| --- | --- | --- |
| First model study | `f71b9259abd055291c7c865659363aa1b3d9ec0f` | `results/review` |
| Wider code grammar | `0667a194f52545e6b48edfb0efe1063979cb06c5` | Follow-up `grammar` folder |
| Follow-up context and neural v2 | `d393c4e4f473381fa11127952482264076563545` | Follow-up root model folders |
| Compiled controller v1 | `340fd713b3b1587fea6115028a402a05ddb59618` | Follow-up `compiled` folder |
| Compiled controller v2 | `556001254fcaf074dd0203753d9b76ebf1fba69b` | Follow-up `compiled/native` folder |

The follow-up archive is `results/review-overnight`. Manifests keep model digests,
exact requests, decoding settings, source fingerprints, phase times and actual
counts. A cache hit reuses the saved request's measured duration and token charge;
it is not a new independent sample. Never pool different protocols or versions.

## First cycle: closed

The bank has 500 Freeway states: 300 train, 100 validation, 100 test. Source episodes
never cross splits; each contributes at most two states. Four categories have 125
states each. This is a balanced diagnostic bank, not natural gameplay frequency.

Input selection used ten validation states, six formats, five reading probes and
one sampling seed. Forecast/consistency profiling used eight validation states.
D was unprofiled. Fixed cutoffs were A/B .8 and C/D/E .9; test results chose nothing.
Calibration uses complete first-token A/B/C scores, never invented missing scores.

The main test completed 8,000 trials: four Qwen3 artifacts, twenty held-out boards,
two forecast questions, horizons 1–10, and five decoding seeds. Thinking was off.
The 1.7B commitment cohort finished 800/800 trials; 4B stopped at 436/800.
Live base/selected/all completed ten shared seeds per model: sixty runs, zero goals,
and one extra censored 4B attempt. All 250 simulator controls finished.

Throughput changes protected coverage: at 08:36 the queue prioritized isolated
repairs and live play after forecasts; at 13:44 it switched to 4B after ten shared
1.7B seeds. Completed trials and cached replies were preserved. Collection ended
at 14:55 UTC. Broader repairs, random and monitor live comparisons stayed partial
or unrun; scheduling did not change scoring or choose favorable outcomes.

## Labels and corrections that matter

The oracle finds first actions admitting a collision-free continuation through
ten ticks. It prefers survival, then a goal, then progress. Empty safe sets remain
empty. Required foresight is the first search depth whose chosen move belongs to
that ten-tick safe set. This is finite-horizon safety, not safety forever.

Repeating one action for k ticks is a different question. Old one-tick action logs
versus repeated-action beliefs measure risk disagreement, not a sufficient logical
contradiction. The separate commitment task repeats the chosen action for three
ticks so beliefs and choices refer to the same thing. Oracle injection and shields
are assisted upper bounds.

The first frozen engine redraws the initial board after a collision on terminal
tick 100; upstream retains moved cars. Current code fixes that display. The old
records stay unchanged; only their exact declared source permits the old display
in the audit. No action follows it, so goals, collisions and elapsed ticks agree.
Shortened decision budgets now censor unfinished episodes instead of calling them
failures. The first queue's 100-decision budget cannot truncate a fresh 100-tick game.

## Second cycle: frozen probes closed, live play collecting

The wider `code_v2` grammar adds bounded conditionals and `sign`. The original
contract stays available. Its 1,237 forecast trials include 900 on 1.7B and a partial
337 on 4B. Five generated programs per model are reused; trials are not independent
program samples. Separate-call chaining is a different pilot: one request per tick,
with malformed intermediate replies counted wrong. It saved 72 trials across models.

D tests six diary sources, lengths 0/3/6/12, action-only versus outcome versions,
and five fixed repairs. Diaries are openly transplanted from other episodes while
the current board stays fixed. Synthetic outcomes are replayed. Real diaries retain
whether a proposed move was applied, including the old source's terminal display.
Outcome feedback is oracle assistance about the source episode, not the test board.

E varies labels, skins, game naming, a strategy hint, paraphrases and contact-rule
reversal. Ordinary skins leave physics unchanged; reversal changes physics and
reference labels, so it is not a pure framing contrast.

Each model targets four test boards and five decoding seeds. To protect both D and
E, D gets at most 35 minutes/model, E twenty, chaining five, all before 10:50 UTC.
This throughput-based change preserved completed records. D saved 840 trials on
1.7B and 546 on 4B; E saved 340 each. Validation saved 336 D and 64 matched C trials.
Those four-board, one-seed validation profiles are noisy. Cutoffs stayed fixed.
Worker restarts before testing preserved package fingerprints and source history.

## Neural v2 live play

V2 checks generated code on fixed training seeds 1000–1005, horizons 0–10, then
allows one feedback retry. This is simulator-assisted synthesis. Failure disables
the program and keeps model beliefs; unavailable calibration keeps the proposed
move. Generated forecasts reach the action readout even without binding, and
board-conditioned calibration receives them; the masked prior does not.

V1 did not have that corrected unbound path and excluded local program time.
V2 charges local elapsed checking/execution time as well as requests. The raw field
`cpu_seconds` means local wall time, not process CPU usage. Later error-prefix
cleanup changes future logs only; the frozen worker remains unchanged.

Fresh live seeds are 60000–60049. The queue runs base/selected/all within each seed,
1.7B first, then 4B as time allows. This condition order is fixed, not randomized.
Five simulator controls use the same fifty seeds with zero inference delay; they
are not cost-matched agents. Episodes cut short by the cap stay censored.

The paper reports program availability separately from success. Availability is
not proof of execution. A repair that never passes training checks cannot show
whether a successful code fix transfers. Final lanes include collision resets;
proposed S streaks exclude implicit thinking-time waits.

## Offline compiled pilot

All five saved programs per model are retained, including failures. Canonical
object parsing and a fixed action selector assist them. An executable-only gate
allows wrong physics; a strict fixed-training gate rejects failures and uses S.
All ten samples fail at least one training check. None is selected by test success.

Each model/version crosses five programs, fifty fresh environment seeds, two gates
and two delay charges: saved generation once per episode or the same charge each
turn. Per-turn charging is synthetic, not a fresh model call. Local execution time
is measured in both cases. Each version completed 2,000 cases with zero new model
requests. These are not independent programs or an isolated diagnosis-targeting test.

The v1 audit caught NumPy scalar inputs being rejected where plain JSON numbers
worked. Its audit now rebuilds the original typed trajectory. V2 normalizes finite
inputs and repeats the same programs/seeds/design. Both raw versions are preserved.
The concurrently frozen neural v2 worker keeps its original numeric adapter.

## Analysis and publication

Malformed replies count as wrong in accuracy. Numeric error and confidence summaries
condition on valid answers. Intervals cluster source episodes; compiled intervals
resample programs and shared environment seeds as crossed axes. Degenerate bootstrap
intervals do not prove certainty on new samples. Tiny D/E sign-flip tests and Holm
correction remain exploratory. Censored episodes stay out of completed-run means.

Shape RMSE is in-sample. A separate post-hoc analysis leaves a whole source episode
out, fits fixed line/exponential/cliff families and a flat control, clips predictions,
and scores held-out horizons. Missing horizons skip a fold. Overlapping training
folds limit descriptive intervals. This is not frozen-to-live predictive validity.

The interim archive contains closed probes, both compiled versions and controls;
neural live logs will join after collection stops. Packaging uses a fresh directory,
rejects changing inputs, and hashes every JSON evidence file exactly once. Audits
rebuild prompts/scores, cross-check saved generation sources and replay physical
trajectories against pinned upstream. CI checks both archives. The final paper must
be rebuilt from saved records and every page visually inspected.

The focused [primary-source comparison](related_work.md) makes existing code-policy,
verifier, stage-diagnosis, inertia and conversion-gap work explicit. No first/best
claim, unique internal cause, or superiority of diagnosis targeting is established.
Other-game banks, full 200/300-state matrices, extra families/thinking sweeps,
causal cascades, matched random/leave-out/monitor comparisons, latency utility rules
and conditional training remain open. Working code alone does not finish them.

## Full-bank search cross-check

A separate path search now checks all 500 Freeway states' safe sets, optimal
scores/choices and required foresight. It never calls the memoized oracle or
labeler. Failed paths stop; reaching the goal is a proven upper bound, so that
branch needs no further exploration. Transition results may be cached, but path
values are not. The saved report ties the check to bank and source-file hashes.
This checks search logic using shared replay physics; upstream differential tests
remain a separate check of physics. A node-budget limit raises an error rather
than issuing a partial certificate. No model requests are involved.

The context tables show valid answers alongside total trials. Action-label changes
also change the required output vocabulary, so their accuracy mixes decision
quality with format compliance. A condition with no valid pairs has an unavailable
flip rate, not zero evidence of an effect. Raw records and scoring stay unchanged.
