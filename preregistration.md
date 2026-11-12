# Before looking at model results

Registered November 12, 2026. This file must land on main before model collection.
The source plan is *Diagnose-then-Fix: Research Plan for Small-LLM Real-Time Game
Agents* (November 1, 2026). These are hypotheses, not findings.

## Questions and outcomes

- A1: lane/relation input improves perception over ASCII, especially below 8B.
  Primary outcome: paired perception accuracy; secondary: signed location error.
- B1: forecast accuracy decays faster than linear below 8B; thinking slows decay.
  Compare linear, exponential, and change-point fits by held-out squared error.
- B2: confidence stays high as horizon accuracy falls. Report ECE in ten fixed bins.
- B3: errors begin disproportionately at wraps and speed/timer transitions.
- C1: 4B/8B have the highest own-forecast/action contradiction rate.
- C2: forecast-bound selection closes most of the own/oracle injection gap when
  forecast accuracy at required foresight exceeds 70%. Report raw differences too.
- D1-D3: passive diaries raise stay rate; inertia grows with context and does not
  decrease with size; outcomes reduce harmful inertia.
- E1-E3: null-state calibration reduces passive loops; skins shift actions more
  than forecasts; naming the game helps at 4B and above.
- X1-X3: critical steps cluster in the first quarter; history dominates cascades
  below 8B; the monitor improves success at matched or lower measured cost.
- H4: held-out frozen diagnosis predicts the best live repair and live success.

## Data and splits

Use RealtimeGym at commit 3d5b3ef37dafee25ca2e8d2e8d71f5fe842eda7b.
Do not silently replace its dynamics with modulo motion. Freeway is primary;
Snake and Overcooked are separate replications. A moving-hazard fallback must be
named explicitly, never called Overcooked. Training repairs are compute-conditional.

Target 500 unique states per game, balanced across safe advance, unsafe advance,
wait helps, and retreat required where those categories have meaning. Report
achieved counts and exhaustion. Keep source episode seeds disjoint across
train/validation/test (60/20/20), deduplicate states, and never tune on test.
Use 200 states for A and 300 for B-E, sampled only within the relevant split.
Report any smaller local pilot as exploratory, not as completion of this matrix.

Qwen3 0.6B/1.7B/4B/8B/14B, thinking on/off; second family and larger ceiling
reported separately. Freeze exact model digests, quantization, prompt text,
three paraphrases, decoding settings and software revision in run manifests.
Five sampling seeds (0-4), forecast horizons 1-10. Log malformed responses,
timeouts and truncations; include them as incorrect in the primary denominator.
Never score a repaired malformed response as an unrepaired model success.

## Oracle definitions

A safe first action admits a collision-free continuation through the stated
horizon; it is not a claim of indefinite safety. A progressing action also admits
a goal-reaching continuation. A repeated-action collision probe is a distinct
quantity and must be labeled as such. Finite-horizon dead ends have empty safe
sets, not invented safe actions. Required foresight is the first depth whose
planner choice belongs to the reference horizon's safe set; also report the
reference horizon and cases with no safe action or no such depth. Compare
available foresight at 80% against this explicitly finite-horizon quantity.
Do not call a shallow planner an exact long-horizon oracle.

## Controls and analysis

Random, stay, advance, constant-velocity extrapolation, finite-depth planner and
reference oracle. Compare isolated repairs, all repairs, diagnosis-selected,
random same-count repairs, leave-one-out, and monitor. Oracle forecasts/shields
are upper bounds, excluded from headline model-only comparisons. Add generic
"think carefully" prompts at comparable token budgets.

Select renderers and repair thresholds on validation only, then freeze them.
Bootstrap 95% intervals over episode/state clusters (2,000 replicates), preserving
paired conditions and repeated samples. Seeds are not independent situations.
Report all planned contrasts with Holm correction within stage; report effect
sizes and contradictions. Live target: 50 episode seeds per condition. Replay
measured delay with a declared tick duration and default action, including
terminal events while reasoning; count all calls, tokens and retries.

## Local execution and deviations

Only the user's local machine is authorized; no paid APIs. Start with simulator
validation and a small throughput pilot. Full-model runs are sequential to fit
16 GB RAM. Missing families, training, ceiling checks or unfinished runs remain
explicitly pending. Do not claim novelty until the cited literature is checked.
A paper may report completed pilot results but must identify its scope and avoid
invented significance, citations or empirical claims.
