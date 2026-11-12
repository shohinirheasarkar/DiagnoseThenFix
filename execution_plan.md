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
