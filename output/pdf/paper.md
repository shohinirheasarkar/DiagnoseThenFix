Diagnose, Then Repair

A simulator-checked study of small language-model agents in real-time Freeway

Research paper for review | November 12, 2026 | Time-capped local study


## Abstract

A correct action in a changing world requires a readable observation, an adequate forecast, and a decision that follows that forecast. We implement a stage-oriented diagnostic suite with replayable states, six renderers, simulator-checked labels, and targeted inference-time repairs. This report analyzes 8,000 held-out forecast trials from 4 installed Qwen3 model artifacts collected on a 16 GB Apple M3 laptop. The situation bank contains 500 states with disjoint source episodes across splits. We compare forecast accuracy at horizons 1-10 against constant-velocity extrapolation, and separately evaluate consistency repairs and measured-delay live play where completed. Episode-weighted exact-head accuracy spans 6.9%-15.6% across the collected artifacts. Live collection produced 60 completed model-condition runs over 10 environment seeds (0 goals) and 1 censored runs. Missing or malformed answers count as incorrect. This is an exploratory, single-game artifact; the full multi-game, thinking-mode, and training-based study remains incomplete. We make no first-of-its-kind or causal root-cause claim from this cycle.


## 1. Motivation and related work

RealtimeGym makes reasoning time consequential: the environment can change before a model finishes deciding [1]. We build on that simulator, rather than proposing a new game benchmark. Our question is whether the stage that appears weak on frozen states points to a useful repair during live play.

The decomposition itself is established. Belief-action work distinguishes represented information from its use in decisions [2]; AgentDebug studies modular and cascading failures [3]. Conversational inertia motivates controlled history interventions [4]. Text-world simulation studies motivate explicit transition probes [5], while executable world models motivate computational offloading [6]. Semantic inertia motivates label and framing controls [7]. Those constructs guide the implementation; this report does not rebrand them as new contributions.


## 2. Simulator and reference labels

We pin RealtimeGym to commit 3d5b3ef37dafee25ca2e8d2e8d71f5fe842eda7b. Freeway has nine columns and ten lanes; the player stays in column 4. Its actual motion uses per-object timers, car lengths, and a separate wrap tick. A modulo update alone is therefore not simulator ground truth. Our immutable replay engine is checked against original-game trajectories, including collisions and board resets.

The reference safe set contains first actions admitting a collision-free continuation through ten ticks. An empty set is retained. The optimal set first prefers survival, then reaching the goal, then forward progress; ties remain valid alternatives. Required foresight h* is the first search depth whose chosen action belongs to the ten-tick safe set. This is finite-horizon safety, not an indefinite guarantee. The collision probe asks about repeating one action, a distinct quantity from existence of a safe continuation. We never interchange those labels.

The bank has 500 unique situations: 125 in each of safe advance, dangerous advance, waiting helps, and retreat required. Waiting helps requires that waiting is reference-safe and avoids a collision caused by immediate advance; retreat required means D is the sole safe action. Splits contain 300/100/100 states. Each episode contributes at most two states; source episode counts are 152/50/51. These categories are sampling strata, not estimates of how often each situation occurs in ordinary play.


## 3. Experimental protocol

We registered hypotheses before model collection. A subsequent execution plan records the user-imposed eight-hour cap and its reduced scope. Renderer selection uses ten validation states, six representations, five perception probes, and sampling seed 0. Forecast and consistency profiles use eight validation states each. The main forecast test targets twenty held-out states, two question types, ten horizons, and five sampling seeds per model. The forecasted head is object 0; object coverage is a limitation. No test result chooses the renderer or the repair profile.

The installed artifacts belong to the Qwen3 family [9]. Sizes in tables are nominal Ollama tags; exact digests and reported model metadata are retained. Thinking is off, temperature is 0.7, context limit is 8,192 tokens, and the answer budget is 512 tokens. Exact prompts, model digests, quantization metadata, raw replies, timing and software fingerprints are logged. Requests are cached by all decoding settings and model identity. Malformed outputs, truncations, and timeouts remain in the primary denominator. Calibration error is conditional on valid confidence reports. Accuracy and mean confidence weight source episodes equally; ECE pools valid trials. Intervals are percentile 95% bootstrap intervals over source episodes, with 2,000 resamples; sampling repeats do not become independent boards. Boundary estimates can yield degenerate resampling intervals, which do not establish certainty on new states.

Validation representation screen (single sampling seed, ten states, five probes). These selection scores are not held-out evidence that one format generalizes.

Model tag | ASCII | Coordinates | Prose | Lanes | Relations | JSON

--- | --- | --- | --- | --- | --- | ---

1.7b | 36.0% | 40.0% | 34.0% | 6.0% | 28.0% | 30.0%

4b | 62.0% | 60.0% | 46.0% | 24.0% | 42.0% | 56.0%

8b | 44.0% | 46.0% | 48.0% | 22.0% | 16.0% | 50.0%

14b | 58.0% | 60.0% | 62.0% | 40.0% | 52.0% | 62.0%


## 4. Foresight measurements

The position task asks for an exact head column after k car ticks. The collision task asks whether repeatedly advancing causes any collision within k ticks. The linear control predicts round((x + k/p) mod 9), where p is the simulator speed parameter, then applies the same endpoint collision geometry. It intentionally ignores timer phase and delayed wrap. It is a declared cheap baseline, not an exact physics model. Random controls are uniform over the 17 possible head columns (-4 through 12) and over two collision labels. An always-collision control exposes label imbalance.

Model tag | Boards / episodes | Trials | Position accuracy [95% CI] | Collision [95% CI] / always collision | Invalid

--- | --- | --- | --- | --- | ---

1.7b | 20/18 | 2000 | 10.3% [6.4, 14.4] | 15.6% [3.9, 30.6] / 84.4% | 0

4b | 20/18 | 2000 | 15.6% [9.4, 22.8] | 53.0% [45.2, 60.2] / 84.4% | 0

8b | 20/18 | 2000 | 6.9% [3.8, 10.3] | 15.6% [3.9, 30.6] / 84.4% | 0

14b | 20/18 | 2000 | 12.1% [4.0, 22.8] | 15.6% [3.9, 30.6] / 84.4% | 1

On the sampled held-out boards, episode-weighted position accuracy ranges from 6.9% to 15.6% across the collected artifacts. These are aggregate descriptive results for the selected input formats, not estimates over a natural gameplay distribution.

The declared linear control has higher point accuracy in 40 of 40 collected model-by-horizon position comparisons. This count does not test statistical significance and does not explain which internal stage produced the errors.

Intervals in the table aggregate across horizons. Per-horizon curves below are the primary diagnostic. A periodic or nonmonotone curve is not automatically evidence of compounding error: wrap and timer phase can make individual horizons easier. The two foresight summaries are the largest horizon reaching 80% accuracy and the contiguous prefix that stays above 80%; they answer different questions. Curve-shape confirmation and multiple-model thinking comparisons are not completed in this cycle.

Figure 1. Forecast accuracy by horizon. Shading shows episode-cluster bootstrap intervals. Each panel uses the actual completed trials; inspect the evidence JSON for per-horizon counts. Oracle predictions define the horizontal ceiling. The linear control is computed on the same states.

Model tag | Selected input | Largest k at 80% | 80% prefix | Median h* | Prefix covers h*

--- | --- | --- | --- | --- | ---

1.7b | coordinates | 0 | 0 | 1 | 0.0%

4b | ascii | 0 | 0 | 1 | 0.0%

8b | json | 0 | 0 | 1 | 0.0%

14b | prose | 0 | 0 | 1 | 0.0%

Model tag | Mean valid confidence | Position ECE | Collision ECE

--- | --- | --- | ---

1.7b | 98.2% | 88.2% | 82.3%

4b | 96.6% | 81.2% | 43.5%

8b | 96.2% | 90.7% | 81.7%

14b | 97.0% | 87.9% | 81.0%

Confidence and ECE use only valid self-reports; primary accuracy includes every trial. ECE uses ten fixed confidence bins. ECE is descriptive and does not establish a cognitive mechanism.

Model tag | Signed error | Absolute error | Linear RMSE | Exponential RMSE | Cliff RMSE

--- | --- | --- | --- | --- | ---

1.7b | -0.90 | 2.68 | 0.080 | 0.040 | 0.041

4b | 0.60 | 2.42 | 0.064 | 0.064 | 0.040

8b | 2.70 | 3.32 | 0.048 | 0.050 | 0.046

14b | 1.57 | 3.30 | 0.045 | 0.045 | 0.042

Cell errors condition on valid numeric position answers. Shape fits use the episode-weighted position means: ordinary least-squares line, a*exp(b*k) with b on a fixed [-1,1] grid at 0.005 spacing, and two constant levels with one breakpoint. RMSE is in-sample accuracy error. The cliff has an extra selected parameter; lower RMSE is not evidence that it generalizes or that compounding error caused the curve. No confirmatory shape test is claimed.

Model tag | Sequences | Timer wait | Timer move | Fast step | Wrap tick | No error

--- | --- | --- | --- | --- | --- | ---

1.7b | 100 | 16 | 59 | 20 | 5 | 0

4b | 100 | 52 | 23 | 20 | 5 | 0

8b | 100 | 51 | 24 | 20 | 5 | 0

14b | 100 | 50 | 20 | 20 | 10 | 0

Earliest incorrect direct-forecast horizon, grouped by the actual object-0 transition at that horizon. A sequence is one board and sampling seed with all ten horizons; repeats are not independent boards. These are separately asked predictions, not one chained rollout. Counts do not adjust for event exposure and cannot establish that wrap causes errors. Freeway objects keep their lane and direction, so direction-change onset is not an event in this simulator. Incomplete sequences are excluded here and retained in accuracy.


## 5. Targeted repairs and consistency

Implemented forecast conditions include direct answers, rule reminders, scratchpad tables, chained tables, model-written code, and a supplied deterministic forecast. The table reports only completed contrasts. Code is interpreted in a bounded arithmetic subset with no imports, attributes, or arbitrary Python execution. Invalid programs count as wrong. The supplied forecast and simulator shield are upper-bound conditions, excluded from model-only claims.

Model | Position repair | Boards / trials | Accuracy [95% CI] | Paired change [pp] | Invalid

--- | --- | --- | --- | --- | ---

1.7b | base | 5 / 75 | 17.3% [4.0, 30.7] | 0.0 [0.0, 0.0] | 0

1.7b | code | 5 / 75 | 10.7% [0.0, 21.3] | -6.7 [-13.3, -1.3] | 67

1.7b | tool | 5 / 75 | 100.0% [100.0, 100.0] | 82.7 [69.3, 96.0] | 0

4b | base | 5 / 75 | 25.3% [6.7, 44.0] | 0.0 [0.0, 0.0] | 0

4b | code | 5 / 75 | 8.0% [0.0, 16.0] | -17.3 [-42.7, 5.3] | 67

4b | tool | 5 / 75 | 100.0% [100.0, 100.0] | 74.7 [56.0, 93.3] | 0

Repair intervals are exploratory and unadjusted; no significance claim is made. The paired change column is candidate minus base in percentage points. The small repair bank limits external validity. Intervals condition on the five decoding seeds; cached programs are reused across states rather than counted as fresh program samples. Base and repaired trials are paired on the same source episodes, target horizons and sampling seeds where available.

The restricted-code condition has 134 invalid executions out of 150 reported position trials. Program reuse makes these correlated failures. The contrast measures compatibility with the declared grammar as well as program specification; it cannot settle the usefulness of more permissive forecast offloading.

Consistency probes separate a forecast-only prompt from a fresh action prompt. We re-ask actions with no forecast, the model forecast, and the oracle forecast. The reported contradiction event is choosing an action the elicited forecast labels colliding under repetition. Because live choices apply for one tick, this is a repeated-action risk-disagreement proxy, not a logical contradiction or the formal belief-action gap. Commitment-matched results, if completed, are reported separately below. The proxy also does not capture every instance of skipped safe progress. Forecast-bound selection deterministically prioritizes U, S, D among actions the model marks noncolliding. It removes that measured contradiction by construction where its beliefs admit a noncolliding action; it cannot make an incorrect forecast true.


## 5.1. Commitment-matched consistency

A separate diagnostic explicitly commits the chosen action for exactly three ticks without revision. Collision or deadline failure makes the commitment unsafe; a goal ends it successfully. The preferred choice is U if safe, then S, then D; if none is safe, the required output is null. The forecast refers to that same commitment. We count unsafe choice when the elicited beliefs admit a safe action, skipped believed-safe progress, and failure to abstain when all commitments are believed unsafe. This fixes the temporal mismatch of the legacy proxy, while remaining a behavioral report-based measurement rather than a causal mechanism test.

None uses a fresh action prompt with a separately elicited forecast for comparison; own/oracle inject model/reference forecasts. Combined jointly reports action and forecast, and generic adds a check instruction. Bound selects from model beliefs. Retry makes one extra action call after an unsafe choice or failed abstention, retaining the initial beliefs. Shield applies the priority selector to oracle beliefs, making it a construction ceiling. Action accuracy enforces the full preference rule; inconsistency reports the three events defined above. This follow-up diagnostic does not refit the already-frozen live policy.

Model tag | Condition | Boards / trials | Action accuracy [95% CI] | Forecast accuracy | Inconsistency, valid records | Invalid

--- | --- | --- | --- | --- | --- | ---

1.7b | none | 20/100 | 16.7% [0.0, 33.3] | 61.5% | 0.0% [0.0, 0.0] (n=100) | 0

1.7b | own | 20/100 | 17.8% [2.2, 35.6] | 61.5% | 3.3% [0.0, 8.9] (n=100) | 0

1.7b | oracle (UB) | 20/100 | 16.7% [0.0, 33.3] | 100.0% | 83.3% [66.7, 100.0] (n=100) | 0

1.7b | combined | 20/100 | 16.7% [0.0, 33.3] | 61.1% | 0.0% [0.0, 0.0] (n=100) | 0

1.7b | generic | 20/100 | 16.7% [0.0, 33.3] | 60.4% | 0.0% [0.0, 0.0] (n=100) | 0

1.7b | bound | 20/100 | 16.7% [0.0, 33.3] | 61.5% | 0.0% [0.0, 0.0] (n=100) | 0

1.7b | retry | 20/100 | 16.7% [0.0, 33.3] | 61.1% | 0.0% [0.0, 0.0] (n=100) | 0

1.7b | shield (UB) | 20/100 | 100.0% [100.0, 100.0] | 100.0% | 0.0% [0.0, 0.0] (n=100) | 0

4b | none | 11/55 | 50.9% [23.6, 78.2] | 61.2% | 96.4% [89.1, 100.0] (n=55) | 0

4b | own | 11/55 | 18.2% [7.3, 32.7] | 63.0% | 36.1% [16.7, 58.3] (n=21) | 34

4b | oracle (UB) | 11/55 | 5.5% [0.0, 12.7] | 100.0% | 86.4% [68.2, 100.0] (n=50) | 5

4b | combined | 11/55 | 54.5% [27.3, 81.8] | 38.8% | 100.0% [100.0, 100.0] (n=55) | 0

4b | generic | 11/54 | 54.5% [27.3, 81.8] | 46.7% | 100.0% [100.0, 100.0] (n=54) | 0

4b | bound | 11/54 | 18.2% [0.0, 45.5] | 61.4% | 0.0% [0.0, 0.0] (n=54) | 0

4b | retry | 11/54 | 54.5% [27.3, 81.8] | 38.8% | 100.0% [100.0, 100.0] (n=54) | 0

4b | shield (UB) | 11/54 | 100.0% [100.0, 100.0] | 100.0% | 0.0% [0.0, 0.0] (n=54) | 0

UB marks oracle information or a simulator shield. Bound selection uses model beliefs only; zero inconsistency is a construction property and does not imply correct dynamics. Invalid outputs count against primary accuracy; inconsistency and mean per-action forecast accuracy condition on valid action/forecast records. The parenthesized n is that conditional record count. Separate component rates are retained in the evidence JSON. All action and forecast prompts use the same declared commitment horizon.


## 6. Composed agent and live validation

The composed agent selects a representation and repairs from validation diagnostics using fixed stage cutoffs: 0.8 for A/B and 0.9 for C/D/E. A is ASCII perception accuracy, B is pooled forecast accuracy, and C is one minus the legacy risk-disagreement rate. D is unprofiled and set to 1. E is set to 0.5 when the largest null-state conditional action probability exceeds 0.6, otherwise 1. A structured-input flag keeps ASCII when ASCII wins validation. Forecast code and forecast-bound selection use model beliefs. Prior calibration subtracts null-state from observed-state action log scores, following the contextual-calibration idea [8] and requiring all A/B/C candidates. A separate one-token action interface is used for those scores, which is a prompt-interface confound for causal attribution. History is explicitly unprofiled in this cycle. The all-repair condition is a control; random same-count and leave-one-out variants are implemented for subsequent tests. Policy messages retain up to twelve raw proposal turns, subject to the server context limit.

Live decisions advance the world by ceil(measured request time / one second), with S as the default action during delay. The chosen move applies afterward if the episode has not ended. Retries and extra model calls add delay and generated tokens. Local Python parsing and arithmetic time is not included in this replay. A successfully constructed forecast program is reused within its episode; failed generation or construction can be attempted again on later decisions. Every attempt is counted. Cache hits reuse recorded durations and generated-token counts, so costs describe logical policy requests in the replay. Fixed null-prior scores are charged at every calibration decision even if cached. This replays observed latency; it is not an asynchronous deployment benchmark. Episodes cut short by the time cap or a shortened decision budget are censored. Simulator baselines have zero inference delay and therefore provide reference controls rather than latency-matched competitors.

Model | Condition | Done / censored | Success [Wilson 95% CI] | Generated tokens / ep. | Collisions / ep. | Error decisions / total

--- | --- | --- | --- | --- | --- | ---

1.7b | base | 10/0 | 0.0% [0.0, 27.8] | 1478 | 8.7 | 0/113

1.7b | selected | 10/0 | 0.0% [0.0, 27.8] | 1765 | 1.0 | 107/118

1.7b | all | 10/0 | 0.0% [0.0, 27.8] | 2746 | 1.2 | 160/174

4b | base | 10/1 | 0.0% [0.0, 27.8] | 1071 | 5.0 | 0/86

4b | selected | 10/0 | 0.0% [0.0, 27.8] | 1525 | 0.0 | 61/61

4b | all | 10/0 | 0.0% [0.0, 27.8] | 1709 | 0.0 | 78/78

simulator control | advance | 50/0 | 24.0% [14.3, 37.4] | 0 | 35.4 | 0/3908

simulator control | oracle | 50/0 | 80.0% [67.0, 88.8] | 0 | 0.1 | 0/1463

simulator control | planner | 50/0 | 80.0% [67.0, 88.8] | 0 | 0.1 | 0/1463

simulator control | random | 50/0 | 2.0% [0.4, 10.5] | 0 | 12.2 | 0/4970

simulator control | stay | 50/0 | 0.0% [0.0, 7.1] | 0 | 0.0 | 0/5000

Costs, collisions and errors summarize completed episodes. Errors are parse, controller or program failures, counted once per decision; the controller falls back to S on those errors. Incorrect forecasts that parse successfully are not counted as controller errors.

Live success intervals use the Wilson score interval to avoid degenerate bounds when every episode succeeds or fails. The requested bootstrap intervals are also retained in the evidence JSON. These methods agree on the point estimate but quantify uncertainty differently.

Completed model conditions have observed success rates from 0.0% to 0.0%. Paired repair contrasts use shared episode seeds where available. Partial conditions and timing differences limit comparisons across models; a failed repair remains part of the evidence.

Model | Repair vs base | Shared episodes | Success change [pp, 95% CI] | Generated-token change / ep.

--- | --- | --- | --- | ---

1.7b | selected | 10 | 0.0 [0.0, 0.0] | 287

1.7b | all | 10 | 0.0 [0.0, 0.0] | 1268

4b | selected | 10 | 0.0 [0.0, 0.0] | 454

4b | all | 10 | 0.0 [0.0, 0.0] | 638

Paired changes use only shared, uncensored episode seeds and percentile bootstrap intervals across episodes. They are exploratory and unadjusted. Identical outcomes on every observed pair can give a degenerate bootstrap interval; this does not establish zero population difference. Token changes include additional calls and retries, but are not a cost-matched comparison.


## 7. Limits and unfinished tests

The complete research plan is not finished. This cycle does not establish cross-game generality, thinking-mode effects, causal cascade attribution, or frozen-to-live predictive validity. Snapshot and bounded-search support for Snake and Overcooked is tested, but their diagnostic banks and empirical replications are pending. The 0.6B model, a second model family, larger-model ceiling, full three-paraphrase robustness, diary/outcome history study, framing and danger reversal study, cost-matched monitors, leave-one-repair-out live tests, and compute-conditional training remain outside the completed evidence unless a separate result file explicitly records them.

The task-specific input includes exact timers and motion rules, which makes it a rule-application test rather than inference from natural gameplay alone. Forecast errors can include input-reading errors: the main test does not condition on per-state perception success. Code offload removes some parsing burden as well as arithmetic, so its effect does not isolate a unique internal failure mechanism. The position target advances car motion without collision-driven resets. The frozen prompt calls these car ticks without explicitly stating that conditioning; the ambiguity further limits failure attribution. Renderer tests retain all object facts; this differs from the plan's truncated nearest-object lane table and radius-limited relation list. The bank is balanced by design and changes the natural state distribution. Self-reported confidence is not a calibrated probability. Small repair samples, one main model family, and an output budget that can truncate reasoning all limit inference. The live base is an instrumented forecast-and-action prompt, and retained history contains raw model proposals that may differ from the controller-executed action. A plain action-only baseline and execution-corrected history remain future controls. No novelty priority, venue readiness, or broad model-size law is claimed.


## 8. Reproducibility and review status

The main collector loads frozen source commit f71b9259abd055291c7c865659363aa1b3d9ec0f; commitment-test manifests record their separate module hashes. Canonical wins use lane 0, although upstream redraws the player at lane 9 in its terminal observation. The frozen collector also redraws initial cars after a collision on tick 100; current code retains the upstream terminal cars. This last-board display difference changes no success label, collision count, elapsed tick or later decision. The offline audit checks each physical tick against upstream and applies the frozen display convention only for that declared source revision.

The repository contains the preregistration, execution-plan deviations, pinned upstream revision, labeled bank, tests, runner, raw response logs or losslessly compressed copies, and this paper generator. The evidence JSON records the SHA-256 of input logs. Each model manifest records exact identity and decoding settings. Published numbers are computed from saved trials, including failures; they are not filled from the research plan. Author list, affiliation, and venue format are left for human review.

At report generation, the main four-model forecast target is complete. This PDF is a review artifact for the completed local cycle, not a claim that the full proposed project is done.


## 9. Conclusion

None of the 60 completed model-condition runs reached the goal. These runs reuse 10 environment seeds across models and conditions. This cycle does not demonstrate a goal-completion benefit from diagnosis-selected repairs under the chosen latency setting. Small live cohorts and wide Wilson bounds limit this negative result.

The artifact makes stage probes and repair decisions inspectable against a pinned simulator. The completed results describe the tested prompts, states and local model artifacts; they do not establish that a weak probe score uniquely identifies an internal failure cause. Commitment matching removes one measurement ambiguity, and recording delay and all calls makes the live comparison auditable. A broader claim that diagnosis reliably selects better agents requires the pending cross-game, cost-matched and intervention controls.


## References

[1] Wen et al. (2025). Real-Time Reasoning Agents in Evolving Environments. <link href="https://arxiv.org/abs/2511.04898" color="#215c8c">https://arxiv.org/abs/2511.04898</link>

[2] Okcu et al. (2026). The belief-action gap in language model agents. <link href="https://arxiv.org/abs/2610.17444" color="#215c8c">https://arxiv.org/abs/2610.17444</link>

[3] Zhu et al. (2025). Where LLM Agents Fail and How They can Learn From Failures. <link href="https://arxiv.org/abs/2509.25370" color="#215c8c">https://arxiv.org/abs/2509.25370</link>

[4] Wan et al. (2026). Mitigating Conversational Inertia in Multi-Turn Agents. ICML, PMLR 306. <link href="https://proceedings.mlr.press/v306/wan26a.html" color="#215c8c">https://proceedings.mlr.press/v306/wan26a.html</link>

[5] Wang et al. (2024). Can Language Models Serve as Text-Based World Simulators? ACL. <link href="https://aclanthology.org/2024.acl-short.1/" color="#215c8c">https://aclanthology.org/2024.acl-short.1/</link>

[6] Lehrach et al. (2025). Code World Models for General Game Playing. <link href="https://arxiv.org/abs/2510.04542" color="#215c8c">https://arxiv.org/abs/2510.04542</link>

[7] Xu et al. (2026). Code over Words: Overcoming Semantic Inertia via Code-Grounded Reasoning. Findings of ACL. <link href="https://aclanthology.org/2026.findings-acl.819/" color="#215c8c">https://aclanthology.org/2026.findings-acl.819/</link>

[8] Zhao et al. (2021). Calibrate Before Use: Improving Few-shot Performance of Language Models. ICML, PMLR 139. <link href="https://proceedings.mlr.press/v139/zhao21c.html" color="#215c8c">https://proceedings.mlr.press/v139/zhao21c.html</link>

[9] Yang et al. (2025). Qwen3 Technical Report. <link href="https://arxiv.org/abs/2505.09388" color="#215c8c">https://arxiv.org/abs/2505.09388</link>
