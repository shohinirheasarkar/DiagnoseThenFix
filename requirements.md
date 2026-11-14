# What the plan asks for

Checked against all 19 pages of the uploaded November 1 plan and the chat request.
A working tool is not a finished experiment. A finished experiment is not proof
that a fix helps. This list keeps those three things apart.

| Plan | Working code | Evidence still needed |
| --- | --- | --- |
| Shared bank, exact replay, labels, splits (p2) | Freeway: 500 states with full-bank search audit; other games: replay adapters | 500 labeled states each for Snake and Overcooked |
| Six formats, five reading probes (p3-5) | Yes, Freeway; signed-error plot from saved forecasts | 200-state study; action transfer; perception scaffold |
| Forecasts, k=1..10, confidence (p5-7) | Yes; exploratory held-out curve fits | 300-state matrix; thinking on; three paraphrases; confirmatory curve tests |
| Direct/chained/rule/table/code/tool repairs (p6-7) | Prompt variants, separate chained calls, two bounded code formats | Full isolated comparisons; calibrated error onset controls |
| Train forecast repair (p7) | No | Conditional on spare compute; keep training separate |
| Matched forecast/action gap and injection (p8-9) | Yes, separate commitment task | Finish four-model, 300-state, thinking matrix; gap closure and progress |
| Binding, retry, shield (p8-9) | Yes | Full matched comparisons; shield is a simulator ceiling |
| Diary: six types, four lengths (p9-11) | Six sources, replayed outcomes, real diaries, paired harm analysis | Full 300-state/model/thinking matrix; natural live history transfer |
| History repairs (p10-11) | Five frozen diary transforms; no-history live switch | Live delta/outcome/clip/balanced policies; full comparisons |
| Null/labels/skins/name/danger reversal (p11-13) | Null scores, labels, skins, naming, paired forecasts, reversal physics | Full empirical matrix; validation-picked framing |
| Calibration/framing/neutral/strategy repairs (p12-13) | Calibration and strategy; input switches | Full frozen and live comparisons |
| Cascade replay and four monitor triggers (p13-15) | Basic hooks and triggers | Real stage interventions; 100 failures/model; channel attribution; cost-matched monitor |
| Diagnosis and composed agent (p15-16) | Switches; matched C and diary D validation profiles | Matched random/leave-out comparisons; learned validation thresholds |
| Latency-aware agent (p16) | World advances during measured delay; saved-program latency replay | Repair utility rule; cost-matched experiments; frozen-to-live prediction |
| Live comparisons: 50 seeds/condition (p16) | Yes | First cycle: 10 paired seeds/model. Follow-up: 50 on 1.7B for base/selected/all; 4B partial: 27/27/27 completed for base/selected/all. Other planned comparisons remain open |
| Five Qwen sizes + another family + ceiling (p2) | Local Ollama client | 0.6B, second family, ceiling; no automatic downloads or paid services |
| Five seeds, 95% intervals, Holm (p2,18) | Bootstrap, paired estimates, cluster sign-flip tests, Holm | Full planned contrasts with cluster-aware tests; keep missing cells visible |
| Novelty audit and related work (p18-19) | Cited paper and focused 13-source comparison | Exhaustive priority search; no unsupported first/best claim |
| Reproducible repo and review paper (chat) | Tested PRs, raw evidence, paper generator and PDF | Local-cycle evidence and review paper are saved; full planned studies remain open |

The minimum version still needs the planned Stage D coverage and complete
50-seed comparisons across models and agents. The small diary pilot and the
finished 1.7B base/selected/all cohort cover only part of that target. The full version
also needs the other games, broader models, cascades and conditional training.
The two capped local cycles did not finish the full multiweek plan.
Missing results stay marked missing.

## Second cycle, November 13

Start: 07:39 UTC (02:39 EST). Hard stop: 15:39 UTC (10:39 EST).
Reserve the final 30 minutes for evidence checks, paper, PRs and cleanup. No paid
APIs, new model downloads, or changes to the first cycle's frozen results.

Order: repair the code interface; add honest paired history/framing studies;
collect balanced exploratory cohorts on 1.7B and 4B; extend repair/live coverage
as time allows; audit and publish the actual results. Changes are exploratory
follow-ups after seeing the first cycle, not new preregistered confirmations.

## Focused study, November 14

The separate, frozen [targeted-feedback protocol](targeted_protocol.md) completed
200 logical generation requests, 120 final candidates, 1,458,000 forecast checks
and 36,000 uncensored game episodes. The independent full replay audit passes.
Earlier archives and their paper remain separate.

The repair treatments did not produce a useful semantic contrast: all 17 usable
1.7B pairs had identical final code; all 4B repairs failed the required output
contract. Alternative JSON fields were inspected statically, never rescored.
Zero observed differences therefore do not establish equivalence or show that
diagnosis is ineffective.

Missing: the planned 100-seed assisted reference, broader models/games and the
original multiweek matrix. The small development reference is not a substitute.
Most primitive test inputs overlap training inputs despite disjoint board seeds.
Research calls closed at the fixed 16:11:19 UTC cutoff; subsequent work audits
and packages existing records. See the new paper for limits and exact counts.
