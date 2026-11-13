# What earlier work already covers

Checked against primary sources on November 13, 2026. This is a focused comparison
of the plan's reading list, not an exhaustive search proving priority. The last
column is our interpretation of what this cycle can support.

| Source | Already established | Our narrower contribution or limit |
| --- | --- | --- |
| [Wen et al., RealtimeGym](https://arxiv.org/html/2511.04898v1), §3 and appendix C.4 | Moving worlds, reasoning delays, reactive/planning agents, and code policies | We reuse their games. Saved code and real-time reasoning are not new here. |
| [Jin et al., Seeing Without Understanding](https://proceedings.mlr.press/v306/jin26a.html) | Separate perception, rule application and simulation tests | Our text probes do not establish a new stage decomposition or isolate internal causes. |
| [Okcu et al., Belief–Action Gap](https://arxiv.org/html/2610.17444v1), §§2,5,6 | Distinguish available information, action consistency and causal control | Our repeated-action commitment task aligns forecast and choice horizons; it does not read internal beliefs. |
| [Zhu et al., AgentDebug](https://arxiv.org/html/2509.25370v1), §3.2 | Module-level failures, counterfactual correction and targeted feedback | Our full causal cascade and diagnosis-targeting tests remain unfinished. |
| [Wan et al., Conversational Inertia](https://proceedings.mlr.press/v306/wan26a.html) | Longer-context imitation and context preference learning | Our transplanted diaries hold the current board fixed. They are not a natural-history replication. |
| [Schmied et al., Greedy Agents](https://arxiv.org/html/2504.16078v1), §4.2 | Greediness, frequency bias and the knowing–doing gap; RL mitigation | Safe repeated choices can still make no progress. We did not run RL training. |
| [Lehrach et al., Code World Models](https://arxiv.org/html/2510.04542v1), §4 and appendix E | Executable models, refinement, search and rejection of poor samples | We test bounded motion functions and retain all five samples. Our human selector is assistance. |
| [Lou et al., AutoHarness](https://arxiv.org/html/2603.03329v1), §§3,4.3 | Environment-guided harness synthesis and fully compiled policies | Training checks and skipping new decision-time LLM calls are established ideas. |
| [Xu et al., Code over Words](https://aclanthology.org/2026.findings-acl.819/) | Conflicting semantic priors and code-grounded reasoning | Our contact reversal is a small changed-rule task; ordinary skins keep physics fixed. |
| [Chai et al., XiangqiBench](https://arxiv.org/html/2610.02425v1), §§5.2,6.3 | A good first move or simulated line need not yield a verified win | Frozen forecasts and live goals stay separate. Predicting repair benefit remains open. |
| [Li et al., Spatial World Models](https://arxiv.org/abs/2604.10690) | Spatial results depend on representation and prompting | Our six-format validation selects an input; it does not prove format-invariant understanding. |
| [Huang et al., Text2Space](https://arxiv.org/abs/2604.14641) | Separate layout construction from reading; train construction skills | We supply rendered boards and do not test layout generation or that training method. |
| [Wang et al., Text-Based World Simulators](https://aclanthology.org/2024.acl-short.1/) | Simulator-checked state-transition prediction | We measure explicit Freeway horizons, including its timer and wrap rules. No first-ever simulation claim. |

The useful artifact here is the replayable Freeway evidence: horizon curves,
matched commitment probes, controlled diary/framing contrasts and separately
versioned delay replays. The completed data do not establish that diagnosis-selected
repairs beat equally costly untargeted repairs. That claim still needs the planned
random, leave-out and cost-matched comparisons.
