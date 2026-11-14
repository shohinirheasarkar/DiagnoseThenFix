# What this experiment can add

Primary sources checked November 14, 2026. This focused audit does not prove priority.

| Work | Relevant prior result | Boundary for this study |
| --- | --- | --- |
| [Wen et al., Real-Time Reasoning Agents in Evolving Environments (2025)](https://arxiv.org/html/2511.04898v1) | Studies agents whose world moves while they reason. | Moving-world evaluation and the Freeway environment are inherited ideas. |
| [Lehrach et al., Code World Models for General Game Playing (2025), §4.1](https://arxiv.org/html/2510.04542v1) | Synthesizes executable game models and refines them with failed unit-test traces, serially or through tree search. | Executable forecasts, simulator feedback, and refinement are not new. |
| [Lou et al., AutoHarness (2026), §§3–4.1](https://arxiv.org/html/2603.03329v1) | Synthesizes game harnesses; samples failed steps, consolidates errors, and chooses functions to repair based on failure. | Neither feedback-guided game code nor selective repair is new. |
| [Orvalho et al., Counterexample Guided Program Repair (AAAI 2025), method and feedback sections](https://arxiv.org/html/2502.07786v1) | Combines MaxSAT fault localization, program sketches, and counterexample-driven LLM repair. | Diagnosis plus repair predates this project. Our categories are behavioral tags, not formal fault localization. |
| [Zhu et al., Where LLM Agents Fail and How They can Learn From Failures (2025), §3.2](https://arxiv.org/html/2509.25370v1) | AgentDebug uses a failure taxonomy and targeted feedback for agent recovery. | We do not introduce failure taxonomies or general diagnosis-driven agents. |

The narrow question is whether selecting correct failure examples from the worst
predefined category improves repair more than uniformly selecting correct failures,
under equal example counts and generation limits. Shared initial programs make the
comparison paired. Closed-loop play tests whether forecast gains transfer through
one fixed, human-written controller.

The inspected sources establish the ingredients. Their inspected method sections
do not establish this exact paired category-selection comparison in Freeway.
That observation is not a claim that no other paper has tested it. One game, two
sizes in one model family, and a simple motion function limit generalization.

Before stronger novelty language, expand the search into counterexample selection,
active testing, and failure-driven curriculum learning. Do not present a positive
result as the first demonstration that diagnosis can help code repair.
