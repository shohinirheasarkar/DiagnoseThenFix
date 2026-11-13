# Second-cycle evidence

This snapshot contains closed probes and offline replays from November 13.
It includes the complete 1.7B live cohort: fifty seeds per condition for
base/selected/all, 150 completed runs, no censored runs and no goals. The 4B live
cohort is still collecting and is excluded for now. The full
research plan remains unfinished. The first study stays in `results/review`.

- Root model folders: 1,386 held-out diary trials, 680 framing trials, 336 diary
  validation trials, 64 matched validation trials, and 72 forecast-pilot trials.
- `grammar`: 1,237 forecast trials; the 4B cohort is partial.
- `compiled`: 2,000 original replay cases. `compiled/native`: 2,000 cases after
  the numeric adapter fix. Keep versions separate. Both reuse the same ten saved
  generation requests: five programs per model, not thousands of new programs.
- Root baselines: 250 fresh-seed simulator controls, with no model delay.

`archive.json` hashes every JSON evidence file and its uncompressed source.
Each study keeps its own design, source revision and actual counts.
The field `cpu_seconds` records local elapsed checking/execution time, not
process CPU usage. It contributes to the simulated wall-clock delay. Recheck with:

```sh
python scripts/check_evidence.py results/review-overnight
```
