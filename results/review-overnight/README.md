# Second-cycle evidence

The November 13 collection is closed. The follow-up saved 231 completed live runs and one cut short by the deadline.
No run reached the goal. The 1.7B model finished fifty seeds each for
base/selected/all; 4B finished twenty-seven each. The 4B cohort is partial.
The full research plan remains unfinished. The first study stays in `results/review`.

- Root model folders: 1,386 held-out diary trials, 680 framing trials, 336 diary
  validation trials, 64 matched validation trials, and 72 forecast-pilot trials.
- Neural live files keep every completed and censored attempt. Conditions use
  fresh seeds 60000–60049 in fixed base/selected/all order. Missing runs stay missing.
- `grammar`: 1,237 forecast trials; the 4B cohort is partial.
- `compiled`: 2,000 original replay cases. `compiled/native`: 2,000 after the
  numeric adapter fix. Both reuse five saved programs per model. Keep versions separate.
- Root baselines: 250 fresh-seed simulator controls, with no inference delay.

`archive.json` hashes every JSON evidence file and its uncompressed source.
Each study keeps its own design, source revision and actual counts. The restart
record preserves the collector hashes and completed log prefixes. `cpu_seconds`
means local elapsed checking/execution time, not process CPU usage. Recheck with:

```sh
python scripts/check_evidence.py results/review-overnight
```
