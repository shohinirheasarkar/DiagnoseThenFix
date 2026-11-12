# Saved evidence

This is the November 12 local cycle. Completed counts and results are in
[the review paper](../../output/pdf/diagnose-then-repair.pdf) and its
[evidence JSON](../../output/pdf/evidence.json). The full multi-game study is
unfinished. Earlier engineering pilots are not pooled here.

Each `.jsonl.gz` file is a lossless trial or episode log, including failures.
`archive.json` records hashes of both the compressed and original bytes. With
the dependencies in the main README installed, these commands make no model calls:

```sh
python scripts/check_evidence.py results/review
python paper/build.py --results results/review --output output/pdf/diagnose-then-repair.pdf
```

The main collector used source commit
`f71b9259abd055291c7c865659363aa1b3d9ec0f`. Its source fingerprint and
`cycle.json` identify the loaded code; a manifest's `git_revision` records the
checkout at launch. To repeat it, extract that revision into a separate folder,
set `DIAGNOSE_SOURCE_ROOT` to its absolute path, and run `scripts/research_cycle.py`
with a future `--deadline` and a new `--output` folder. Existing designs cannot be
silently overwritten. Exact prompts, model digests and decoding settings are in
the logs and manifests.

`C-commitment` is a separate, explicitly repeated-action protocol with its own
module hashes. Older C folders measure repeated-action risk disagreement, not a
formal belief-action gap. Oracle forecasts and shields are upper bounds.

Run the audit from this archive root. It checks physical steps against pinned
RealtimeGym and uses the declared frozen-source convention for the last board
after a terminal collision. The current engine corrects that display; success,
ticks and collisions are unchanged.
