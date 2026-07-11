# TODO

## W&B rollout video slider pinned to epoch 1 ("Step: 626")

### Symptom
In the W&B UI the `rollout/video` media panel shows `Step: 626`, `Index: 0` and the
step slider does not advance while training continues (observed at epoch 18/25).
`626 ≈ one epoch` (10000 samples / batch size 16 = 625 optimizer steps), so the panel
is pinned to the end of epoch 1.

### What is NOT broken (verified)
- All per-epoch GIFs are generated and distinct on disk:
  `runs/<run>/media/sim-rollout-epoch-001.gif ... -025.gif` (25 unique files).
- The trainer logs a freshly-named GIF every epoch with a correctly incrementing
  `global_step` (625, 1250, ... 11250) — `trainer.py:171` passes `self.global_step`.
- So the videos are produced and sent every epoch. This is a display issue, not a
  training/logging-frequency bug.

### Likely cause
`src/vat_mini/tracking.py:80-81`:
```python
self._run.define_metric("trainer/global_step")
self._run.define_metric("*", step_metric="trainer/global_step")
```
The `"*"` wildcard also matches the `rollout/video` media key. W&B media (Video)
panels are indexed by the run's internal step and do not play well with a custom
`step_metric`; remapping everything to `trainer/global_step` is a known rough edge
that can leave the media slider pinned to the first logged step.
`Index: 0` is a red herring — only one Video is logged per call, so Index is always 0;
the *Step* slider is what should scrub across epochs.

### Secondary cost
Each GIF is ~5.5 MB and one is logged per epoch (`rollout_every_epochs: 1`), so a
25-epoch run uploads ~140 MB of video to W&B — for clips that currently can't be
scrubbed. The 25 sim rollouts also add ~10 min to the run.

### Proposal
1. Stop applying the wildcard `step_metric` to media. Either narrow the glob to the
   scalar metric keys, or explicitly exclude `rollout/*` media so Video panels use the
   default internal step and the slider populates per epoch.
2. Reduce `tracking.rollout_every_epochs` (e.g. 5) to cut upload cost and wall-clock
   time. Can be a config/default change or a per-run `--set`.

### Note / context
Even with the slider working, all 25 GIFs are ~5.5 MB full-length (~200-step) sim
rollouts that never solve the task (success rate 0), so they look nearly identical.
The frozen slider is masking a non-difference, not hidden progress. The pretrain
plateau (val MAE ~0.058-0.061 after epoch ~12) is tracked separately as the real
modeling bottleneck.
