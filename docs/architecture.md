# Architecture

VAT Mini is a deliberately small vision-action policy. It keeps the control flow and phase boundaries found in larger training systems while using a synthetic task that can run locally.

## The task

Each observation is a rendered RGB GridWorld. A blue agent must move to a red target with one of five discrete actions: `stay`, `up`, `down`, `left`, or `right`. The data generator uses a shortest-path expert to produce deterministic training and validation trajectories from different seeds. Steps after an episode terminates are padding and carry an explicit `valid_steps` mask; losses and metrics never learn from that padding.

This environment is intentionally simple. It tests whether the complete system can learn a visual state-to-action mapping, run without teacher forcing, save and reload checkpoints, and report honest rollout metrics. It is not a robotics simulator or evidence that a learned policy will transfer to real-world control.

## Model data flow

For every timestep `t`:

1. `VisionEncoder` converts the RGB frame into one visual embedding with a small convolutional network.
2. `VisionActionTransformer` adds three vectors: the visual embedding, the embedding of action `t-1`, and a learned position embedding.
3. A causal transformer processes the sequence. Its attention mask prevents timestep `t` from reading future tokens.
4. A linear action head produces five logits for action `t`.

The first timestep receives a learned begin-action token. During supervised training, previous expert actions are shifted right. During rollout, previous model actions are fed back autoregressively.

```text
RGB frames [B,T,3,H,W]
       │
       ▼
CNN frame encoder ──► visual tokens [B,T,D]
                           + previous-action embeddings
                           + position embeddings
                                      │
                                      ▼
                             causal Transformer
                                      │
                                      ▼
                              action logits [B,T,5]
```

The current encoder emits one token per frame. Patch tokens, language conditioning, continuous action heads, and cross-attention are reasonable future extensions, but they are not silently implied by the current code.

## Training stages

### 1. Supervised policy pretraining

`configs/pretrain.yaml` trains the policy with behavior cloning: cross-entropy between predicted actions and shortest-path expert actions. In this repository, “pretraining” means supervised policy pretraining on demonstrations. It is not masked-image modeling or self-supervised representation learning.

### 2. Advantage-weighted post-training

`configs/posttrain.yaml` initializes from the pretraining checkpoint and weights action imitation losses using normalized reward-to-go. Its training split injects controlled action noise, creating higher- and lower-return decisions for the weighting objective to distinguish; validation remains a clean expert split. Higher-return decisions receive more weight, with temperature and maximum-weight controls to keep optimization stable.

This is an educational, discrete-action version of advantage-weighted imitation. It is not RLHF, DPO, online reinforcement learning, or preference tuning.

## Module boundaries

| Module | Responsibility |
| --- | --- |
| `config.py` | Typed experiment configuration, validation, YAML loading, CLI overrides |
| `data.py` | Environment, rendering, expert demonstrations, datasets, data loaders |
| `model.py` | Visual encoder and causal vision-action transformer |
| `objectives.py` | Phase-specific loss functions |
| `trainer.py` | Optimization loop, metrics, evaluation, checkpoint cadence |
| `evaluation.py` | Teacher-forced metrics and closed-loop rollouts |
| `tracking.py` | Optional W&B adapter for metrics, rollout media, and final artifacts |
| `checkpoint.py` | Portable, atomic checkpoint persistence |
| `device.py` | MPS/CUDA/CPU selection and seeding |
| `cli.py` | Thin command orchestration |

The dependencies point inward: the CLI composes the system; individual modules do not know about command-line parsing. W&B stays behind the `ExperimentTracker` protocol. When tracking is disabled, `DisabledTracker` makes the same calls no-ops and the package never imports W&B.

## Training and telemetry cadence

One batch produces one optimizer step. One epoch is a complete pass through every batch in the training loader. The two cadences deliberately log different evidence:

```text
epoch
  batch → forward → loss → backward → optimizer step → batch metrics
  batch → forward → loss → backward → optimizer step → batch metrics
  ...
  end of loader → held-out validation → epoch metrics → fixed-seed rollout GIF
```

Batch metrics answer whether optimization is numerically moving. Epoch validation answers whether the policy generalizes to held-out expert trajectories. The fixed-seed rollout answers whether closed-loop behavior is improving on the same initial condition rather than merely improving under teacher forcing.

The final evaluation still runs multiple seeded episodes and writes aggregate rollout metrics into the local checkpoint and `metrics.jsonl`. W&B receives that final local checkpoint plus the resolved config and metrics file as a versioned model artifact. It supplements the local artifact contract; it does not replace it.

Tracking is configured under `tracking`:

| Field | Meaning |
| --- | --- |
| `enabled` | Construct the W&B adapter; false leaves the dependency optional |
| `project` / `entity` | W&B destination |
| `mode` | `online` for live sync or `offline` for later upload |
| `run_name` | Optional human-readable run label |
| `rollout_every_epochs` | Visual rollout cadence; scalar epoch metrics still log every epoch |

## Apple silicon defaults

- `device: auto` selects MPS, then CUDA, then CPU. The tiny smoke config pins CPU because its kernels are too small to amortize Metal dispatch; the larger configs keep automatic selection.
- Training uses float32 by default. Mixed precision and compilation are intentionally not required for the baseline.
- Data loaders default to `num_workers: 0` and do not pin memory, avoiding common multiprocessing and CUDA-specific assumptions on macOS.
- Checkpoints load through CPU before the model moves to the selected accelerator.
- CPU tests remain the deterministic compatibility gate; MPS reproducibility is best-effort.

## What “production-shaped” means here

Included now:

- validated, typed configuration;
- separate data/model/objective/trainer/evaluation boundaries;
- deterministic split generation;
- atomic, stage-aware checkpoints and resolved config snapshots;
- JSONL metrics suitable for later ingestion;
- optional W&B metrics, fixed-seed rollout GIFs, and final model artifacts;
- unit and end-to-end smoke tests;
- CPU fallback and explicit device errors.

Not included yet:

- distributed training or a project-specific remote object store;
- real robot data, video decoding, language instructions, or continuous controls;
- automatic mixed precision, compilation, gradient accumulation, or sharding;
- self-supervised visual pretraining, offline RL, preference optimization, or safety constraints.

Those belong in later milestones after the local learning loop is understood and measured.
