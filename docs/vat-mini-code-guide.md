# VaT-mini code guide

## System flow

```text
config
  -> GridWorld demonstration generation
  -> DataLoader batches
  -> vision encoder
  -> causal transformer
  -> action logits
  -> stage-specific loss
  -> optimizer update
  -> checkpoint and metrics
  -> closed-loop rollout evaluation
```

The main implementation contract is:

```text
P(action_t | frame_0 ... frame_t, action_0 ... action_(t-1))
```

## Repository map

| File | Responsibility |
| --- | --- |
| [`config.py`](../src/vat_mini/config.py) | Experiment schema, validation, YAML loading, and overrides |
| [`data.py`](../src/vat_mini/data.py) | GridWorld, rendering, expert policy, trajectories, and DataLoaders |
| [`model.py`](../src/vat_mini/model.py) | CNN vision encoder and causal transformer policy |
| [`objectives.py`](../src/vat_mini/objectives.py) | Behavior cloning and reward-weighted imitation losses |
| [`trainer.py`](../src/vat_mini/trainer.py) | Optimization loop, evaluation cadence, metrics, and checkpoints |
| [`evaluation.py`](../src/vat_mini/evaluation.py) | Teacher-forced and closed-loop evaluation |
| [`checkpoint.py`](../src/vat_mini/checkpoint.py) | Portable atomic checkpoint persistence |
| [`tracking.py`](../src/vat_mini/tracking.py) | Optional experiment tracking adapter |
| [`device.py`](../src/vat_mini/device.py) | Device selection and deterministic seeding |
| [`cli.py`](../src/vat_mini/cli.py) | Command composition and operator interface |

For the architecture concepts behind these components, see [Vision-action transformer architecture](vision-action-transformer-notes.md).

## 1. Configuration is the experiment contract

[`ExperimentConfig`](../src/vat_mini/config.py#L61) composes four groups:

```text
data      -> dataset and batching choices
model     -> network dimensions
training  -> stage, optimizer settings, and objective controls
tracking  -> optional telemetry settings
```

The main pretraining model uses:

```text
sequence length:       12
image size:            32 x 32
actions:               5
vision width:          32
embedding dimension:   96
transformer layers:    2
attention heads:       4
feed-forward width:    192
```

Validation catches incompatible contracts before training. For example, the embedding dimension must be divisible by the number of heads, and the data sequence length cannot exceed the model's maximum sequence length.

The two main experiment files are:

- [`pretrain.yaml`](../configs/pretrain.yaml)
- [`posttrain.yaml`](../configs/posttrain.yaml)

## 2. Environment state and observations

[`VisualGridWorld`](../src/vat_mini/data.py#L25) owns the true state:

```text
(agent row, agent column, target row, target column)
```

The model never receives these integers. It receives an RGB rendering containing a blue agent and red target.

The five actions are:

```text
0 stay
1 up
2 down
3 left
4 right
```

The environment reward is implemented in [`step`](../src/vat_mini/data.py#L70):

```text
reaching target:       1.0
otherwise:             0.05 * progress - 0.01
```

`progress` is the reduction in Manhattan distance. A useful move gets a small positive reward; an unhelpful move receives a penalty.

## 3. Expert demonstration generation

[`expert_action`](../src/vat_mini/data.py#L58) chooses a shortest-path move using row and column deltas.

When vertical and horizontal moves are equally good, the expert randomly chooses an axis. This creates multiple valid shortest paths instead of forcing one arbitrary route.

[`generate_demonstrations`](../src/vat_mini/data.py#L139) produces fixed-shape arrays:

| Field | Dataset shape | Meaning |
| --- | --- | --- |
| `observations` | `[N, T, 3, H, W]` | rendered frame before each action |
| `actions` | `[N, T]` | executed action IDs |
| `rewards` | `[N, T]` | immediate rewards |
| `valid_steps` | `[N, T]` | real timestep versus padding |

When an episode ends early, the remaining slots are filled with zeros and `stay` actions. `valid_steps` ensures those placeholders do not contribute to loss or metrics.

Training and validation use different random seeds. Validation always uses clean expert behavior, even when post-training injects noise into its training trajectories.

## 4. Batch tensor contract

The DataLoader creates:

```text
observations: [B, T, 3, H, W] float32
actions:      [B, T]           int64
rewards:      [B, T]           float32
valid_steps:  [B, T]           bool
```

With the main configs:

```text
B = 32
T = 12
H = W = 32
```

[`build_dataloaders`](../src/vat_mini/data.py#L242) shuffles training data with a seeded generator and leaves validation ordered.

## 5. Vision encoder

[`VisionEncoder`](../src/vat_mini/model.py#L11) converts each frame into one 96-dimensional vector:

```text
[3, 32, 32]
  -> Conv2d stride 2 + GELU
  -> Conv2d stride 2 + GELU
  -> adaptive average pool to [64, 4, 4]
  -> flatten to [1024]
  -> linear projection
  -> [96]
```

The `4 x 4` pooled feature map preserves coarse location. Global average pooling would more aggressively erase whether the agent is left, right, above, or below the target.

This encoder emits one token per frame. It does not emit ViT-style patch tokens.

## 6. Timestep token construction

[`VisionActionTransformer.forward`](../src/vat_mini/model.py#L69) first flattens the batch and time axes so every frame can pass through the CNN:

```text
[B, T, 3, H, W]
-> [B*T, 3, H, W]
-> CNN
-> [B*T, D]
-> [B, T, D]
```

It then creates each timestep token:

```text
token_t = vision(frame_t)
        + embedding(previous_action_t)
        + embedding(position_t)
```

All three tensors have shape `[B, T, D]` after broadcasting.

## 7. Shifted actions

[`shifted_actions`](../src/vat_mini/model.py#L63) transforms target actions before they enter the model:

```text
targets: [a0,  a1, a2, a3]
inputs:  [BOS, a0, a1, a2]
```

The special beginning-action ID is `num_actions`, which is `5` in this model. The embedding table therefore contains six entries: five real actions plus `BOS`.

This prevents the token at `t` from seeing `action_t`, its own target.

## 8. Transformer stack

The model constructs [`nn.TransformerEncoderLayer`](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html) with:

```text
d_model          = 96
nhead            = 4
dim_feedforward  = 192
activation       = GELU
batch_first      = true
norm_first       = true
```

Although PyTorch calls this an encoder layer, the supplied causal mask gives the stack decoder-only information flow.

Each head operates on:

```text
96 / 4 = 24 dimensions
```

The stack contains two layers in the main configs. A final layer normalization is applied by `nn.TransformerEncoder`.

## 9. Causal mask

The model creates:

```python
torch.triu(
    torch.ones(T, T, dtype=torch.bool),
    diagonal=1,
)
```

Entries above the diagonal are blocked:

```text
query 0 -> key 0
query 1 -> keys 0, 1
query 2 -> keys 0, 1, 2
query 3 -> keys 0, 1, 2, 3
```

The key invariant is:

> Prediction `t` may use observations through `t` and actions before `t`, but never future observations or `action_t`.

[`test_causal_mask_blocks_future_observations`](../tests/test_model.py#L28) changes future frames and verifies that earlier logits remain unchanged.

## 10. Action head and full shape trace

[`action_head`](../src/vat_mini/model.py#L61) is a linear projection from width 96 to five logits.

One main-config forward pass is:

```text
observations                         [32, 12, 3, 32, 32]
reshape for frame encoder            [384, 3, 32, 32]
vision encoder                       [384, 96]
restore batch and time               [32, 12, 96]

shifted action IDs                   [32, 12]
action embeddings                    [32, 12, 96]
position embeddings                  [1, 12, 96]

summed timestep tokens               [32, 12, 96]
causal transformer                   [32, 12, 96]
action head                          [32, 12, 5]
```

## 11. Pretraining objective

For `training.stage: pretrain`, [`Trainer._build_objective`](../src/vat_mini/trainer.py#L40) selects [`BehaviorCloningObjective`](../src/vat_mini/objectives.py#L18).

The objective:

1. computes cross-entropy for every `[batch, time]` position;
2. multiplies each loss by `valid_steps`;
3. sums the valid losses;
4. divides by the valid-timestep count.

Rewards are explicitly discarded. This stage is ordinary supervised action prediction.

“Pretraining” here means the first policy-training stage. It is not masked-image or self-supervised representation pretraining.

## 12. Post-training objective

[`posttrain.yaml`](../configs/posttrain.yaml) changes three important things:

```text
initial checkpoint:  runs/pretrain/latest.pt
learning rate:       0.0001
training noise:      0.25
```

The 25% action noise creates trajectories containing both better and worse actions.

[`AdvantageWeightedImitationObjective`](../src/vat_mini/objectives.py#L38) calculates:

```text
returns_to_go = reverse(cumulative_sum(reverse(rewards)))
```

It standardizes valid returns across the batch:

```text
advantages = (returns_to_go - valid_mean) / (valid_std + 1e-6)
```

It then creates detached weights:

```text
weights = exp(advantages / temperature)
weights = clamp(weights, max=maximum_weight)
weights = weights * valid_steps
```

The cross-entropy loss becomes a weighted average. Higher-return portions of the data produce larger parameter updates.

### Implementation boundary

This is not full advantage-weighted regression. There is no learned value function and no explicit `Q(s,a) - V(s)` estimate. The implementation standardizes observed return-to-go against the batch-wide distribution.

“Reward-weighted imitation” is the most literal description.

## 13. Optimizer step ownership

[`Trainer`](../src/vat_mini/trainer.py#L19) owns:

- the model;
- AdamW optimizer;
- current objective;
- device;
- checkpoint manager;
- global optimizer-step count;
- experiment tracker.

[`_train_epoch`](../src/vat_mini/trainer.py#L115) performs:

```text
move batch to device
-> shift actions
-> forward pass
-> objective
-> clear old gradients
-> backward pass
-> clip gradient norm
-> AdamW update
-> increment global_step
-> log batch metrics
```

Gradient clipping limits the total gradient norm to the configured threshold before the optimizer update.

## 14. Epoch lifecycle

[`Trainer.fit`](../src/vat_mini/trainer.py#L53) runs:

```text
write resolved config

for each epoch:
  train over every batch
  evaluate held-out demonstrations
  append metrics
  optionally record a fixed-seed rollout
  save checkpoint on schedule

run final multi-episode closed-loop evaluation
save final checkpoint with rollout metrics
finish tracker
```

One optimizer step is one batch update. One epoch is one full pass over the training DataLoader.

## 15. Teacher-forced validation

[`evaluate_demonstrations`](../src/vat_mini/evaluation.py#L25) uses shifted expert actions as model inputs.

It reports:

- validation cross-entropy;
- valid-timestep action accuracy;
- majority-class accuracy baseline.

The majority baseline catches a policy that appears accurate only because one action is common.

This path evaluates imitation under clean expert history. It does not test whether the model can recover from its own actions.

## 16. Closed-loop rollout

[`evaluate_rollouts`](../src/vat_mini/evaluation.py#L59) gives the expert no role after environment reset:

```text
render observation
-> append observation to history
-> model chooses action
-> environment executes action
-> append model action to history
-> repeat until success or step limit
```

It reports:

| Metric | Meaning |
| --- | --- |
| `rollout_success_rate` | Fraction of episodes reaching the target |
| `rollout_path_efficiency` | Optimal step count divided by actual steps; zero on failure |
| `rollout_mean_return` | Mean accumulated environment reward |

The maximum rollout length is `2 * (grid_size - 1)`, which is 14 steps for the default `8 x 8` grid.

## 17. `choose_action` execution trace

[`choose_action`](../src/vat_mini/model.py#L92) receives unbatched observation and action histories.

At step zero:

```text
observation_history = [frame_0]
action_history      = []
previous_actions    = [BOS]
model output        = logits for position 0
selected action     = argmax(logits[0, -1])
```

At step one:

```text
observation_history = [frame_0, frame_1]
action_history      = [action_0]
previous_actions    = [BOS, action_0]
selected action     = final-position argmax
```

If the history exceeds `max_sequence_length`, the function keeps only the newest observation window and the actions needed to align with it.

The implementation recomputes the entire window on every step. It does not use a KV cache.

## 18. Checkpoint contract

[`CheckpointManager.save`](../src/vat_mini/checkpoint.py#L24) persists:

```text
model state
optimizer state
resolved config
training stage
epoch
global step
metrics
```

It writes a temporary file and atomically replaces the final path. It maintains both numbered stage checkpoints and `latest.pt`.

[`load_checkpoint`](../src/vat_mini/checkpoint.py#L53) loads onto CPU first and uses `weights_only=True`. This keeps checkpoints portable across CPU, MPS, and CUDA while restricting unsafe Python-object deserialization.

Post-training loads model weights from the pretraining checkpoint, but the new `Trainer` creates a fresh optimizer. This is stage initialization, not an exact training resume.

## 19. CLI composition

[`cli.py`](../src/vat_mini/cli.py) is intentionally thin.

```text
load config
-> seed libraries
-> select device
-> create data loaders
-> construct model and trainer
-> optionally load initial model checkpoint
-> run requested operation
```

Available operations are:

```text
inspect
generate-data
train
evaluate
smoke
```

The CLI owns argument parsing. Model, data, and objective modules do not know about command-line flags.

## 20. Important implementation limitations

### The current frame already reveals the full GridWorld state

The agent and target are both visible. A frame-only policy could solve the current task. The transformer is architectural scaffolding for future tasks where history matters.

### One token represents an entire frame

The temporal transformer cannot directly attend to individual spatial regions. All spatial information must survive the CNN projection.

### Post-training is batch-relative

Return normalization uses the valid returns present in the current batch. The same trajectory can receive slightly different weights under different batch composition.

### No padding mask enters the transformer

Padding is removed from loss and metrics, but padded token representations are still processed. Because valid positions occur before padding and attention is causal, real positions cannot read later padded positions. This keeps valid predictions clean.

### Rollout inference recomputes history

This is simple for short sequences but scales poorly. A longer-running controller should use KV caching or another streaming state representation.

### No exact resume path

Checkpoints contain optimizer state, but `initialize_from_checkpoint` loads only model state because it does not pass the trainer optimizer to `load_checkpoint`.

## 21. Code-reading order

1. [`config.py`](../src/vat_mini/config.py) — learn the legal experiment shape.
2. [`data.py`](../src/vat_mini/data.py) — trace the source of every tensor.
3. [`model.py`](../src/vat_mini/model.py) — follow shapes and information access.
4. [`objectives.py`](../src/vat_mini/objectives.py) — understand what produces gradients.
5. [`trainer.py`](../src/vat_mini/trainer.py) — locate mutable runtime state.
6. [`evaluation.py`](../src/vat_mini/evaluation.py) — compare clean-history and self-generated-history execution.
7. [`checkpoint.py`](../src/vat_mini/checkpoint.py) — inspect the persistence boundary.
8. [`cli.py`](../src/vat_mini/cli.py) — see how the pieces are composed.

At every boundary, ask:

```text
What is the tensor contract?
Who owns the mutable state?
What information is this component allowed to see?
How is correctness measured here?
```

## 22. Useful next experiments

1. Remove the transformer and establish a frame-only MLP baseline.
2. Remove previous-action embeddings and compare rollout behavior.
3. Disable the causal mask and observe the misleading training improvement.
4. Increase action noise and inspect the return-weight distribution.
5. Add obstacles and replace the Manhattan expert with BFS.
6. Hide part of the environment so history becomes necessary.
7. Emit spatial patch tokens instead of one frame token.
8. Add a learned value function and implement a closer version of AWR.
9. Add KV caching for incremental rollout inference.
10. Compare token accuracy against rollout success across checkpoints.

