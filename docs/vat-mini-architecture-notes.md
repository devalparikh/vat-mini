# VaT-mini architecture

## The whole system in one sentence

VaT-mini watches a short sequence of images, remembers the actions already taken, and predicts the next discrete action.

```text
past and current images + past actions -> next action
```

More precisely, at timestep `t` it models:

```text
P(action_t | image_0 ... image_t, action_0 ... action_(t-1))
```

That conditional-probability statement is the architecture's main contract. Most implementation details exist to enforce it.

## First map: what each part does

```text
GridWorld       -> creates states, images, rewards, and expert actions
Dataset         -> freezes many trajectories into tensors
Vision encoder  -> turns each image into one vector
Transformer     -> lets each timestep read relevant earlier timesteps
Action head     -> turns the final vector at each timestep into 5 action scores
Objective       -> says which predictions should be rewarded during training
Trainer         -> owns optimizer state and repeatedly updates parameters
Rollout loop    -> runs the model as the controller, without expert help
```

The shortest useful model of the model itself is:

```text
one frame -> CNN -> one visual vector
visual vector + previous-action vector + position vector -> one timestep token
sequence of timestep tokens -> causal transformer -> contextualized tokens
each contextualized token -> linear layer -> five action logits
```

## Why this is not KNN

KNN is a useful bridge because both systems answer a prediction query from examples.

In KNN:

1. choose a fixed distance function;
2. find nearby stored examples;
3. let their labels vote.

In VaT-mini:

1. learn the representation used to compare information;
2. use attention to compute a different weighted lookup for every timestep and layer;
3. transform the retrieved information through learned MLPs;
4. output a probability distribution over actions.

A rough analogy is:

```text
KNN neighbor score        <-> attention score
neighbor labels/features  <-> value vectors
weighted neighbor vote    <-> weighted sum of values
```

But do not take the analogy literally:

- attention searches only the tokens in the current sequence, not the training dataset;
- its similarity function is learned, not a fixed Euclidean distance;
- it performs a soft weighted lookup, not a hard top-`k` selection;
- different attention heads can learn different notions of relevance;
- training knowledge is compressed into parameters rather than retained as explicit neighbors.

The clean mental model is: **attention is a learned, differentiable lookup over the current context**.

## 1. The task and the data contract

The environment is an `8 x 8` GridWorld in the main configs. A rendered RGB image contains a blue agent and red target. The action space is:

```text
0 stay
1 up
2 down
3 left
4 right
```

The environment owns the true state:

```text
(agent row, agent column, target row, target column)
```

The model never receives those four integers. It receives only the rendered image. This forces the vision encoder to recover the spatial information needed for control.

An expert policy uses Manhattan distance to select an action that moves toward the target. When vertical and horizontal moves are equally good, it randomly chooses an axis. This gives multiple correct shortest paths rather than one rigid route. See [`VisualGridWorld.expert_action`](../src/vat_mini/data.py#L58).

Each dataset item is one fixed-length trajectory:

| Field | Shape for one sample | Meaning |
| --- | --- | --- |
| `observations` | `[T, 3, H, W]` | RGB frame at every step |
| `actions` | `[T]` | action executed from that frame |
| `rewards` | `[T]` | immediate reward after that action |
| `valid_steps` | `[T]` | whether the timestep is real or padding |

After the agent reaches the target, the remaining array slots are padding. `valid_steps` is therefore part of the semantic contract, not bookkeeping: padded targets must contribute neither loss nor accuracy. Dataset creation is in [`generate_demonstrations`](../src/vat_mini/data.py#L139).

For a batch, PyTorch adds a batch axis:

```text
observations: [B, T, 3, H, W]
actions:      [B, T]
rewards:      [B, T]
valid_steps:  [B, T]
```

With the main config, `B=32`, `T=12`, `H=W=32`.

## 2. The vision encoder: image to state vector

An MLP expects a vector, but one frame is a `3 x H x W` image. The vision encoder is the adapter.

```text
[3, 32, 32]
  -> Conv + GELU
  -> Conv + GELU
  -> adaptive pooling to [channels, 4, 4]
  -> flatten
  -> linear projection
  -> [D]
```

In the main model, `D=96`. See [`VisionEncoder`](../src/vat_mini/model.py#L11).

The CNN learns local visual detectors and gradually combines them. The final linear projection creates one summary vector for the entire frame.

This is not a Vision Transformer. A ViT would normally split an image into patches and treat those patches as a sequence of tokens. VaT-mini uses a CNN and emits exactly **one token per frame**. Its transformer models relationships across time, not relationships among image patches.

One subtle design choice matters: the encoder pools to a `4 x 4` feature map before flattening instead of globally averaging all spatial positions. That preserves coarse location. If the encoder erased all location information, “blue is left of red” and “blue is right of red” could look identical to the policy.

## 3. Building one timestep token

The transformer requires every item in its sequence to have the same width `D`. VaT-mini creates the token at time `t` by adding three `D`-dimensional vectors:

```text
x_t = vision(frame_t) + action_embedding(action_(t-1)) + position_embedding(t)
```

Implemented in [`VisionActionTransformer.forward`](../src/vat_mini/model.py#L69).

Each term answers a different question:

| Term | Information it supplies |
| --- | --- |
| visual embedding | What does the world look like now? |
| previous-action embedding | What did the controller just do? |
| position embedding | Where is this token in the trajectory? |

Why add instead of concatenate? Addition is a cheap fusion mechanism that preserves width `D`. The model can learn compatible coordinate systems for the three embeddings. Concatenation would also work, but it would require another projection and more parameters.

The tradeoff is that addition does not preserve an explicit boundary between modalities. This is fine for a small model, but larger systems often use separate token types, cross-attention, or richer fusion blocks.

### Why the action must be shifted

The target at time `t` is `action_t`. Feeding `action_t` into the same token would reveal the answer.

Training therefore converts:

```text
target actions:   [a0, a1, a2, a3]
model inputs:     [BOS, a0, a1, a2]
```

`BOS` is a learned “begin action” ID. The shift is implemented in [`shifted_actions`](../src/vat_mini/model.py#L63).

This is the first anti-leakage boundary:

> The token predicting `action_t` may contain `action_(t-1)`, never `action_t`.

## 4. What the transformer adds

If each frame already shows both agent and target, an MLP over the current visual embedding could learn this toy task. The transformer is included because the repository is teaching the architecture needed when history matters: partial observability, motion, delayed effects, ambiguous frames, or longer plans.

An MLP processes each input row independently. Self-attention lets the representation at timestep `t` collect information from other allowed timesteps before an MLP processes it.

Karpathy's useful framing is:

```text
self-attention = communication between tokens
feed-forward MLP = computation inside each token
```

A transformer layer alternates those two jobs, with residual connections and normalization to keep training stable.

### Query, key, and value without unnecessary mystique

For each token vector `x`, a learned linear layer creates three new vectors:

```text
query: what information am I looking for?
key:   what kind of information do I contain?
value: what information should I send if selected?
```

For a query at timestep `i`, attention scores every permitted key `j`:

```text
score(i, j) = query_i dot key_j / sqrt(head_dimension)
```

Softmax turns those scores into weights that sum to one. The output is the weighted sum of the value vectors:

```text
attention_output_i = sum_j weight(i, j) * value_j
```

This resembles a learned soft KNN lookup: score candidates, normalize relevance, and aggregate their information.

### Why multiple heads exist

The main model has four heads. Each head works in its own learned subspace. One head might focus on the most recent frame, another on previous direction changes, and another on older visual evidence. Those examples are possible interpretations, not guaranteed behaviors; a head has no predefined meaning.

The 96-dimensional token is split across four 24-dimensional heads, their results are combined, and the layer projects back to width 96.

### What “encoder” means in this code

PyTorch calls the module `TransformerEncoder`, but VaT-mini supplies a causal mask. Functionally, it behaves like a decoder-only causal stack: it processes one stream and prevents access to the future.

So do not infer “bidirectional encoder” from the class name. The information-flow mask determines the semantics.

The layer is pre-norm (`norm_first=True`), batch-first, uses GELU, and contains self-attention plus a feed-forward network. See the model construction in [`model.py`](../src/vat_mini/model.py#L46) and the [PyTorch `TransformerEncoderLayer` reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html).

## 5. The causal mask: the central invariant

During training, the whole trajectory is processed in parallel. Without a mask, the token at time `0` could read frames and actions from later in the completed demonstration.

VaT-mini creates an upper-triangular Boolean mask:

```text
key timestep ->  0  1  2  3
query 0          yes no no no
query 1          yes yes no no
query 2          yes yes yes no
query 3          yes yes yes yes
```

The implementation is [`torch.triu(..., diagonal=1)`](../src/vat_mini/model.py#L86): entries above the diagonal are blocked.

The full no-leakage invariant is:

> A prediction at time `t` may use frames through `t` and actions before `t`, but it may not use future frames or the target action at `t`.

Two separate mechanisms enforce it:

1. right-shifting actions prevents same-step target leakage;
2. causal attention prevents future-token leakage.

The test [`test_causal_mask_blocks_future_observations`](../tests/test_model.py#L28) changes future observations and verifies that earlier logits remain unchanged. This is a strong architecture test because it checks observable behavior rather than an implementation detail.

## 6. Action prediction

After the transformer, every timestep has a contextualized vector of width `D`. A linear layer maps it to five logits:

```text
[B, T, D] -> Linear(D, 5) -> [B, T, 5]
```

A logit is an unnormalized action score. Softmax can turn the five scores into probabilities. Cross-entropy does that conversion internally during training.

During rollout, VaT-mini uses `argmax`: choose the action with the highest logit. There is no sampling, beam search, or planning algorithm. The network is the policy.

## 7. One complete forward-pass trace

Suppose one batch contains 32 trajectories of length 12.

```text
observations                         [32, 12, 3, 32, 32]
reshape frames for CNN               [384, 3, 32, 32]
CNN + projection                     [384, 96]
restore batch and time               [32, 12, 96]

shifted action IDs                   [32, 12]
action embeddings                    [32, 12, 96]
position embeddings                  [1, 12, 96]  # broadcasts across batch

sum three embedding streams          [32, 12, 96]
causal transformer, 2 layers         [32, 12, 96]
action head                          [32, 12, 5]
```

For a valid timestep, its five logits are compared with the expert action ID. For padding, the computed logits exist but the loss mask discards them.

## 8. Stage one: behavior cloning

Behavior cloning turns control into ordinary supervised classification:

```text
input:  observation and allowed history
label:  expert action
loss:   cross-entropy
```

For each valid timestep:

```text
loss_t = -log P(expert_action_t | allowed history)
```

The batch loss is the mean across valid timesteps. Rewards are ignored in this stage. See [`BehaviorCloningObjective`](../src/vat_mini/objectives.py#L18).

This is called “pretraining” in the repo only because it is the first policy-training stage. It is not internet-scale pretraining, masked-image pretraining, or self-supervised learning.

### Teacher forcing

During behavior-cloning training, the model receives the expert's previous action, not its own previous prediction. This is teacher forcing.

It makes training parallel and stable, but creates a distribution mismatch:

```text
training: previous action came from expert
rollout:  previous action came from model
```

If the model makes one mistake during rollout, it may enter a state rarely seen in expert demonstrations. The next prediction becomes harder, causing errors to compound. This is the classic imitation-learning covariate-shift problem described by [Ross, Gordon, and Bagnell's DAgger paper](https://proceedings.mlr.press/v15/ross11a.html).

## 9. Stage two: reward-weighted imitation

The post-training config starts from the behavior-cloning checkpoint, lowers the learning rate, and trains on demonstrations with 25% random action injection.

Noise creates a mix of better and worse decisions. Reward then provides a way to prefer actions associated with better remaining outcomes.

For timestep `t`, the code first computes undiscounted return-to-go:

```text
G_t = reward_t + reward_(t+1) + ... + reward_(T-1)
```

It standardizes valid returns across the batch:

```text
A_t = (G_t - mean(G_valid)) / (std(G_valid) + epsilon)
```

Then it creates a positive weight:

```text
w_t = min(exp(A_t / temperature), maximum_weight)
```

Finally, it computes weighted cross-entropy:

```text
loss = sum_t(w_t * cross_entropy_t) / sum_t(w_t)
```

Higher-return parts of the data influence the parameter update more. The weight is detached, so gradients flow through the policy loss, not through the return calculation. See [`AdvantageWeightedImitationObjective`](../src/vat_mini/objectives.py#L38).

### Important naming boundary

This stage is inspired by advantage-weighted regression (AWR), which uses supervised policy regression weighted by estimated advantages. The original [AWR paper](https://arxiv.org/abs/1910.00177) also learns a value function.

VaT-mini does **not** learn `V(state)` and does not calculate the conventional advantage `Q(state, action) - V(state)`. It standardizes observed return-to-go against a batch-wide mean. “Reward-weighted imitation” is the most literal name; “advantage-weighted imitation” is a useful approximation, not an exact implementation of full AWR.

It is also not online reinforcement learning. Training never asks the current policy to gather new experience and then improve from it.

## 10. The training runtime and state ownership

The `Trainer` owns mutable training state:

- model parameters;
- AdamW optimizer state;
- selected objective;
- global optimizer-step count;
- checkpoint manager;
- metrics tracker.

One optimizer step is:

```text
batch tensors -> device
shift target actions
forward pass
masked objective
zero old gradients
backpropagate
clip gradient norm
AdamW update
increment global_step
```

Implemented in [`Trainer._train_epoch`](../src/vat_mini/trainer.py#L115).

One epoch is one full pass over the training DataLoader. After each epoch, the trainer evaluates held-out demonstrations and writes metrics/checkpoints. At the end, it also runs closed-loop episodes.

The software boundaries are intentionally conventional:

```text
CLI -> constructs config, data loaders, model, device, trainer
Trainer -> coordinates learning and persistence
Model -> knows tensor transformations, not files or CLI flags
Objective -> knows how to score predictions, not optimizer state
Evaluation -> reads a trained model without mutating it
Checkpoint manager -> owns persistence format and atomic replacement
```

That separation makes experiments replaceable. For example, the behavior-cloning objective can change without rewriting the model, and W&B can be disabled without changing the trainer's control flow.

## 11. Training-time execution versus rollout execution

These paths look similar but answer different questions.

### Teacher-forced evaluation

```text
entire held-out expert trajectory
-> model receives shifted expert actions
-> compare every predicted action with expert action
-> validation loss and token accuracy
```

This asks: **when the history is clean, can the model imitate the expert's next action?**

### Closed-loop rollout

```text
reset environment
-> observe current image
-> model chooses action
-> environment changes
-> append model action and new image to history
-> repeat
```

This asks: **when the model owns the history it creates, does it reach the target?**

The loop is implemented in [`evaluate_rollouts`](../src/vat_mini/evaluation.py#L59). Its main metrics are:

- success rate: fraction of episodes that reach the target;
- path efficiency: shortest-path length divided by actual steps, or zero on failure;
- mean return: average accumulated environment reward.

Closed-loop success is the more important policy metric. High teacher-forced accuracy can hide catastrophic compounding errors.

## 12. Autoregressive inference, step by step

At the first step:

```text
observations = [frame_0]
actions      = []
model input  = [(frame_0, BOS, position_0)]
output       = action_0
```

At the second step:

```text
observations = [frame_0, frame_1]
actions      = [action_0]
model input  = [
  (frame_0, BOS,      position_0),
  (frame_1, action_0, position_1),
]
output = last-position prediction = action_1
```

The history is recomputed on every step. This is simple and correct for short sequences, but inefficient for long ones. Production autoregressive transformers usually cache per-layer keys and values so earlier history is not recomputed.

If history exceeds `max_sequence_length`, [`choose_action`](../src/vat_mini/model.py#L92) keeps only the most recent window.

## 13. What the model can and cannot learn

It can learn:

- visual location features for agent and target;
- mappings from spatial relationships to discrete movement actions;
- temporal correlations within its context window;
- how previous actions relate to current observations;
- which demonstrated actions deserve more weight under the post-training objective.

It cannot inherently:

- search a map or run an explicit shortest-path algorithm;
- understand language instructions;
- output continuous motor commands;
- remember beyond the context window;
- recover from arbitrary unseen states reliably;
- infer uncertainty or safety constraints from this objective;
- generalize to real robot images merely because it consumes RGB.

The learned policy may approximate the expert's rule, but the code does not contain a symbolic planner inside the model.

## 14. Why call it a vision-action transformer?

“Vision-action” describes the input/output contract:

```text
visual observations -> action predictions
```

“Transformer” describes the temporal sequence processor.

This places the project in the same broad family as robotics transformers, but at a dramatically smaller scale. For example, [RT-1](https://arxiv.org/abs/2212.06817) learns from large, diverse real-robot data and tokenizes robot actions. VaT-mini instead uses synthetic GridWorld images, one task, five discrete actions, a tiny CNN, and two transformer layers.

The architectural rhyme is real:

```text
encode observations -> process context -> predict actions autoregressively
```

The capability comparison is not.

## 15. Deliberate simplifications and their upgrade paths

| Current choice | Why it is useful here | Likely larger-system replacement |
| --- | --- | --- |
| one CNN token per frame | easy tensor flow | patch/spatial tokens or pretrained vision encoder |
| additive token fusion | minimal code | typed tokens, cross-attention, or multimodal projector |
| five discrete actions | simple classification | tokenized multi-dimensional or continuous actions |
| fixed history recomputation | transparent | KV cache and streaming inference |
| behavior cloning | stable first objective | DAgger-style data collection, offline RL, or richer imitation learning |
| return-weighted post-training | shows reward preference simply | learned critic/value baseline and principled offline RL |
| full causal attention | simple and exact | optimized attention or bounded temporal memory |
| synthetic expert | unlimited deterministic labels | recorded teleoperation or real/sim robot trajectories |
| one fixed task | isolates architecture | language/task conditioning and diverse datasets |

## 16. Failure modes worth watching

### Data leakage

If current target actions or future frames become visible to earlier predictions, training metrics can look excellent while rollout fails. The shifted-action and causal-mask invariants are non-negotiable.

### Padding leakage

If padded steps contribute to loss, the model learns artificial `stay` targets and accuracy becomes misleading.

### Action imbalance

A model can look competent by overpredicting the most common action. Validation therefore reports the majority-class baseline.

### Exposure bias

Teacher forcing evaluates histories created by the expert. Deployment uses histories created by the model. A small mistake changes the future input distribution.

### Shortcut learning

The environment is visually regular. The model may learn color/layout shortcuts that do not survive appearance changes.

### Reward-weighting instability

Exponentiating standardized returns can create very large weights. Temperature and maximum-weight clipping control this, but batch composition still affects the weights.

### Metric confusion

Token accuracy measures local imitation. Success rate measures control. A useful policy needs both, but rollout behavior is the final authority.

## 17. A practical code-reading order

Read one boundary at a time:

1. [`config.py`](../src/vat_mini/config.py) — the experiment schema and legal values.
2. [`data.py`](../src/vat_mini/data.py) — where observations, labels, rewards, and masks come from.
3. [`model.py`](../src/vat_mini/model.py) — the prediction function and causal contract.
4. [`objectives.py`](../src/vat_mini/objectives.py) — why a parameter update is considered better.
5. [`trainer.py`](../src/vat_mini/trainer.py) — ownership of mutable training state.
6. [`evaluation.py`](../src/vat_mini/evaluation.py) — teacher forcing versus actual control.
7. [`checkpoint.py`](../src/vat_mini/checkpoint.py) — the persisted state boundary.
8. [`cli.py`](../src/vat_mini/cli.py) — composition and operator interface.

When reading, keep asking four SWE-style questions:

```text
What is the input contract?
Who owns this state?
What information is this component allowed to see?
How is correctness measured at this boundary?
```

## 18. Concepts to remember

If only ten things stick, make them these:

1. The model predicts `action_t` from images through `t` and actions before `t`.
2. The CNN converts each frame into one token; this is not a ViT patch encoder.
3. A timestep token adds visual, previous-action, and positional embeddings.
4. Attention is a learned soft lookup over tokens in the current context.
5. The transformer alternates cross-token communication with per-token MLP computation.
6. Right-shifted actions and the causal mask are separate anti-leakage mechanisms.
7. Behavior cloning is supervised next-action classification.
8. Teacher-forced accuracy does not prove closed-loop control.
9. Post-training upweights actions associated with higher return-to-go, but is not full AWR.
10. VaT-mini demonstrates the architecture's boundaries, not real-world robotics capability.

## References and next learning steps

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — original transformer architecture and scaled dot-product attention.
- Andrej Karpathy, [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) — especially useful for attention as token communication, causal masking, and building the mechanism in code.
- Ross, Gordon, and Bagnell, [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html) — why sequential prediction violates the simple i.i.d. picture and why behavior-cloning errors compound.
- Peng et al., [Advantage-Weighted Regression](https://arxiv.org/abs/1910.00177) — the fuller algorithm that motivates reward/advantage-weighted supervised policy updates.
- Brohan et al., [RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817) — a useful comparison point for how the basic observation-to-action pattern scales toward real robotics.
- PyTorch, [`TransformerEncoderLayer`](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html) — exact semantics of the reference layer used by this implementation.

A good learning sequence is:

```text
trace one dataset sample
-> trace all tensor shapes through forward()
-> derive one attention row by hand
-> verify the causal mask
-> compare teacher-forced and rollout execution
-> derive behavior-cloning loss
-> derive reward-weighted loss
-> change one architecture boundary and predict the effect before training
```
