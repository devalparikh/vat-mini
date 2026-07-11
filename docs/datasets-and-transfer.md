# Datasets, embeddings, and transfer learning

Conceptual notes on how this model represents actions, and what happens if you
train on other datasets (different tasks, cameras, or robot arms). This is
background/intuition, not a how-to — the runnable path is
[docs/sim-rollout.md](sim-rollout.md).

## Embedding space vs. action tokenization

These are two different things and it's easy to conflate them.

- **Embedding space** — the shared latent vector space (`embedding_dim`, 128 here)
  that the transformer operates in. Every input is mapped into it and summed into
  one token per timestep: `vision(frame) + embed(prev_action) + position`. This
  model **does** use an embedding space.
- **Action tokenization** — *discretizing* a continuous action into a vocabulary
  of bins and predicting them like word tokens (e.g. RT-1/RT-2 bucket each action
  dimension into 256 classes, chosen via softmax). This model **does not** do this.

How this model actually handles each action type:

| | Discrete (GridWorld) | Continuous (RoboMimic) |
|---|---|---|
| Previous action → token | `nn.Embedding` lookup table | `nn.Linear(action_dim → embedding_dim)` projection |
| Action output | softmax over `num_actions` classes | regress `action_dim` floats, `tanh` head |
| Tokenized? | roughly (a lookup table) | **no — continuous regression** |

So for RoboMimic the 7-DoF action is **projected** into the embedding space on the
way in (a continuous linear map, not a lookup) and **regressed** as 7 real numbers
on the way out. Embedding space yes; action tokenization no.

## Training on other datasets

Two independent layers, and they behave very differently.

### Training is largely simulator-agnostic
The model only consumes `(camera frames, actions)`. Any dataset reshaped into the
loader's layout (`obs/<camera>`, `actions`, fixed-length windows) can *train* the
policy — no simulator involved. See [docs/sim-rollout.md](sim-rollout.md) for the
part that *is* simulator-specific (closed-loop rollouts only work in a sim you
have wired in; robosuite here).

### Does diverse data help? It depends on *what* varies
Mixing tasks/cameras/arms is the premise behind robot foundation models
(Open X-Embodiment, Octo, π0). It can help — but not for free, and mostly at
scale/capacity this model doesn't have.

- **Benefits when it works:** richer visual representations (more scenes, objects,
  lighting → less brittle), shared low-level skills (reach/grasp/lift recur), and
  positive transfer to data-poor tasks.
- **The embodiment gap is the main cost:** a different arm has a *different action
  space*. A 7-DoF Panda delta-EEF action means nothing to another robot's joint
  commands. Naively mixing them makes the model fit contradictory action
  distributions → **negative transfer** (worse, not better). Big models handle
  this with per-dataset normalization, action tokenization, or an
  embodiment/task ID the model conditions on.
- **Camera/viewpoint gap:** different pose/intrinsics changes pixels a lot — good
  for representation robustness, bad if the policy overfits to one view.

### For *this* small model specifically
Naive cross-embodiment mixing would likely **hurt**: it's tiny (3 layers, 128-dim),
its action head is a fixed 7-D `tanh` tied to Panda delta-EEF, and it's
single-camera with no task/embodiment conditioning. A small model on diverse data
tends to average conflicting behaviors rather than get smarter — bigger diversity
of knowledge needs a bigger model. Lower-risk options, in order:

1. **Same arm (Panda), different tasks** (Lift/Square/Transport). Same action
   space, richer visual + skill data — the most promising, low-effort experiment.
2. **Pretrain the vision encoder on diverse images, then fine-tune the policy on
   Can** (see below).
3. **Add a task/embodiment ID embedding** (an extra token) if you genuinely want
   to mix heterogeneous data — the minimal architectural change to avoid negative
   transfer.

## Why pretraining the vision encoder helps

Intuition: **"understand the scene" is the hard, general, reusable part; the
policy is small and task-specific.**

- The CNN's job is to turn pixels into a useful summary (where's the can, where's
  the gripper, the geometry). That visual skill is the same across Can/Lift/Square.
- Learning to see **using only action-prediction as the signal is slow and
  data-hungry** — "did the action match" is a weak, indirect teacher for vision.
- Pretraining the encoder on lots of diverse images means it *already* extracts
  clean spatial/object features before policy training starts. The tiny policy on
  top then only learns the easy last-mile map: good features → action. Less data,
  less overfitting to one camera.
- Crucially this **sidesteps the embodiment problem**: images have no action space,
  so diverse *visual* data can't cause the action-conflict that mixing diverse
  *action* data does. You get the representation benefit without the cost.

Analogy: transfer learning with an ImageNet-pretrained backbone or a pretrained
language encoder — don't relearn "how to see" per task; learn it once, broadly,
then attach a small task head.

**One-liner:** seeing is general and reusable; acting is specific — pretrain the
eyes on cheap diverse data, and spend the small policy budget only on the arm.
