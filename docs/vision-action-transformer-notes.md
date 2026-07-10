# Vision-action transformer architecture

## The core idea

A vision-action transformer is a policy that predicts actions from visual history.

```text
past and current images + past actions -> next action
```

At timestep `t`, the model represents:

```text
P(action_t | image_0 ... image_t, action_0 ... action_(t-1))
```

This is the central contract. The model may use the current observation and past experience, but it cannot use future observations or the action it is currently supposed to predict.

## The architecture at a glance

```text
Environment       -> produces the world state the agent must respond to
Observation       -> captures the visible state as an image
Vision encoder    -> turns each image into a useful visual vector
Action embedding  -> turns the previous action ID into a vector
Position embedding -> tells the model where this timestep occurs in the sequence
Timestep token    -> combines visual, previous-action, and position information
Causal transformer -> lets each timestep use relevant earlier context
Action head       -> turns each transformed token into action scores
Action selection  -> converts the newest action scores into the executed action
Environment step  -> applies that action and produces the next observation
```

The execution loop is:

```text
observe -> encode -> add context -> transform -> score -> act -> observe again
```

### Environment

The system being controlled. It owns the real state and defines what actions do. This could be a GridWorld, simulator, video game, or robot.

### Observation

The information exposed to the model at one moment. Here it is an image. An observation does not have to reveal the complete underlying state.

### Vision encoder

A neural network that converts raw pixels into a smaller vector containing features useful for predicting actions. It creates the representation consumed by the sequence model.

### Action embedding

A learned lookup table that converts a discrete action ID into a vector. This lets previous actions live in the same vector space as the visual representation.

### Position embedding

A learned vector representing the timestep index. Attention alone does not inherently know which token came first, so position information provides sequence order.

### Timestep token

The complete input representation for one moment. A simple vision-action model creates it by adding:

```text
visual vector + previous-action vector + position vector
```

### Causal transformer

The temporal reasoning component. It lets a timestep combine its own information with useful earlier information while blocking access to the future.

### Action head

A final learned projection that converts the transformer's output vector into one score per available action.

### Action selection

The rule that turns action scores into a decision. A simple policy uses the highest-scoring action. Other systems may sample, apply constraints, or pass the scores to a planner.

### Environment step

The environment applies the selected action, changes its state, and returns the next observation and usually a reward or completion signal. This closes the control loop.

## The learning system around the architecture

The model above defines how predictions are made. These components define how it learns and how its behavior is measured:

```text
Trajectory       -> one ordered sequence of observations, actions, and rewards
Dataset          -> stores many trajectories used as training examples
Expert policy    -> provides the desired action labels for demonstrations
Objective        -> converts model predictions and targets into one loss value
Optimizer        -> uses gradients to update the model's parameters
Trainer          -> owns and coordinates the repeated training process
Checkpoint       -> saves learned parameters and training state
Evaluation       -> measures predictions without updating the model
Rollout          -> lets the model control the environment using its own actions
```

### Trajectory

One episode or partial episode arranged in time order:

```text
observation_0, action_0, reward_0,
observation_1, action_1, reward_1,
...
```

### Dataset

A collection of trajectories converted into tensors and batches. It defines the experience from which the model can learn.

### Expert policy

The source of desired actions in imitation learning. It may be a human, scripted controller, planner, or stronger model.

### Objective

The mathematical rule that decides what counts as a better prediction. Behavior cloning uses cross-entropy against expert actions. Reward-weighted imitation gives more influence to higher-return actions.

### Optimizer

The algorithm that changes model parameters using their gradients. The objective produces the loss; backpropagation produces gradients; the optimizer applies the update.

### Trainer

The runtime coordinator. It loads batches, runs the model, calculates loss, performs backpropagation, calls the optimizer, records metrics, and saves checkpoints.

### Checkpoint

A saved snapshot of learned model parameters and, when needed, optimizer and training state. It lets training continue later or lets evaluation reload a trained policy.

### Evaluation

A measurement pass with parameter updates disabled. It can measure next-action prediction on held-out trajectories or task success in the environment.

### Rollout

A closed-loop episode where the model's chosen action changes the next input it receives. Rollouts reveal compounding mistakes that teacher-forced prediction metrics can hide.

## Important vocabulary

```text
Token          -> one fixed-width vector processed at one sequence position
Embedding      -> a learned conversion from an ID or input into a vector
Context        -> the observations and actions available to the current prediction
Attention      -> learned weighted information sharing between tokens
Attention head -> one independently learned attention lookup
Causal mask    -> blocks each timestep from reading future timesteps
Logits         -> raw action scores before normalization
Softmax        -> converts logits into probabilities that sum to one
Loss           -> one number measuring how wrong the predictions are
Gradient       -> how each parameter should change to reduce the loss
Teacher forcing -> training with real previous actions instead of model actions
Return         -> total reward accumulated over part or all of a trajectory
Policy         -> the rule or model that selects an action from available context
```

## What attention does

Each timestep begins with only its own information: the current visual representation, the previous action, and its position in the sequence.

Attention lets that timestep retrieve information from other allowed timesteps. It does this in three stages:

1. score how relevant every allowed timestep is;
2. turn those scores into normalized weights;
3. combine the other timesteps' information using those weights.

The relevance calculation is learned during training. The model is not given rules such as “always use the newest frame” or “look back two steps.” It learns which relationships help reduce action-prediction loss.

The result is a new representation containing both the timestep's original information and useful context from its history.

## Turning vision into tokens

Transformers process sequences of vectors called tokens. Images must first be converted into that format.

Two common approaches are:

### One token per frame

```text
frame -> CNN -> one visual vector
```

This is compact and works well when the scene is simple. The temporal transformer operates across frames.

### Multiple patch tokens per frame

```text
frame -> image patches -> one vector per patch
```

This preserves more spatial detail but creates a much longer sequence. The model must reason across both space and time.

A CNN that emits one vector per frame is not a Vision Transformer. A ViT normally treats image patches as tokens and processes their spatial relationships with a transformer.

## Constructing a timestep token

A common design combines three pieces of information:

```text
x_t = vision(frame_t) + action_embedding(action_(t-1)) + position_embedding(t)
```

Each term answers a different question:

| Term | Question answered |
| --- | --- |
| Visual embedding | What does the world look like now? |
| Previous-action embedding | What did the controller just do? |
| Position embedding | Where is this moment in the trajectory? |

Addition is a lightweight fusion method. Because all three vectors have the same width, they can be added without increasing the token size.

Larger systems may instead use separate token types, projections, or cross-attention. Those approaches preserve clearer modality boundaries but require more machinery.

## Why actions are shifted right

The target at time `t` is `action_t`. Giving that action to the input token would reveal the answer.

The input action sequence is therefore shifted:

```text
target actions: [a0,  a1, a2, a3]
input actions:  [BOS, a0, a1, a2]
```

`BOS` means “beginning of sequence.” It fills the previous-action slot before any action has occurred.

The invariant is:

> The token predicting `action_t` may contain `action_(t-1)`, never `action_t`.

## What the transformer adds

An MLP processes one input vector independently. A transformer lets each timestep retrieve relevant information from other timesteps before applying an MLP.

A useful decomposition is:

```text
self-attention    = communication between tokens
feed-forward MLP  = computation inside each token
```

A transformer layer alternates those jobs. Residual connections preserve the existing representation, while normalization keeps optimization stable.

## Query, key, and value

Each token is projected into three learned vectors:

```text
query: what information am I looking for?
key:   what kind of information do I contain?
value: what information should I send if selected?
```

For a query at timestep `i`, the model scores an allowed key at timestep `j`:

```text
score(i, j) = query_i dot key_j / sqrt(head_dimension)
```

Softmax turns the scores into weights that sum to one:

```text
weight(i, j) = softmax(score(i, j))
```

The result is a weighted combination of values:

```text
output_i = sum_j weight(i, j) * value_j
```

The query and key decide relevance. The value carries the information.

## Why multiple heads exist

Each attention head performs its own lookup in a different learned subspace.

For a control policy, different heads could focus on:

- the newest observation;
- a previous direction change;
- an older visual clue;
- whether a recent action produced the expected result.

These are possible learned behaviors, not predefined roles. A head receives no human-written meaning.

## Causal attention

Training can process an entire completed trajectory in parallel. Without a mask, an early timestep could inspect later observations.

A causal mask blocks that access:

```text
key timestep ->  0    1    2    3
query 0          yes  no   no   no
query 1          yes  yes  no   no
query 2          yes  yes  yes  no
query 3          yes  yes  yes  yes
```

The complete anti-leakage rule requires two mechanisms:

1. shifted actions prevent access to the current target action;
2. causal attention prevents access to future tokens.

These solve different leakage problems. A model needs both.

## From transformer output to an action

The transformer produces one contextualized vector per timestep. A linear action head maps each vector to one score per possible action:

```text
context vector -> linear layer -> action logits
```

A logit is an unnormalized score. Softmax converts the scores into probabilities.

During deployment, a policy might:

- choose the highest-scoring action with `argmax`;
- sample from the probability distribution;
- apply constraints before selecting an action;
- use the scores inside a larger planning system.

The simplest policy chooses `argmax` directly.

## Behavior cloning

Behavior cloning treats control as supervised classification:

```text
input: observation and allowed history
label: expert action
loss:  cross-entropy
```

At each valid timestep:

```text
loss_t = -log P(expert_action_t | allowed history)
```

The model learns to imitate the expert actions contained in its dataset. It does not discover actions through trial and error.

## Teacher forcing and exposure bias

During training, the model is usually given the expert's previous action. This is teacher forcing.

```text
training: previous action came from the expert
rollout:  previous action came from the model
```

Teacher forcing makes training stable and parallel. The downside is distribution mismatch.

If the deployed model makes one mistake, it may enter a state that rarely appeared in the expert data. Its next prediction becomes less reliable, which can cause further mistakes. This is exposure bias, or covariate shift in imitation learning.

The important consequence is:

> High next-action accuracy on expert histories does not prove that the policy works when controlling the environment itself.

## Reward-weighted imitation

Plain behavior cloning treats every demonstrated action equally. Reward-weighted imitation assigns more influence to actions associated with better outcomes.

First calculate return-to-go:

```text
G_t = reward_t + reward_(t+1) + ... + reward_(T-1)
```

Then create a positive weight from the relative quality of that return:

```text
w_t = exp(advantage_t / temperature)
```

Finally weight the supervised action loss:

```text
loss = sum_t(w_t * cross_entropy_t) / sum_t(w_t)
```

The policy still imitates actions from a fixed dataset, but actions associated with better remaining outcomes have greater influence.

Full advantage-weighted regression typically learns a value function and estimates:

```text
advantage(state, action) = Q(state, action) - V(state)
```

A simpler educational implementation may approximate advantage using normalized observed returns. That demonstrates the weighting idea but is not the complete AWR algorithm.

## Teacher-forced evaluation versus closed-loop evaluation

These evaluations answer different questions.

### Teacher-forced evaluation

```text
held-out expert trajectory
-> model receives previous expert actions
-> predictions are compared with expert actions
```

This asks:

> Can the model imitate the next action when its history is clean?

### Closed-loop evaluation

```text
observe environment
-> model chooses action
-> environment changes
-> model observes the result
-> repeat
```

This asks:

> Can the policy recover from and continue through the history it creates itself?

Closed-loop metrics often include:

- task success rate;
- path or action efficiency;
- accumulated return;
- safety violations;
- recovery after mistakes.

For a deployed policy, closed-loop behavior is the final authority.

## Autoregressive inference

At the first step:

```text
observations = [frame_0]
actions      = []
input token  = frame_0 + BOS + position_0
output       = action_0
```

At the second step:

```text
observations = [frame_0, frame_1]
actions      = [action_0]
input tokens = [
  frame_0 + BOS      + position_0,
  frame_1 + action_0 + position_1,
]
output = action_1
```

The selected action becomes part of the next input. That feedback loop is what makes execution autoregressive.

Simple systems recompute the whole history every step. Larger transformers usually cache per-layer keys and values so past tokens do not need to be recomputed.

## What this architecture learns

Depending on its data, the model can learn:

- visual features relevant to control;
- spatial relationships between objects;
- temporal correlations and motion;
- relationships between earlier actions and later observations;
- mappings from state and history to action probabilities.

It does not automatically gain:

- an explicit search or planning algorithm;
- unlimited memory;
- language understanding;
- continuous motor control;
- calibrated uncertainty;
- safety guarantees;
- real-world generalization from synthetic training data.

A neural policy can approximate the behavior of a planner without containing a symbolic planner.

## Vision-action models versus larger robotics transformers

The reusable architecture pattern is:

```text
encode observations -> process context -> predict actions autoregressively
```

Large robotics models extend it with:

- pretrained visual representations;
- multiple spatial tokens per frame;
- language or task conditioning;
- diverse robots and environments;
- tokenized multi-dimensional actions;
- much larger datasets and parameter counts;
- deployment-time safety and control layers.

Sharing the pattern does not imply sharing the capabilities.

## Common failure modes

### Target leakage

The model sees the current target action or future observations during training. Metrics become unrealistically strong and deployment fails.

### Padding leakage

Padded timesteps contribute to loss and create fake training targets.

### Action imbalance

The model achieves misleading accuracy by predicting the most common action.

### Exposure bias

The model performs well on expert histories but cannot recover from its own mistakes.

### Shortcut learning

The vision encoder relies on superficial colors, layouts, or rendering artifacts that do not generalize.

### Reward-weight instability

Exponentiated weights allow a small number of samples to dominate training. Temperature and maximum-weight clipping help control this.

### Metric confusion

Next-action accuracy is mistaken for task-level competence.

## What to remember

1. A vision-action transformer predicts the next action from visual and action history.
2. The vision encoder converts pixels into tokens the transformer can process.
3. Attention is a learned soft lookup across the current context.
4. Self-attention communicates across tokens; the feed-forward MLP processes each token.
5. Previous actions must be shifted so the current target is never exposed.
6. A causal mask prevents future observations from leaking into earlier predictions.
7. Behavior cloning is supervised next-action prediction from expert data.
8. Teacher forcing creates a gap between training and deployment histories.
9. Reward weighting prefers higher-return parts of a fixed dataset but is not automatically full reinforcement learning.
10. Closed-loop success matters more than teacher-forced accuracy.

## References

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Andrej Karpathy, [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- Ross, Gordon, and Bagnell, [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html)
- Peng et al., [Advantage-Weighted Regression](https://arxiv.org/abs/1910.00177)
- Brohan et al., [RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817)
