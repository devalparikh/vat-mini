# Closed-loop simulator rollouts (RoboMimic)

By default the periodic `rollout/video` for the RoboMimic (continuous-action)
config is a **teacher-forced demonstration replay**: the same recorded validation
episode every epoch, where the policy's only contribution is the predicted-vs.-
ground-truth action MAE shown in the caption. It never drives anything.

With `tracking.sim_rollout: true` the rollout instead drives the policy in a real
[robosuite](https://robosuite.ai) simulator. The video then shows the policy
actually attempting the task, and a genuine `rollout_success_rate` is reported —
a far more meaningful signal than action MAE, since a low-MAE policy can still
fail over a long horizon as small errors compound.

This was validated on macOS (Apple Silicon, MPS training). MuJoCo offscreen
rendering works fine there via `MUJOCO_GL=cgl` (set automatically on Darwin).

## Usage

```bash
make setup-sim                 # installs the `[sim]` extra (see caveats below)
# configs/robomimic-can.yaml already sets tracking.sim_rollout: true
make robomimic-can
```

Relevant config (under `tracking:`):

| Key | Default | Meaning |
|---|---|---|
| `sim_rollout` | `false` | Enable closed-loop sim rollouts (RoboMimic only) |
| `sim_rollout_episodes` | `5` | Episodes for the end-of-training `rollout_success_rate` |
| `sim_rollout_max_steps` | `400` | Max steps per episode (200 in the robomimic-can config) |

If the simulator is unavailable on the machine, the trainer prints a warning,
falls back to the demonstration replay, and disables further attempts for the
run (so it does not retry and spam every epoch).

## How it works

`src/vat_mini/robomimic_rollout.py` is the **only** module that touches the
simulator. Everything is imported lazily and any build/render failure raises
`SimRolloutUnavailable`, which the trainer catches.

- The environment is rebuilt with `robosuite.make` from the `env_args` embedded
  in the dataset HDF5 (`data.attrs["env_args"]`) — the exact task, robot,
  controller, and cameras the data was collected with. RoboMimic's own env
  wrapper is **not** used because it still imports the deprecated `mujoco_py`.
- The policy is stepped autoregressively via
  `VisionActionTransformer.choose_action_continuous`: at each step it sees the
  frames it has already observed plus the actions it has already taken.
- Camera frames are preprocessed with `data.frames_to_observation`, the exact
  transform used in training, so the policy sees identically-processed pixels.
- Results are logged through `tracker.log_sim_rollout` as a `SimRolloutTrace`
  (video played back near real time, plus `rollout/success`, `rollout/return`,
  `rollout/steps`, and a per-step action table).

## Dependency notes / pitfalls

The datasets were collected with an older robosuite; running them against a
modern, macOS-installable stack required pinning and two shims (all handled in
code — listed here so they are not re-derived):

- **`egl_probe` needs CMake** and fails on CMake 4.x (it declares
  `cmake_minimum_required(VERSION 2.8.12)`). Build it with the env var
  `CMAKE_POLICY_VERSION_MINIMUM=3.5` and `--no-build-isolation` so it uses a
  working CMake. The C++ itself compiles fine on macOS.
- **`robosuite==1.4.1`** is pinned: 1.5.x rejects the dataset's flat `OSC_POSE`
  controller config format.
- **`mujoco>=3.1,<3.11`** is used (not the 2.3.x that robosuite 1.4.1 targets)
  because 2.3.x has no Python 3.12 / arm64 wheels. The one incompatible call,
  `mj_fullM` (its signature changed in mujoco 3.x), is bridged by a NumPy
  reimplementation of the sparse→dense inertia expansion in
  `robomimic_rollout.py` — it needs only the model, so it is version-independent.
- **Image orientation**: `robosuite.macros.IMAGE_CONVENTION` is forced to
  `"opencv"` so rendered frames come out upright, matching how frames are stored
  in the dataset (and thus what the policy saw during training).

## Known limitation: sim-render domain gap

Frames rendered by robosuite 1.4.1 differ from the stored training frames by
~28/255 mean pixel error (lighting/textures) even with correct orientation,
because the data was collected with a different robosuite build. The policy was
trained on slightly different-looking pixels, so **early sim success rates can
read low even when action MAE is good** — this is a domain gap, not necessarily
a bad policy. For a fully faithful evaluation, rendering with the exact
collection-time robosuite build (typically on Linux) would close this gap.
