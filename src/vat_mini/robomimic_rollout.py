"""Closed-loop RoboMimic rollouts for continuous-action policies.

This module is the only place that depends on the robosuite/MuJoCo simulator
stack. Everything is imported lazily so the rest of the package (and
teacher-forced evaluation) keeps working when those optional, hard-to-install
dependencies are absent. Any failure to build or render the environment raises
``SimRolloutUnavailable`` so callers can degrade gracefully to a demonstration
replay instead of crashing training.

The environment is rebuilt directly with ``robosuite.make`` from the ``env_args``
recorded in the dataset HDF5, rather than through RoboMimic's env wrapper: that
wrapper still imports the deprecated ``mujoco_py`` binding, which does not
install on modern machines. Two compatibility shims let the robosuite 1.4.x that
collected these datasets run against a current ``mujoco`` 3.x:

* ``mj_fullM`` changed signature in mujoco 3.x, so we reimplement the sparse to
  dense inertia-matrix conversion in NumPy (it needs only the model).
* the camera image convention is forced to ``opencv`` so rendered frames come
  out upright, matching how frames are stored in the dataset (and thus what the
  policy saw during training).

The policy is stepped autoregressively: at each timestep it sees the history of
camera frames it has already observed plus the actions it has already taken, and
predicts the next action, which is applied to the simulator. Camera frames are
preprocessed with :func:`vat_mini.data.frames_to_observation`, the exact
transform used in training, so the pixels match.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

from vat_mini.config import ExperimentConfig
from vat_mini.data import frames_to_observation
from vat_mini.evaluation import SimRolloutTrace
from vat_mini.model import VisionActionTransformer


class SimRolloutUnavailable(RuntimeError):
    """Raised when the simulator cannot be imported, built, or rendered."""


def _install_mujoco_compat() -> None:
    """Make robosuite 1.4.x run against mujoco 3.x.

    ``mj_fullM`` gained a new signature in mujoco 3.x (it now reads the sparse
    inertia from ``MjData`` instead of taking it as an argument), which robosuite
    1.4.x calls the old way. We replace it with a NumPy reimplementation of the
    original sparse-to-dense expansion, which depends only on the model's degree
    of freedom tree, so it is independent of the installed mujoco version.
    """
    import mujoco

    if getattr(mujoco.mj_fullM, "_vat_mini_shim", False):
        return

    def mj_fullM(model, dense, sparse):  # noqa: ANN001 — mirrors the C signature
        result = np.asarray(dense)
        result.fill(0.0)
        addresses = model.dof_Madr
        parents = model.dof_parentid
        for row in range(model.nv):
            address = int(addresses[row])
            column = row
            while column >= 0:
                result[row, column] = result[column, row] = sparse[address]
                address += 1
                column = int(parents[column])

    mj_fullM._vat_mini_shim = True
    mujoco.mj_fullM = mj_fullM


def _build_environment(config: ExperimentConfig):
    """Rebuild the recorded robosuite environment from the dataset metadata.

    The dataset HDF5 embeds the exact ``env_args`` (task, robot, controller,
    cameras) used to collect it, so we rebuild the identical env rather than
    guessing its configuration.
    """
    # MuJoCo needs a GL backend for offscreen rendering; CGL is the native macOS
    # one. Respect an explicit choice (e.g. ``egl`` on a Linux GPU box).
    if "MUJOCO_GL" not in os.environ and sys.platform == "darwin":
        os.environ["MUJOCO_GL"] = "cgl"

    try:
        _install_mujoco_compat()
        import robosuite
        import robosuite.macros as macros
    except ImportError as error:
        raise SimRolloutUnavailable(
            "RoboMimic sim rollout needs the sim extra; install with "
            "`make setup-sim` or `pip install -e '.[sim]'`"
        ) from error

    # Store rendered frames upright, matching the dataset's stored orientation.
    macros.IMAGE_CONVENTION = "opencv"

    dataset_path = config.data.dataset_path
    try:
        import h5py

        with h5py.File(dataset_path, "r") as archive:
            env_args = json.loads(archive["data"].attrs["env_args"])
    except Exception as error:  # noqa: BLE001 — missing file/attrs surface uniformly
        raise SimRolloutUnavailable(
            f"could not read env metadata from dataset {dataset_path}: {error}"
        ) from error

    env_kwargs = dict(env_args["env_kwargs"])
    # Force offscreen-only rendering regardless of how the data was collected.
    env_kwargs.update(has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True)
    try:
        environment = robosuite.make(env_name=env_args["env_name"], **env_kwargs)
    except Exception as error:  # noqa: BLE001 — build/render failures surface uniformly
        raise SimRolloutUnavailable(
            f"could not build the robosuite environment (often MuJoCo offscreen "
            f"rendering on this platform): {error}"
        ) from error
    return environment, config.data.camera_key


def _camera_frame(observation: dict, camera_key: str) -> np.ndarray:
    """Extract an ``[H, W, 3]`` uint8 frame from a robosuite observation dict."""
    frame = np.asarray(observation[camera_key])
    if frame.ndim != 3:
        raise SimRolloutUnavailable(f"unexpected camera frame shape: {frame.shape}")
    if frame.dtype != np.uint8:  # already upright via the opencv convention
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    return frame


@torch.no_grad()
def record_sim_rollout(
    model: VisionActionTransformer,
    config: ExperimentConfig,
    device: torch.device,
    seed: int,
    max_steps: int | None = None,
) -> SimRolloutTrace:
    """Run one closed-loop episode and return its frames, actions, and outcome."""
    if config.model.action_type != "continuous":
        raise SimRolloutUnavailable("sim rollout requires a continuous-action policy")

    model.eval()
    environment, camera_key = _build_environment(config)
    image_size = config.data.image_size
    horizon = max_steps or config.tracking.sim_rollout_max_steps

    # Seeding NumPy before reset makes robosuite's initial object placement (and
    # thus the recorded video) reproducible across epochs.
    np.random.seed(seed)
    try:
        observation = environment.reset()
        raw_frames = [_camera_frame(observation, camera_key)]
        observations: list[torch.Tensor] = []
        actions: list[np.ndarray] = []
        rewards: list[float] = []
        success = False

        for _ in range(horizon):
            observation_tensor = frames_to_observation(
                np.asarray(raw_frames[-1])[None], image_size
            )[0].to(device)
            observations.append(observation_tensor)
            history = torch.stack(observations)
            action_history = (
                torch.from_numpy(np.stack(actions)).float().to(device)
                if actions
                else torch.zeros(0, config.model.action_dimension, device=device)
            )
            action = model.choose_action_continuous(history, action_history)
            action_np = action.detach().cpu().numpy().astype(np.float32)
            observation, reward, done, _ = environment.step(action_np)
            actions.append(action_np)
            rewards.append(float(reward))
            raw_frames.append(_camera_frame(observation, camera_key))
            if bool(environment._check_success()):
                success = True
                break
            if done:
                break
    finally:
        environment.close()

    frames = frames_to_observation(np.stack(raw_frames), image_size).numpy()
    return SimRolloutTrace(
        frames=frames,
        actions=np.stack(actions) if actions else np.zeros((0, config.model.action_dimension)),
        rewards=np.asarray(rewards, dtype=np.float32),
        success=success,
        total_return=float(sum(rewards)),
    )


@torch.no_grad()
def evaluate_sim_rollouts(
    model: VisionActionTransformer,
    config: ExperimentConfig,
    device: torch.device,
    episodes: int,
    seed: int,
) -> dict[str, float]:
    """Run several closed-loop episodes and summarise task success and return."""
    successes = 0
    returns: list[float] = []
    steps: list[int] = []
    for episode in range(episodes):
        trace = record_sim_rollout(model, config, device, seed=seed + episode)
        successes += int(trace.success)
        returns.append(trace.total_return)
        steps.append(trace.steps)
    return {
        "rollout_success_rate": successes / max(episodes, 1),
        "rollout_mean_return": float(np.mean(returns)) if returns else 0.0,
        "rollout_mean_steps": float(np.mean(steps)) if steps else 0.0,
    }
