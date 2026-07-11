"""Teacher-forced and closed-loop policy evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from vat_mini.data import VisualGridWorld
from vat_mini.model import VisionActionTransformer


@dataclass(frozen=True)
class RolloutTrace:
    frames: np.ndarray
    actions: tuple[int, ...]
    rewards: tuple[float, ...]
    success: bool
    total_return: float


@dataclass(frozen=True)
class DemonstrationTrace:
    """One teacher-forced episode for continuous-action visual comparison.

    Robomimic has no simulator wired in here, so instead of a closed-loop
    rollout we replay a real demonstration: the recorded camera frames with the
    policy's predicted action next to the ground-truth action at each step.
    """

    frames: np.ndarray
    predicted_actions: np.ndarray
    ground_truth_actions: np.ndarray

    @property
    def action_mae(self) -> float:
        return float(np.abs(self.predicted_actions - self.ground_truth_actions).mean())


@torch.no_grad()
def record_demonstration(
    model: VisionActionTransformer,
    batches: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
) -> DemonstrationTrace | None:
    """Teacher-force the first validation episode and record predicted vs. true actions."""
    batch = next(iter(batches), None)
    if batch is None:
        return None
    model.eval()
    observations = batch["observations"][:1].to(device)
    actions = batch["actions"][:1].to(device)
    valid = batch["valid_steps"][0].to(device)
    predictions = model(observations, model.shifted_actions(actions))
    keep = valid.cpu().numpy().astype(bool)
    return DemonstrationTrace(
        frames=observations[0].cpu().numpy()[keep],
        predicted_actions=predictions[0].cpu().numpy()[keep],
        ground_truth_actions=actions[0].cpu().numpy()[keep],
    )


@torch.no_grad()
def evaluate_demonstrations(
    model: VisionActionTransformer,
    batches: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    token_count = 0
    batch_count = 0
    action_counts: torch.Tensor | None = None
    absolute_error = 0.0
    for batch in batches:
        observations = batch["observations"].to(device)
        actions = batch["actions"].to(device)
        valid_steps = batch["valid_steps"].to(device)
        predictions = model(observations, model.shifted_actions(actions))
        valid = valid_steps.flatten()
        if model.config.action_type == "continuous":
            per_step_mse = F.mse_loss(predictions, actions, reduction="none").mean(dim=-1)
            total_loss += float(per_step_mse[valid_steps].mean().item())
            absolute_error += float(
                predictions.sub(actions).abs().mean(dim=-1)[valid_steps].sum().item()
            )
        else:
            token_losses = F.cross_entropy(
                predictions.flatten(0, 1), actions.flatten(), reduction="none"
            )
            total_loss += float(token_losses[valid].mean().item())
            correct += int(((predictions.argmax(dim=-1) == actions) & valid_steps).sum().item())
            counts = torch.bincount(actions[valid_steps], minlength=model.config.num_actions).cpu()
            action_counts = counts if action_counts is None else action_counts + counts
        token_count += int(valid_steps.sum().item())
        batch_count += 1
    if model.config.action_type == "continuous":
        return {
            "validation_action_mse": total_loss / max(batch_count, 1),
            "validation_action_mae": absolute_error / max(token_count, 1),
        }
    return {
        "validation_loss": total_loss / max(batch_count, 1),
        "validation_token_accuracy": correct / max(token_count, 1),
        "validation_majority_class_baseline": (
            int(action_counts.max().item()) / max(token_count, 1) if action_counts is not None else 0.0
        ),
    }


@torch.no_grad()
def evaluate_rollouts(
    model: VisionActionTransformer,
    device: torch.device,
    grid_size: int,
    image_size: int,
    episodes: int = 32,
    seed: int = 10_000,
) -> dict[str, float]:
    """Run the learned policy without expert actions or teacher forcing."""
    model.eval()
    environment = VisualGridWorld(grid_size, image_size, np.random.default_rng(seed))
    successes = 0
    efficiencies: list[float] = []
    returns: list[float] = []
    maximum_steps = max(2 * (grid_size - 1), 1)
    for _ in range(episodes):
        observation = environment.reset()
        optimal_steps = max(environment.shortest_distance, 1)
        observations: list[torch.Tensor] = []
        actions: list[int] = []
        episode_return = 0.0
        for step in range(1, maximum_steps + 1):
            observations.append(torch.from_numpy(observation).to(device))
            observation_history = torch.stack(observations)
            action_history = torch.tensor(actions, dtype=torch.long, device=device)
            action = model.choose_action(observation_history, action_history)
            observation, reward, done = environment.step(action)
            actions.append(action)
            episode_return += reward
            if done:
                successes += 1
                efficiencies.append(optimal_steps / step)
                break
        else:
            efficiencies.append(0.0)
        returns.append(episode_return)
    return {
        "rollout_success_rate": successes / max(episodes, 1),
        "rollout_path_efficiency": float(np.mean(efficiencies)),
        "rollout_mean_return": float(np.mean(returns)),
    }


@torch.no_grad()
def record_rollout(
    model: VisionActionTransformer,
    device: torch.device,
    grid_size: int,
    image_size: int,
    seed: int,
) -> RolloutTrace:
    """Record one fixed-seed closed-loop episode for visual comparison across epochs."""
    model.eval()
    environment = VisualGridWorld(grid_size, image_size, np.random.default_rng(seed))
    observation = environment.reset()
    frames = [observation.copy()]
    observations: list[torch.Tensor] = []
    actions: list[int] = []
    rewards: list[float] = []
    maximum_steps = max(2 * (grid_size - 1), 1)
    success = False
    for _ in range(maximum_steps):
        observations.append(torch.from_numpy(observation).to(device))
        action_history = torch.tensor(actions, dtype=torch.long, device=device)
        action = model.choose_action(torch.stack(observations), action_history)
        observation, reward, success = environment.step(action)
        actions.append(action)
        rewards.append(reward)
        frames.append(observation.copy())
        if success:
            break
    return RolloutTrace(
        frames=np.stack(frames),
        actions=tuple(actions),
        rewards=tuple(rewards),
        success=success,
        total_return=float(sum(rewards)),
    )
