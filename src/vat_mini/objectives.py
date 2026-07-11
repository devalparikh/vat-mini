"""Stage-specific learning objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class ObjectiveResult:
    loss: torch.Tensor
    token_accuracy: float
    mean_weight: float


class BehaviorCloningObjective:
    """Supervised next-action prediction on expert demonstrations."""

    def __call__(
        self,
        logits: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        valid_steps: torch.Tensor,
    ) -> ObjectiveResult:
        del rewards
        token_losses = F.cross_entropy(logits.flatten(0, 1), actions.flatten(), reduction="none")
        valid = valid_steps.flatten().float()
        loss = (token_losses * valid).sum() / valid.sum().clamp_min(1.0)
        accuracy = (
            ((logits.argmax(dim=-1) == actions) & valid_steps).sum() / valid_steps.sum().clamp_min(1)
        ).item()
        return ObjectiveResult(loss, accuracy, 1.0)


class ContinuousBehaviorCloningObjective:
    """Masked regression for continuous robot-control vectors."""

    def __call__(
        self,
        predictions: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        valid_steps: torch.Tensor,
    ) -> ObjectiveResult:
        del rewards
        per_step_mse = F.mse_loss(predictions, actions, reduction="none").mean(dim=-1)
        valid = valid_steps.float()
        loss = (per_step_mse * valid).sum() / valid.sum().clamp_min(1.0)
        mae = (
            (predictions.sub(actions).abs().mean(dim=-1) * valid).sum()
            / valid.sum().clamp_min(1.0)
        ).item()
        return ObjectiveResult(loss, mae, 1.0)


class AdvantageWeightedImitationObjective:
    """Reward-to-go weighted imitation, a stable discrete-action post-training step."""

    def __init__(self, temperature: float, maximum_weight: float):
        if temperature <= 0:
            raise ValueError("advantage temperature must be positive")
        self.temperature = temperature
        self.maximum_weight = maximum_weight

    def __call__(
        self,
        logits: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        valid_steps: torch.Tensor,
    ) -> ObjectiveResult:
        returns_to_go = torch.flip(torch.cumsum(torch.flip(rewards, dims=(1,)), dim=1), dims=(1,))
        valid_returns = returns_to_go[valid_steps]
        advantages = (returns_to_go - valid_returns.mean()) / (
            valid_returns.std(unbiased=False) + 1e-6
        )
        weights = torch.exp(advantages / self.temperature).clamp(max=self.maximum_weight).detach()
        weights = weights * valid_steps.float()
        token_losses = F.cross_entropy(
            logits.flatten(0, 1), actions.flatten(), reduction="none"
        ).reshape_as(actions)
        loss = (token_losses * weights).sum() / weights.sum().clamp_min(1.0)
        accuracy = (
            ((logits.argmax(dim=-1) == actions) & valid_steps).sum() / valid_steps.sum().clamp_min(1)
        ).item()
        return ObjectiveResult(loss, accuracy, weights.sum().div(valid_steps.sum().clamp_min(1)).item())


class ContinuousAdvantageWeightedImitationObjective(AdvantageWeightedImitationObjective):
    """Reward-to-go weighting applied to continuous-action regression."""

    def __call__(self, predictions, actions, rewards, valid_steps) -> ObjectiveResult:
        returns_to_go = torch.flip(torch.cumsum(torch.flip(rewards, dims=(1,)), dim=1), dims=(1,))
        valid_returns = returns_to_go[valid_steps]
        advantages = (returns_to_go - valid_returns.mean()) / (
            valid_returns.std(unbiased=False) + 1e-6
        )
        weights = torch.exp(advantages / self.temperature).clamp(max=self.maximum_weight).detach()
        weights = weights * valid_steps.float()
        per_step_mse = F.mse_loss(predictions, actions, reduction="none").mean(dim=-1)
        loss = (per_step_mse * weights).sum() / weights.sum().clamp_min(1.0)
        mae = (
            predictions.sub(actions).abs().mean(dim=-1).mul(valid_steps).sum()
            / valid_steps.sum().clamp_min(1)
        ).item()
        return ObjectiveResult(loss, mae, weights.sum().div(valid_steps.sum().clamp_min(1)).item())
