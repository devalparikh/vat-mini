"""Phase-aware, portable checkpoint IO."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import torch

from vat_mini.config import ExperimentConfig


class CheckpointManager:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_config_snapshot(self, config: ExperimentConfig) -> None:
        destination = self.output_dir / "config.json"
        destination.write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: ExperimentConfig,
        epoch: int,
        global_step: int,
        metrics: dict[str, float],
    ) -> Path:
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.to_dict(),
            "stage": config.training.stage,
            "epoch": epoch,
            "global_step": global_step,
            "metrics": metrics,
        }
        epoch_path = self.output_dir / f"{config.training.stage}-epoch-{epoch:03d}.pt"
        temporary_path = epoch_path.with_suffix(".tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(epoch_path)
        latest_path = self.output_dir / "latest.pt"
        latest_temporary = latest_path.with_suffix(".tmp")
        torch.save(payload, latest_temporary)
        latest_temporary.replace(latest_path)
        return epoch_path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    # Restrict deserialization to tensors and primitive containers so a checkpoint
    # cannot execute arbitrary Python code while it is being opened.
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        raise ValueError(
            "checkpoint contains unsafe or unsupported serialized objects; "
            "only weights-only checkpoints are accepted"
        ) from error

    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    model_state = payload.get("model")
    if not isinstance(model_state, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in model_state.items()
    ):
        raise ValueError("checkpoint model state must map parameter names to tensors")

    model.load_state_dict(model_state)
    if optimizer is not None and "optimizer" in payload:
        optimizer_state = payload["optimizer"]
        if not isinstance(optimizer_state, dict):
            raise ValueError("checkpoint optimizer state must be a mapping")
        optimizer.load_state_dict(optimizer_state)
    return payload
