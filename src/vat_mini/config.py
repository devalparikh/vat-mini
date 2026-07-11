"""Typed experiment configuration and small YAML/CLI override loader."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass
class DataConfig:
    dataset_type: str = "gridworld"
    dataset_path: str | None = None
    train_samples: int = 256
    validation_samples: int = 64
    sequence_length: int = 12
    image_size: int = 32
    grid_size: int = 8
    batch_size: int = 16
    num_workers: int = 0
    action_noise: float = 0.0
    camera_key: str = "agentview_image"
    validation_fraction: float = 0.1
    frame_stride: int = 1


@dataclass
class ModelConfig:
    action_type: str = "discrete"
    num_actions: int = 5
    action_dimension: int = 7
    vision_width: int = 32
    embedding_dim: int = 96
    transformer_layers: int = 2
    attention_heads: int = 4
    feedforward_dim: int = 192
    dropout: float = 0.1
    max_sequence_length: int = 32


@dataclass
class TrainingConfig:
    stage: str = "pretrain"
    epochs: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    gradient_clip_norm: float = 1.0
    log_every_steps: int = 10
    checkpoint_every_epochs: int = 1
    advantage_temperature: float = 0.5
    maximum_advantage_weight: float = 20.0
    initial_checkpoint: str | None = None


@dataclass
class TrackingConfig:
    enabled: bool = False
    project: str = "vat-mini"
    entity: str | None = None
    mode: str = "online"
    run_name: str | None = None
    rollout_every_epochs: int = 1
    # When enabled (RoboMimic only), the periodic rollout drives the policy in a
    # real simulator instead of replaying a demonstration. Falls back to the
    # teacher-forced replay if the sim stack is unavailable on this machine.
    sim_rollout: bool = False
    sim_rollout_episodes: int = 5
    sim_rollout_max_steps: int = 400


@dataclass
class ExperimentConfig:
    seed: int = 7
    output_dir: str = "runs/tiny"
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def validate(self) -> None:
        if self.data.train_samples <= 0 or self.data.validation_samples <= 0:
            raise ValueError("data sample counts must be positive")
        if self.data.sequence_length <= 0 or self.data.batch_size <= 0:
            raise ValueError("data.sequence_length and data.batch_size must be positive")
        if self.data.num_workers < 0:
            raise ValueError("data.num_workers cannot be negative")
        if self.model.attention_heads <= 0 or self.model.embedding_dim <= 0:
            raise ValueError("model attention_heads and embedding_dim must be positive")
        if self.model.embedding_dim % self.model.attention_heads:
            raise ValueError("model.embedding_dim must be divisible by attention_heads")
        if self.data.sequence_length > self.model.max_sequence_length:
            raise ValueError("data.sequence_length exceeds model.max_sequence_length")
        if self.training.stage not in {"pretrain", "posttrain"}:
            raise ValueError("training.stage must be 'pretrain' or 'posttrain'")
        if self.data.dataset_type not in {"gridworld", "robomimic_hdf5"}:
            raise ValueError("data.dataset_type must be 'gridworld' or 'robomimic_hdf5'")
        if self.model.action_type not in {"discrete", "continuous"}:
            raise ValueError("model.action_type must be 'discrete' or 'continuous'")
        if self.data.dataset_type == "gridworld" and self.model.action_type != "discrete":
            raise ValueError("the gridworld dataset requires model.action_type='discrete'")
        if self.data.dataset_type == "gridworld" and self.model.num_actions != 5:
            raise ValueError("the grid-world dataset currently defines exactly 5 actions")
        if self.data.dataset_type == "robomimic_hdf5":
            if self.model.action_type != "continuous":
                raise ValueError("RoboMimic requires model.action_type='continuous'")
            if not self.data.dataset_path:
                raise ValueError("RoboMimic requires data.dataset_path")
        if self.model.action_dimension <= 0:
            raise ValueError("model.action_dimension must be positive")
        if self.data.grid_size < 2 or self.data.image_size < 1:
            raise ValueError("grid_size must be >= 2 and image_size must be positive")
        if not 0.0 <= self.data.action_noise <= 1.0:
            raise ValueError("data.action_noise must be between 0 and 1")
        if not 0.0 < self.data.validation_fraction < 1.0:
            raise ValueError("data.validation_fraction must be between 0 and 1")
        if self.data.frame_stride <= 0:
            raise ValueError("data.frame_stride must be positive")
        if self.training.epochs <= 0:
            raise ValueError("training.epochs must be positive")
        if self.training.log_every_steps <= 0 or self.training.checkpoint_every_epochs <= 0:
            raise ValueError("training log and checkpoint intervals must be positive")
        if self.training.learning_rate <= 0 or self.training.gradient_clip_norm <= 0:
            raise ValueError("training learning_rate and gradient_clip_norm must be positive")
        if self.training.advantage_temperature <= 0:
            raise ValueError("training.advantage_temperature must be positive")
        if self.training.maximum_advantage_weight <= 0:
            raise ValueError("training.maximum_advantage_weight must be positive")
        if not self.tracking.project.strip():
            raise ValueError("tracking.project cannot be empty")
        if self.tracking.mode not in {"online", "offline"}:
            raise ValueError("tracking.mode must be 'online' or 'offline'")
        if self.tracking.rollout_every_epochs <= 0:
            raise ValueError("tracking.rollout_every_epochs must be positive")
        if self.tracking.sim_rollout_episodes <= 0:
            raise ValueError("tracking.sim_rollout_episodes must be positive")
        if self.tracking.sim_rollout_max_steps <= 0:
            raise ValueError("tracking.sim_rollout_max_steps must be positive")
        if self.tracking.sim_rollout and self.data.dataset_type != "robomimic_hdf5":
            raise ValueError("tracking.sim_rollout is only supported for the robomimic_hdf5 dataset")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ConfigType = TypeVar("ConfigType")


def _update_dataclass(instance: ConfigType, values: dict[str, Any]) -> ConfigType:
    known_fields = {item.name for item in fields(instance)}  # type: ignore[arg-type]
    unknown = set(values) - known_fields
    if unknown:
        raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
    for name, value in values.items():
        current = getattr(instance, name)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"configuration field {name!r} must be a mapping")
            _update_dataclass(current, value)
        else:
            setattr(instance, name, value)
    return instance


def _parse_override(raw_value: str) -> Any:
    return yaml.safe_load(raw_value)


def _apply_override(config: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"override must use key=value syntax: {override!r}")
    dotted_key, raw_value = override.split("=", 1)
    keys = dotted_key.split(".")
    destination = config
    for key in keys[:-1]:
        child = destination.setdefault(key, {})
        if not isinstance(child, dict):
            raise ValueError(f"cannot set nested value under {key!r}")
        destination = child
    destination[keys[-1]] = _parse_override(raw_value)


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as config_file:
        values = yaml.safe_load(config_file) or {}
    if not isinstance(values, dict):
        raise ValueError("configuration root must be a mapping")
    for override in overrides or []:
        _apply_override(values, override)
    config = _update_dataclass(ExperimentConfig(), values)
    config.validate()
    return config
