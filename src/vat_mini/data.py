"""A deterministic visual GridWorld and shortest-path demonstration dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from vat_mini.config import DataConfig

STAY, UP, DOWN, LEFT, RIGHT = range(5)
ACTION_NAMES = ("stay", "up", "down", "left", "right")


@dataclass
class GridState:
    agent_row: int
    agent_column: int
    target_row: int
    target_column: int


class VisualGridWorld:
    """Tiny embodied-control environment rendered as an RGB observation."""

    def __init__(self, grid_size: int, image_size: int, rng: np.random.Generator):
        self.grid_size = grid_size
        self.image_size = image_size
        self.rng = rng
        self.state = GridState(0, 0, 0, 1)

    def reset(self) -> np.ndarray:
        cells = self.rng.choice(self.grid_size * self.grid_size, size=2, replace=False)
        agent, target = (int(cell) for cell in cells)
        self.state = GridState(
            agent // self.grid_size,
            agent % self.grid_size,
            target // self.grid_size,
            target % self.grid_size,
        )
        return self.render()

    @property
    def reached_target(self) -> bool:
        return (self.state.agent_row, self.state.agent_column) == (
            self.state.target_row,
            self.state.target_column,
        )

    @property
    def shortest_distance(self) -> int:
        return abs(self.state.target_row - self.state.agent_row) + abs(
            self.state.target_column - self.state.agent_column
        )

    def expert_action(self) -> int:
        row_delta = self.state.target_row - self.state.agent_row
        column_delta = self.state.target_column - self.state.agent_column
        if row_delta == column_delta == 0:
            return STAY
        # Alternating axes on ties adds variety without sacrificing optimality.
        if abs(row_delta) > abs(column_delta) or (
            abs(row_delta) == abs(column_delta) and self.rng.random() < 0.5
        ):
            return DOWN if row_delta > 0 else UP
        return RIGHT if column_delta > 0 else LEFT

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        row_change = {UP: -1, DOWN: 1}.get(int(action), 0)
        column_change = {LEFT: -1, RIGHT: 1}.get(int(action), 0)
        old_distance = self.shortest_distance
        self.state.agent_row = int(np.clip(self.state.agent_row + row_change, 0, self.grid_size - 1))
        self.state.agent_column = int(
            np.clip(self.state.agent_column + column_change, 0, self.grid_size - 1)
        )
        done = self.reached_target
        progress = old_distance - self.shortest_distance
        reward = 1.0 if done else 0.05 * progress - 0.01
        return self.render(), reward, done

    def render(self) -> np.ndarray:
        image = np.full((3, self.image_size, self.image_size), 0.08, dtype=np.float32)
        boundaries = np.linspace(0, self.image_size, self.grid_size + 1, dtype=int)
        image[:, boundaries[:-1], :] = 0.16
        image[:, :, boundaries[:-1]] = 0.16
        self._paint_cell(image, self.state.target_row, self.state.target_column, (0.9, 0.2, 0.2), boundaries)
        self._paint_cell(image, self.state.agent_row, self.state.agent_column, (0.2, 0.45, 1.0), boundaries)
        return image

    @staticmethod
    def _paint_cell(
        image: np.ndarray,
        row: int,
        column: int,
        color: tuple[float, float, float],
        boundaries: np.ndarray,
    ) -> None:
        top, bottom = boundaries[row], boundaries[row + 1]
        left, right = boundaries[column], boundaries[column + 1]
        image[:, top + 1 : bottom, left + 1 : right] = np.asarray(color)[:, None, None]


class GridWorldSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Fixed expert trajectories, generated once for deterministic experiments."""

    def __init__(
        self,
        sample_count: int,
        sequence_length: int,
        grid_size: int,
        image_size: int,
        seed: int,
        action_noise: float = 0.0,
        arrays: dict[str, np.ndarray] | None = None,
    ):
        if arrays is None:
            arrays = generate_demonstrations(
                sample_count, sequence_length, grid_size, image_size, seed, action_noise
            )
        self.observations = torch.from_numpy(arrays["observations"]).float()
        self.actions = torch.from_numpy(arrays["actions"]).long()
        self.rewards = torch.from_numpy(arrays["rewards"]).float()
        self.valid_steps = torch.from_numpy(arrays["valid_steps"]).bool()

    def __len__(self) -> int:
        return self.actions.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "observations": self.observations[index],
            "actions": self.actions[index],
            "rewards": self.rewards[index],
            "valid_steps": self.valid_steps[index],
        }


def generate_demonstrations(
    sample_count: int,
    sequence_length: int,
    grid_size: int,
    image_size: int,
    seed: int,
    action_noise: float = 0.0,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    observations = np.empty((sample_count, sequence_length, 3, image_size, image_size), dtype=np.float32)
    actions = np.empty((sample_count, sequence_length), dtype=np.int64)
    rewards = np.empty((sample_count, sequence_length), dtype=np.float32)
    valid_steps = np.zeros((sample_count, sequence_length), dtype=np.bool_)
    environment = VisualGridWorld(grid_size, image_size, rng)
    for sample_index in range(sample_count):
        observation = environment.reset()
        for step_index in range(sequence_length):
            observations[sample_index, step_index] = observation
            action = environment.expert_action()
            if rng.random() < action_noise:
                action = int(rng.integers(0, len(ACTION_NAMES)))
            observation, reward, done = environment.step(action)
            actions[sample_index, step_index] = action
            rewards[sample_index, step_index] = reward
            valid_steps[sample_index, step_index] = True
            if done:
                # The remaining array entries are padding and are excluded by valid_steps.
                observations[sample_index, step_index + 1 :] = 0.0
                actions[sample_index, step_index + 1 :] = STAY
                rewards[sample_index, step_index + 1 :] = 0.0
                break
    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "valid_steps": valid_steps,
    }


def save_dataset(path: str | Path, config: DataConfig, seed: int) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    train = generate_demonstrations(
        config.train_samples,
        config.sequence_length,
        config.grid_size,
        config.image_size,
        seed,
        config.action_noise,
    )
    validation = generate_demonstrations(
        config.validation_samples,
        config.sequence_length,
        config.grid_size,
        config.image_size,
        seed + 1,
        # Validation always measures the clean expert contract, even when the
        # post-training split intentionally contains off-policy actions.
        0.0,
    )
    np.savez_compressed(
        destination,
        **{f"train_{key}": value for key, value in train.items()},
        **{f"validation_{key}": value for key, value in validation.items()},
    )
    return destination


def _load_split(path: str | Path, split: str) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {
            name: archive[f"{split}_{name}"]
            for name in ("observations", "actions", "rewards", "valid_steps")
        }


def build_datasets(config: DataConfig, seed: int) -> tuple[GridWorldSequenceDataset, GridWorldSequenceDataset]:
    path = Path(config.dataset_path) if config.dataset_path else None
    if path and path.exists():
        train_arrays = _load_split(path, "train")
        validation_arrays = _load_split(path, "validation")
        train = GridWorldSequenceDataset(0, 0, 0, 0, seed, arrays=train_arrays)
        validation = GridWorldSequenceDataset(0, 0, 0, 0, seed + 1, arrays=validation_arrays)
    else:
        train = GridWorldSequenceDataset(
            config.train_samples,
            config.sequence_length,
            config.grid_size,
            config.image_size,
            seed,
            config.action_noise,
        )
        validation = GridWorldSequenceDataset(
            config.validation_samples,
            config.sequence_length,
            config.grid_size,
            config.image_size,
            seed + 1,
            0.0,
        )
    return train, validation


def build_dataloaders(config: DataConfig, seed: int) -> tuple[DataLoader, DataLoader]:
    train, validation = build_datasets(config, seed)
    generator = torch.Generator().manual_seed(seed)
    common = dict(batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=False)
    return (
        DataLoader(train, shuffle=True, generator=generator, **common),
        DataLoader(validation, shuffle=False, **common),
    )
