"""A deterministic visual GridWorld and shortest-path demonstration dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F

from vat_mini.config import DataConfig

STAY, UP, DOWN, LEFT, RIGHT = range(5)
ACTION_NAMES = ("stay", "up", "down", "left", "right")


def frames_to_observation(frames: np.ndarray, image_size: int) -> torch.Tensor:
    """Convert raw camera frames to the model's observation tensor.

    Accepts a stack of ``[T, H, W, 3]`` (channel-last) or ``[T, 3, H, W]`` frames
    in the ``[0, 255]`` range and returns ``[T, 3, image_size, image_size]`` floats
    in ``[0, 1]``. Used by both the dataset loader and the closed-loop sim rollout
    so the policy sees pixels preprocessed identically in training and evaluation.
    """
    frames = np.asarray(frames)
    if frames.ndim != 4:
        raise ValueError(f"camera observations must have rank 4, received {frames.shape}")
    if frames.shape[-1] == 3:
        frames = np.moveaxis(frames, -1, 1)
    observations = torch.from_numpy(np.ascontiguousarray(frames)).float().div_(255.0)
    if observations.shape[-2:] != (image_size, image_size):
        observations = F.interpolate(
            observations, size=(image_size, image_size), mode="bilinear",
            align_corners=False,
        )
    return observations


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


class RobomimicSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Lazy fixed-length windows over RoboMimic HDF5 demonstrations.

    HDF5 files remain on disk and only the requested camera frames are decoded
    for each batch. This is important for image training on memory-constrained
    local machines.
    """

    def __init__(
        self,
        path: str | Path,
        demonstration_keys: list[str],
        sequence_length: int,
        image_size: int,
        camera_key: str,
        frame_stride: int,
        maximum_samples: int,
        seed: int,
    ):
        self.path = str(path)
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.camera_key = camera_key
        self.frame_stride = frame_stride
        self._archive = None
        h5py = _require_h5py()
        candidates: list[tuple[str, int]] = []
        with h5py.File(self.path, "r") as archive:
            for demonstration_key in demonstration_keys:
                episode = archive[f"data/{demonstration_key}"]
                length = int(episode["actions"].shape[0])
                required_span = (sequence_length - 1) * frame_stride + 1
                for start in range(max(length - required_span + 1, 0)):
                    candidates.append((demonstration_key, start))
        rng = np.random.default_rng(seed)
        if len(candidates) > maximum_samples:
            selected = rng.choice(len(candidates), size=maximum_samples, replace=False)
            self.windows = [candidates[int(index)] for index in selected]
        else:
            self.windows = candidates

    def __len__(self) -> int:
        return len(self.windows)

    def _file(self):
        if self._archive is None:
            self._archive = _require_h5py().File(self.path, "r")
        return self._archive

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_archive"] = None
        return state

    def __del__(self):
        archive = getattr(self, "_archive", None)
        if archive is not None:
            archive.close()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        demonstration_key, start = self.windows[index]
        episode = self._file()[f"data/{demonstration_key}"]
        indices = start + np.arange(self.sequence_length) * self.frame_stride
        frames = np.asarray(episode[f"obs/{self.camera_key}"][indices])
        observations = frames_to_observation(frames, self.image_size)
        actions = torch.from_numpy(np.asarray(episode["actions"][indices]).copy()).float()
        if "rewards" in episode:
            rewards = torch.from_numpy(np.asarray(episode["rewards"][indices]).copy()).float()
        else:
            rewards = torch.zeros(self.sequence_length, dtype=torch.float32)
        return {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "valid_steps": torch.ones(self.sequence_length, dtype=torch.bool),
        }


def _require_h5py():
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError(
            "RoboMimic HDF5 loading requires the robotics extra: pip install -e '.[robotics]'"
        ) from error
    return h5py


def _robomimic_demonstration_split(config: DataConfig, seed: int) -> tuple[list[str], list[str]]:
    if not config.dataset_path or not Path(config.dataset_path).exists():
        raise FileNotFoundError(f"RoboMimic dataset does not exist: {config.dataset_path}")
    h5py = _require_h5py()
    with h5py.File(config.dataset_path, "r") as archive:
        if "data" not in archive:
            raise ValueError("RoboMimic HDF5 file must contain a 'data' group")
        keys = sorted(archive["data"].keys())
        if not keys:
            raise ValueError("RoboMimic HDF5 file contains no demonstrations")
        first = archive[f"data/{keys[0]}"]
        if f"obs/{config.camera_key}" not in first:
            raise ValueError(f"camera key not found in dataset: {config.camera_key}")
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    validation_count = max(1, round(len(keys) * config.validation_fraction))
    if validation_count >= len(keys):
        raise ValueError("RoboMimic dataset needs at least two demonstrations for a split")
    return keys[validation_count:], keys[:validation_count]


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


def build_datasets(config: DataConfig, seed: int) -> tuple[Dataset, Dataset]:
    if config.dataset_type == "robomimic_hdf5":
        train_keys, validation_keys = _robomimic_demonstration_split(config, seed)
        common = dict(
            path=config.dataset_path,
            sequence_length=config.sequence_length,
            image_size=config.image_size,
            camera_key=config.camera_key,
            frame_stride=config.frame_stride,
        )
        return (
            RobomimicSequenceDataset(
                demonstration_keys=train_keys, maximum_samples=config.train_samples,
                seed=seed, **common,
            ),
            RobomimicSequenceDataset(
                demonstration_keys=validation_keys, maximum_samples=config.validation_samples,
                seed=seed + 1, **common,
            ),
        )
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
