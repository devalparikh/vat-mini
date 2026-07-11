from pathlib import Path

import torch

from vat_mini.checkpoint import load_checkpoint
from vat_mini.config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from vat_mini.data import build_dataloaders
from vat_mini.device import seed_everything
from vat_mini.model import VisionActionTransformer
from vat_mini.trainer import Trainer


class RecordingTracker:
    enabled = True

    def __init__(self) -> None:
        self.metric_records: list[tuple[dict[str, float | int], int]] = []
        self.rollout_epochs: list[int] = []
        self.checkpoints: list[Path] = []
        self.exit_codes: list[int] = []

    def log_metrics(self, metrics: dict[str, float | int], step: int) -> None:
        self.metric_records.append((metrics, step))

    def log_rollout(self, trace, epoch: int, step: int) -> None:
        assert trace.frames.ndim == 4
        self.rollout_epochs.append(epoch)

    def log_checkpoint(self, path: str | Path, stage: str) -> None:
        assert stage == "pretrain"
        self.checkpoints.append(Path(path))

    def finish(self, exit_code: int = 0) -> None:
        self.exit_codes.append(exit_code)


def test_one_epoch_writes_portable_checkpoint(tmp_path: Path) -> None:
    config = ExperimentConfig(
        output_dir=str(tmp_path),
        device="cpu",
        data=DataConfig(
            train_samples=8,
            validation_samples=4,
            sequence_length=3,
            image_size=16,
            grid_size=4,
            batch_size=4,
        ),
        model=ModelConfig(
            vision_width=4,
            embedding_dim=16,
            transformer_layers=1,
            attention_heads=2,
            feedforward_dim=32,
            dropout=0.0,
            max_sequence_length=4,
        ),
        training=TrainingConfig(epochs=1, log_every_steps=100),
    )
    seed_everything(config.seed)
    train_loader, validation_loader = build_dataloaders(config.data, config.seed)
    model = VisionActionTransformer(config.model)
    tracker = RecordingTracker()
    Trainer(config, model, torch.device("cpu"), tracker=tracker).fit(
        train_loader, validation_loader
    )
    checkpoint = tmp_path / "latest.pt"
    assert checkpoint.exists()
    fresh_model = VisionActionTransformer(config.model)
    payload = load_checkpoint(checkpoint, fresh_model)
    assert payload["stage"] == "pretrain"
    assert (tmp_path / "config.json").exists()
    batch_records = [metrics for metrics, _ in tracker.metric_records if "train/batch_loss" in metrics]
    assert len(batch_records) == 2
    assert tracker.rollout_epochs == [1]
    assert tracker.checkpoints == [checkpoint]
    assert tracker.exit_codes == [0]


def test_continuous_robomimic_training_path(tmp_path: Path) -> None:
    h5py = __import__("pytest").importorskip("h5py")
    dataset_path = tmp_path / "can.hdf5"
    with h5py.File(dataset_path, "w") as archive:
        data = archive.create_group("data")
        for episode_index in range(4):
            episode = data.create_group(f"demo_{episode_index}")
            observations = episode.create_group("obs")
            observations.create_dataset(
                "agentview_image",
                data=torch.randint(0, 256, (6, 16, 16, 3), dtype=torch.uint8).numpy(),
            )
            episode.create_dataset(
                "actions", data=torch.rand(6, 7).mul(2).sub(1).numpy()
            )
            episode.create_dataset("rewards", data=torch.zeros(6).numpy())
    config = ExperimentConfig(
        output_dir=str(tmp_path / "run"),
        device="cpu",
        data=DataConfig(
            dataset_type="robomimic_hdf5",
            dataset_path=str(dataset_path),
            train_samples=4,
            validation_samples=2,
            sequence_length=3,
            image_size=16,
            batch_size=2,
        ),
        model=ModelConfig(
            action_type="continuous",
            action_dimension=7,
            vision_width=4,
            embedding_dim=16,
            transformer_layers=1,
            attention_heads=2,
            feedforward_dim=32,
            dropout=0.0,
            max_sequence_length=4,
        ),
        training=TrainingConfig(epochs=1, log_every_steps=100),
    )
    config.validate()
    train_loader, validation_loader = build_dataloaders(config.data, config.seed)
    metrics = Trainer(
        config, VisionActionTransformer(config.model), torch.device("cpu")
    ).fit(train_loader, validation_loader)
    assert metrics["validation_action_mse"] >= 0.0
    assert metrics["validation_action_mae"] >= 0.0
    assert "rollout_success_rate" not in metrics
