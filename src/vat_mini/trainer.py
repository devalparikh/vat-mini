"""A small trainer with explicit, inspectable control flow."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import torch

from vat_mini.checkpoint import CheckpointManager, load_checkpoint
from vat_mini.config import ExperimentConfig
from vat_mini.evaluation import evaluate_demonstrations, evaluate_rollouts, record_rollout
from vat_mini.model import VisionActionTransformer
from vat_mini.objectives import AdvantageWeightedImitationObjective, BehaviorCloningObjective
from vat_mini.tracking import ExperimentTracker, build_tracker


class Trainer:
    def __init__(
        self,
        config: ExperimentConfig,
        model: VisionActionTransformer,
        device: torch.device,
        tracker: ExperimentTracker | None = None,
    ):
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.objective = self._build_objective()
        self.checkpoints = CheckpointManager(config.output_dir)
        self.global_step = 0
        self.tracker = tracker or build_tracker(config)

    def _build_objective(self):
        if self.config.training.stage == "pretrain":
            return BehaviorCloningObjective()
        return AdvantageWeightedImitationObjective(
            temperature=self.config.training.advantage_temperature,
            maximum_weight=self.config.training.maximum_advantage_weight,
        )

    def initialize_from_checkpoint(self, path: str | Path) -> None:
        payload = load_checkpoint(path, self.model)
        source_stage = payload.get("stage", "unknown")
        print(f"initialized {self.config.training.stage} model from {path} (stage={source_stage})")

    def fit(self, train_loader: Iterable, validation_loader: Iterable) -> dict[str, float]:
        exit_code = 1
        try:
            self.checkpoints.write_config_snapshot(self.config)
            metrics: dict[str, float] = {}
            for epoch in range(1, self.config.training.epochs + 1):
                train_metrics = self._train_epoch(train_loader, epoch)
                metrics = evaluate_demonstrations(self.model, validation_loader, self.device)
                metrics.update(train_metrics)
                print(json.dumps({"epoch": epoch, **metrics}, sort_keys=True))
                self._append_metrics(epoch, metrics)
                self.tracker.log_metrics(
                    {"epoch": epoch, **self._namespaced_epoch_metrics(metrics)}, self.global_step
                )
                if (
                    self.tracker.enabled
                    and epoch % self.config.tracking.rollout_every_epochs == 0
                ):
                    trace = record_rollout(
                        self.model,
                        self.device,
                        self.config.data.grid_size,
                        self.config.data.image_size,
                        seed=self.config.seed + 2,
                    )
                    self.tracker.log_rollout(trace, epoch, self.global_step)
                if epoch % self.config.training.checkpoint_every_epochs == 0:
                    self.checkpoints.save(
                        self.model, self.optimizer, self.config, epoch, self.global_step, metrics
                    )
            rollout_metrics = evaluate_rollouts(
                self.model,
                self.device,
                self.config.data.grid_size,
                self.config.data.image_size,
                episodes=min(self.config.data.validation_samples, 32),
                seed=self.config.seed + 2,
            )
            metrics.update(rollout_metrics)
            # Refresh the final checkpoint so its metrics include closed-loop behavior.
            self.checkpoints.save(
                self.model,
                self.optimizer,
                self.config,
                self.config.training.epochs,
                self.global_step,
                metrics,
            )
            self._append_metrics(self.config.training.epochs, metrics, record_type="final")
            self.tracker.log_metrics(
                {"epoch": self.config.training.epochs, **self._namespaced_epoch_metrics(metrics)},
                self.global_step,
            )
            self.tracker.log_checkpoint(
                Path(self.config.output_dir) / "latest.pt", self.config.training.stage
            )
            print(json.dumps({"stage": self.config.training.stage, **metrics}, sort_keys=True))
            exit_code = 0
            return metrics
        finally:
            self.tracker.finish(exit_code=exit_code)

    def _train_epoch(self, train_loader: Iterable, epoch: int) -> dict[str, float]:
        self.model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        running_weight = 0.0
        batch_count = 0
        for batch in train_loader:
            observations = batch["observations"].to(self.device)
            actions = batch["actions"].to(self.device)
            rewards = batch["rewards"].to(self.device)
            valid_steps = batch["valid_steps"].to(self.device)
            logits = self.model(observations, self.model.shifted_actions(actions))
            result = self.objective(logits, actions, rewards, valid_steps)

            self.optimizer.zero_grad(set_to_none=True)
            result.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip_norm)
            self.optimizer.step()

            self.global_step += 1
            batch_count += 1
            running_loss += float(result.loss.item())
            running_accuracy += result.token_accuracy
            running_weight += result.mean_weight
            self.tracker.log_metrics(
                {
                    "epoch": epoch,
                    "train/batch_loss": float(result.loss.item()),
                    "train/batch_token_accuracy": result.token_accuracy,
                    "train/batch_mean_advantage_weight": result.mean_weight,
                },
                self.global_step,
            )
            if self.global_step % self.config.training.log_every_steps == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": self.global_step,
                            "loss": float(result.loss.item()),
                            "token_accuracy": result.token_accuracy,
                        }
                    )
                )
        return {
            "train_loss": running_loss / max(batch_count, 1),
            "train_token_accuracy": running_accuracy / max(batch_count, 1),
            "train_mean_advantage_weight": running_weight / max(batch_count, 1),
        }

    @staticmethod
    def _namespaced_epoch_metrics(metrics: dict[str, float]) -> dict[str, float]:
        namespaced: dict[str, float] = {}
        for name, value in metrics.items():
            if name.startswith("train_"):
                namespaced[f"train/epoch_{name.removeprefix('train_')}"] = value
            elif name.startswith("validation_"):
                namespaced[f"validation/{name.removeprefix('validation_')}"] = value
            elif name.startswith("rollout_"):
                namespaced[f"rollout/{name.removeprefix('rollout_')}"] = value
        return namespaced

    def _append_metrics(
        self, epoch: int, metrics: dict[str, float], record_type: str = "epoch"
    ) -> None:
        path = Path(self.config.output_dir) / "metrics.jsonl"
        with path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(
                json.dumps(
                    {
                        "record_type": record_type,
                        "stage": self.config.training.stage,
                        "epoch": epoch,
                        "global_step": self.global_step,
                        **metrics,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
