"""Optional experiment tracking with a no-op local default."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from vat_mini.config import ExperimentConfig
from vat_mini.data import ACTION_NAMES
from vat_mini.evaluation import RolloutTrace


class ExperimentTracker(Protocol):
    @property
    def enabled(self) -> bool: ...

    def log_metrics(self, metrics: dict[str, float | int], step: int) -> None: ...

    def log_rollout(self, trace: RolloutTrace, epoch: int, step: int) -> None: ...

    def log_checkpoint(self, path: str | Path, stage: str) -> None: ...

    def finish(self, exit_code: int = 0) -> None: ...


class DisabledTracker:
    @property
    def enabled(self) -> bool:
        return False

    def log_metrics(self, metrics: dict[str, float | int], step: int) -> None:
        del metrics, step

    def log_rollout(self, trace: RolloutTrace, epoch: int, step: int) -> None:
        del trace, epoch, step

    def log_checkpoint(self, path: str | Path, stage: str) -> None:
        del path, stage

    def finish(self, exit_code: int = 0) -> None:
        del exit_code


class WandbTracker:
    def __init__(self, config: ExperimentConfig):
        try:
            import wandb
            from PIL import Image
        except ImportError as error:
            raise RuntimeError(
                "W&B tracking is enabled but the optional dependency is missing; "
                "run `make setup-tracking` or install `vat-mini[tracking]`"
            ) from error

        self._wandb = wandb
        self._image = Image
        self._media_dir = Path(config.output_dir) / "media"
        self._media_dir.mkdir(parents=True, exist_ok=True)
        self._run = wandb.init(
            project=config.tracking.project,
            entity=config.tracking.entity,
            name=config.tracking.run_name,
            mode=config.tracking.mode,
            config=config.to_dict(),
            dir=config.output_dir,
            tags=[config.training.stage, str(config.device)],
        )
        self._run.define_metric("trainer/global_step")
        self._run.define_metric("*", step_metric="trainer/global_step")

    @property
    def enabled(self) -> bool:
        return True

    def log_metrics(self, metrics: dict[str, float | int], step: int) -> None:
        self._run.log({"trainer/global_step": step, **metrics})

    def log_rollout(self, trace: RolloutTrace, epoch: int, step: int) -> None:
        frames = np.clip(trace.frames * 255.0, 0, 255).astype(np.uint8)
        frames = frames.transpose(0, 2, 3, 1)
        # Make the small environment legible without changing its pixels.
        scale = max(1, 256 // max(frames.shape[1:3]))
        frames = np.repeat(np.repeat(frames, scale, axis=1), scale, axis=2)
        actions = ", ".join(ACTION_NAMES[action] for action in trace.actions)
        caption = (
            f"epoch {epoch} | {'success' if trace.success else 'timeout'} | "
            f"return {trace.total_return:.3f} | {actions}"
        )
        table = self._wandb.Table(
            columns=["epoch", "step", "action", "reward"],
            data=[
                [epoch, index + 1, ACTION_NAMES[action], reward]
                for index, (action, reward) in enumerate(zip(trace.actions, trace.rewards))
            ],
        )
        gif_path = self._media_dir / f"rollout-epoch-{epoch:03d}.gif"
        images = [self._image.fromarray(frame, mode="RGB") for frame in frames]
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=500,
            loop=0,
            optimize=False,
        )
        self._run.log(
            {
                "trainer/global_step": step,
                "rollout/video": self._wandb.Video(str(gif_path), format="gif", caption=caption),
                "rollout/actions": table,
            }
        )

    def log_checkpoint(self, path: str | Path, stage: str) -> None:
        checkpoint = Path(path)
        artifact = self._wandb.Artifact(
            name=f"vat-mini-{stage}-checkpoint",
            type="model",
            metadata={"stage": stage},
        )
        artifact.add_file(str(checkpoint), name="latest.pt")
        for sibling in (checkpoint.parent / "config.json", checkpoint.parent / "metrics.jsonl"):
            if sibling.exists():
                artifact.add_file(str(sibling), name=sibling.name)
        self._run.log_artifact(artifact, aliases=["latest", stage])

    def finish(self, exit_code: int = 0) -> None:
        self._run.finish(exit_code=exit_code)


def build_tracker(config: ExperimentConfig) -> ExperimentTracker:
    if not config.tracking.enabled:
        return DisabledTracker()
    return WandbTracker(config)
