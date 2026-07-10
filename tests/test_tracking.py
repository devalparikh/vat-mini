import torch

from vat_mini.config import ExperimentConfig, ModelConfig
from vat_mini.evaluation import record_rollout
from vat_mini.model import VisionActionTransformer
from vat_mini.tracking import DisabledTracker, build_tracker


def test_disabled_tracker_does_not_require_optional_dependency() -> None:
    tracker = build_tracker(ExperimentConfig())
    assert isinstance(tracker, DisabledTracker)
    assert tracker.enabled is False


def test_record_rollout_returns_visual_trace() -> None:
    model = VisionActionTransformer(
        ModelConfig(
            vision_width=4,
            embedding_dim=16,
            transformer_layers=1,
            attention_heads=2,
            feedforward_dim=32,
            dropout=0.0,
            max_sequence_length=8,
        )
    )
    trace = record_rollout(model, torch.device("cpu"), grid_size=4, image_size=16, seed=9)
    assert trace.frames.ndim == 4
    assert trace.frames.shape[1:] == (3, 16, 16)
    assert trace.frames.shape[0] == len(trace.actions) + 1
    assert len(trace.actions) == len(trace.rewards)
