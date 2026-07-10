from pathlib import Path

import pytest

from vat_mini.config import load_config


def test_load_config_and_override() -> None:
    path = Path(__file__).parents[1] / "configs" / "smoke.yaml"
    config = load_config(path, ["training.epochs=2", "device=cpu"])
    assert config.training.epochs == 2
    assert config.device == "cpu"
    assert config.tracking.enabled is False


def test_unknown_config_field_fails_early() -> None:
    path = Path(__file__).parents[1] / "configs" / "smoke.yaml"
    with pytest.raises(ValueError, match="unknown configuration"):
        load_config(path, ["model.mystery_width=10"])


@pytest.mark.parametrize(
    "override",
    [
        "data.batch_size=0",
        "model.attention_heads=0",
        "training.epochs=0",
        "training.log_every_steps=0",
        "tracking.rollout_every_epochs=0",
    ],
)
def test_invalid_numeric_config_fails_early(override: str) -> None:
    path = Path(__file__).parents[1] / "configs" / "smoke.yaml"
    with pytest.raises(ValueError, match="positive"):
        load_config(path, [override])


def test_tracking_config_overrides() -> None:
    path = Path(__file__).parents[1] / "configs" / "smoke.yaml"
    config = load_config(
        path,
        [
            "tracking.enabled=true",
            "tracking.mode=offline",
            "tracking.run_name=test-run",
            "tracking.rollout_every_epochs=2",
        ],
    )
    assert config.tracking.enabled is True
    assert config.tracking.mode == "offline"
    assert config.tracking.run_name == "test-run"
    assert config.tracking.rollout_every_epochs == 2


def test_unknown_tracking_mode_fails_early() -> None:
    path = Path(__file__).parents[1] / "configs" / "smoke.yaml"
    with pytest.raises(ValueError, match="tracking.mode"):
        load_config(path, ["tracking.mode=maybe"])
