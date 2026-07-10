from pathlib import Path

import pytest
import torch

from vat_mini.checkpoint import load_checkpoint


def _write_marker(path: str) -> None:
    Path(path).write_text("unsafe deserialization executed", encoding="utf-8")


class _UnsafeCheckpointValue:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return _write_marker, (str(self.marker),)


def test_load_checkpoint_rejects_objects_that_can_execute_code(tmp_path: Path) -> None:
    marker = tmp_path / "deserialization-marker"
    checkpoint = tmp_path / "unsafe.pt"
    torch.save({"model": _UnsafeCheckpointValue(marker)}, checkpoint)

    with pytest.raises(ValueError, match="unsafe or unsupported"):
        load_checkpoint(checkpoint, torch.nn.Linear(1, 1))

    assert not marker.exists()


@pytest.mark.parametrize("payload", [[], {"model": []}, {"model": {"weight": "not-a-tensor"}}])
def test_load_checkpoint_validates_payload_shape(tmp_path: Path, payload: object) -> None:
    checkpoint = tmp_path / "malformed.pt"
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="checkpoint"):
        load_checkpoint(checkpoint, torch.nn.Linear(1, 1))
