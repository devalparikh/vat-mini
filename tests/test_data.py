import numpy as np
import torch

from vat_mini.config import DataConfig
from vat_mini.data import STAY, VisualGridWorld, build_datasets, generate_demonstrations


def test_expert_always_reduces_distance() -> None:
    environment = VisualGridWorld(6, 24, np.random.default_rng(4))
    environment.reset()
    while not environment.reached_target:
        before = environment.shortest_distance
        _, _, _ = environment.step(environment.expert_action())
        assert environment.shortest_distance == before - 1


def test_terminal_padding_is_masked() -> None:
    arrays = generate_demonstrations(16, 12, 4, 16, seed=2)
    valid_steps = arrays["valid_steps"]
    assert (~valid_steps).any()
    for actions, mask in zip(arrays["actions"], valid_steps, strict=True):
        assert np.all(actions[~mask] == STAY)


def test_generation_is_deterministic() -> None:
    first = generate_demonstrations(2, 4, 4, 16, seed=9)
    second = generate_demonstrations(2, 4, 4, 16, seed=9)
    assert all(np.array_equal(first[key], second[key]) for key in first)


def test_action_noise_only_changes_training_split() -> None:
    clean_train, clean_validation = build_datasets(
        DataConfig(train_samples=16, validation_samples=8, sequence_length=6, action_noise=0.0),
        seed=12,
    )
    noisy_train, noisy_validation = build_datasets(
        DataConfig(train_samples=16, validation_samples=8, sequence_length=6, action_noise=1.0),
        seed=12,
    )
    assert not torch.equal(clean_train.actions, noisy_train.actions)
    assert torch.equal(clean_validation.actions, noisy_validation.actions)
