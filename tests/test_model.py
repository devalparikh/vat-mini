import torch

from vat_mini.config import ModelConfig
from vat_mini.model import VisionActionTransformer


def small_model() -> VisionActionTransformer:
    return VisionActionTransformer(
        ModelConfig(
            vision_width=4,
            embedding_dim=16,
            transformer_layers=1,
            attention_heads=2,
            feedforward_dim=32,
            dropout=0.0,
            max_sequence_length=8,
        )
    ).eval()


def test_model_output_shape() -> None:
    model = small_model()
    observations = torch.rand(2, 4, 3, 16, 16)
    actions = torch.randint(0, 5, (2, 4))
    assert model(observations, model.shifted_actions(actions)).shape == (2, 4, 5)


def test_causal_mask_blocks_future_observations() -> None:
    torch.manual_seed(1)
    model = small_model()
    observations = torch.rand(1, 4, 3, 16, 16)
    changed_future = observations.clone()
    changed_future[:, 2:] = torch.rand_like(changed_future[:, 2:])
    actions = torch.tensor([[1, 2, 3, 4]])
    first = model(observations, model.shifted_actions(actions))
    second = model(changed_future, model.shifted_actions(actions))
    torch.testing.assert_close(first[:, :2], second[:, :2])

