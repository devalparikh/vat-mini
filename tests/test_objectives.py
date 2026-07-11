import torch

from vat_mini.objectives import AdvantageWeightedImitationObjective, ContinuousBehaviorCloningObjective


def test_advantage_weighting_is_finite_for_one_valid_token() -> None:
    objective = AdvantageWeightedImitationObjective(temperature=0.5, maximum_weight=20.0)
    logits = torch.zeros(1, 2, 5, requires_grad=True)
    actions = torch.tensor([[1, 0]])
    rewards = torch.tensor([[1.0, 0.0]])
    valid_steps = torch.tensor([[True, False]])
    result = objective(logits, actions, rewards, valid_steps)
    assert torch.isfinite(result.loss)
    result.loss.backward()


def test_continuous_behavior_cloning_masks_padding() -> None:
    objective = ContinuousBehaviorCloningObjective()
    predictions = torch.tensor([[[0.0, 0.0], [100.0, 100.0]]], requires_grad=True)
    actions = torch.tensor([[[1.0, -1.0], [0.0, 0.0]]])
    result = objective(
        predictions, actions, torch.zeros(1, 2), torch.tensor([[True, False]])
    )
    torch.testing.assert_close(result.loss, torch.tensor(1.0))
    assert result.token_accuracy == 1.0
    result.loss.backward()
