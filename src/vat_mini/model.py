"""A compact causal vision-action transformer."""

from __future__ import annotations

import torch
from torch import nn

from vat_mini.config import ModelConfig


class AdaptiveAvgPool2d(nn.Module):
    """Adaptive average pooling that also works on MPS.

    ``nn.AdaptiveAvgPool2d`` errors on MPS when the input is not divisible by
    the output (e.g. an 84px frame becomes a 21x21 map, indivisible by 4). We
    compute the same variable-sized pooling windows explicitly, which is cheap
    for the tiny 4x4 output here and behaves identically across devices.
    """

    def __init__(self, output_size: tuple[int, int]):
        super().__init__()
        self.output_height, self.output_width = output_size

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _, _, height, width = features.shape
        row_bounds = self._window_bounds(height, self.output_height)
        column_bounds = self._window_bounds(width, self.output_width)
        rows = [
            torch.stack(
                [features[:, :, top:bottom, left:right].mean(dim=(2, 3)) for left, right in column_bounds],
                dim=-1,
            )
            for top, bottom in row_bounds
        ]
        return torch.stack(rows, dim=-2)

    @staticmethod
    def _window_bounds(input_size: int, output_size: int) -> list[tuple[int, int]]:
        return [
            (index * input_size // output_size, -(-(index + 1) * input_size // output_size))
            for index in range(output_size)
        ]


class VisionEncoder(nn.Module):
    """Converts each RGB frame into one learned visual token."""

    def __init__(self, vision_width: int, embedding_dim: int):
        super().__init__()
        self.convolutional_backbone = nn.Sequential(
            nn.Conv2d(3, vision_width, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(vision_width, vision_width * 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            # A 4x4 map preserves where agent and target appear; global pooling
            # would erase precisely the spatial relation the policy must learn.
            AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.projection = nn.Linear(vision_width * 2 * 4 * 4, embedding_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.projection(self.convolutional_backbone(frames))


class VisionActionTransformer(nn.Module):
    """Predicts each action from the visual history and previous actions.

    At time t the token is: vision(frame_t) + embedding(action_(t-1)) + position_t.
    A causal mask prevents information from future frames from leaking backward.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.begin_action_id = config.num_actions
        self.vision_encoder = VisionEncoder(config.vision_width, config.embedding_dim)
        if config.action_type == "discrete":
            self.previous_action_embedding: nn.Module = nn.Embedding(
                config.num_actions + 1, config.embedding_dim
            )
            output_dimension = config.num_actions
        else:
            self.previous_action_embedding = nn.Linear(config.action_dimension, config.embedding_dim)
            output_dimension = config.action_dimension
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.transformer_layers,
            norm=nn.LayerNorm(config.embedding_dim),
            enable_nested_tensor=False,
        )
        action_projection = nn.Linear(config.embedding_dim, output_dimension)
        self.action_head = (
            action_projection
            if config.action_type == "discrete"
            else nn.Sequential(action_projection, nn.Tanh())
        )

    def shifted_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.config.action_type == "continuous":
            begin = torch.zeros(
                actions.shape[0], 1, self.config.action_dimension,
                dtype=actions.dtype, device=actions.device,
            )
            return torch.cat((begin, actions[:, :-1]), dim=1)
        begin = torch.full(
            (actions.shape[0], 1), self.begin_action_id, dtype=torch.long, device=actions.device
        )
        return torch.cat((begin, actions[:, :-1]), dim=1)

    def forward(self, observations: torch.Tensor, previous_actions: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 5:
            raise ValueError("observations must have shape [batch, time, channels, height, width]")
        batch_size, sequence_length = observations.shape[:2]
        expected_action_shape = (
            (batch_size, sequence_length)
            if self.config.action_type == "discrete"
            else (batch_size, sequence_length, self.config.action_dimension)
        )
        if previous_actions.shape != expected_action_shape:
            raise ValueError(f"previous_actions must have shape {expected_action_shape}")
        if sequence_length > self.config.max_sequence_length:
            raise ValueError("input sequence exceeds max_sequence_length")

        frames = observations.reshape(batch_size * sequence_length, *observations.shape[2:])
        vision_tokens = self.vision_encoder(frames).reshape(batch_size, sequence_length, -1)
        positions = torch.arange(sequence_length, device=observations.device).unsqueeze(0)
        tokens = (
            vision_tokens
            + self.previous_action_embedding(previous_actions)
            + self.position_embedding(positions)
        )
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=observations.device),
            diagonal=1,
        )
        return self.action_head(self.transformer(tokens, mask=causal_mask))

    @torch.no_grad()
    def choose_action(self, observation_history: torch.Tensor, action_history: torch.Tensor) -> int:
        """Greedily select the next action from an unbatched rollout history."""
        if self.config.action_type != "discrete":
            raise RuntimeError("choose_action is only available for discrete closed-loop environments")
        if observation_history.shape[0] > self.config.max_sequence_length:
            observation_history = observation_history[-self.config.max_sequence_length :]
            action_history = action_history[-(self.config.max_sequence_length - 1) :]
        previous_actions = torch.cat(
            (
                torch.tensor([self.begin_action_id], device=observation_history.device),
                action_history,
            )
        ).unsqueeze(0)
        logits = self(observation_history.unsqueeze(0), previous_actions)
        return int(logits[0, -1].argmax().item())

    @torch.no_grad()
    def choose_action_continuous(
        self, observation_history: torch.Tensor, action_history: torch.Tensor
    ) -> torch.Tensor:
        """Predict the next continuous action vector from an unbatched rollout history.

        ``observation_history`` has shape ``[T, C, H, W]`` and ``action_history``
        holds the ``[T - 1, action_dimension]`` actions already taken this episode.
        The begin-of-sequence action is a zero vector, matching ``shifted_actions``.
        """
        if self.config.action_type != "continuous":
            raise RuntimeError(
                "choose_action_continuous is only available for continuous-action policies"
            )
        if observation_history.shape[0] > self.config.max_sequence_length:
            observation_history = observation_history[-self.config.max_sequence_length :]
            action_history = action_history[-(self.config.max_sequence_length - 1) :]
        begin = torch.zeros(
            1, self.config.action_dimension,
            dtype=observation_history.dtype, device=observation_history.device,
        )
        previous_actions = torch.cat((begin, action_history)).unsqueeze(0)
        predictions = self(observation_history.unsqueeze(0), previous_actions)
        return predictions[0, -1]
