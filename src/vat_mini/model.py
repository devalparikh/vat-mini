"""A compact causal vision-action transformer."""

from __future__ import annotations

import torch
from torch import nn

from vat_mini.config import ModelConfig


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
            nn.AdaptiveAvgPool2d((4, 4)),
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
        self.previous_action_embedding = nn.Embedding(config.num_actions + 1, config.embedding_dim)
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
        self.action_head = nn.Linear(config.embedding_dim, config.num_actions)

    def shifted_actions(self, actions: torch.Tensor) -> torch.Tensor:
        begin = torch.full(
            (actions.shape[0], 1), self.begin_action_id, dtype=torch.long, device=actions.device
        )
        return torch.cat((begin, actions[:, :-1]), dim=1)

    def forward(self, observations: torch.Tensor, previous_actions: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 5:
            raise ValueError("observations must have shape [batch, time, channels, height, width]")
        batch_size, sequence_length = observations.shape[:2]
        if previous_actions.shape != (batch_size, sequence_length):
            raise ValueError("previous_actions must have shape [batch, time]")
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
