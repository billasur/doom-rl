"""Telemetry encoder and recurrent policy for the contract's nine buttons."""

import numpy as np
import torch
from gymnasium.spaces import MultiBinary
from torch import nn


VITALS_SCALE = np.array(
    [200, 200, 7, 100, 200, 50, 50, 300, 1, 1, 1, 1, 1, 1, 32, 32, 32],
    dtype=np.float32,
)
ENTITY_SCALE = np.array([4, 1, 1, 32, 32], dtype=np.float32)


class TelemetryEncoder:
    """Flatten a single observation or a batch of vector-environment observations."""

    def __init__(self, observation_space):
        rays = observation_space["rays"]
        entities = observation_space["visible_entities"]
        self.hit_types = np.arange(int(rays["hit_type"].nvec.max()))
        self.entity_types = np.arange(int(entities["entity_type"].nvec.max()))
        self.size = (
            observation_space["vitals"].shape[0]
            + rays["distance"].shape[0] * (1 + self.hit_types.size)
            + entities["mask"].shape[0]
            * (entities["features"].shape[-1] + self.entity_types.size + 1)
        )

    def __call__(self, observation):
        rays = observation["rays"]
        entities = observation["visible_entities"]
        mask = entities["mask"].astype(bool)
        batch_shape = mask.shape[:-1]
        vitals = np.clip(observation["vitals"] / VITALS_SCALE, -1, 1)
        ray_types = rays["hit_type"][..., None] == self.hit_types
        entity_features = np.clip(entities["features"] / ENTITY_SCALE, -1, 1)
        entity_features = np.where(mask[..., None], entity_features, 0.0)
        entity_types = (entities["entity_type"][..., None] == self.entity_types) & mask[..., None]
        return np.concatenate(
            (
                vitals,
                rays["distance"],
                ray_types.reshape(*batch_shape, -1),
                entity_features.reshape(*batch_shape, -1),
                entity_types.reshape(*batch_shape, -1),
                mask,
            ),
            axis=-1,
            dtype=np.float32,
        )


class RecurrentActorCritic(nn.Module):
    hidden_size = 128

    def __init__(self, telemetry_size, action_space):
        super().__init__()
        if not isinstance(action_space, MultiBinary) or action_space.shape != (9,):
            raise ValueError(f"Expected the contract's MultiBinary(9), got {action_space}")
        self.encoder = nn.Sequential(
            nn.Linear(telemetry_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.gru = nn.GRU(128, self.hidden_size)
        self.actor = nn.Linear(self.hidden_size, action_space.shape[0])
        self.critic = nn.Linear(self.hidden_size, 1)

    def forward(self, telemetry, hidden=None):
        """Accept time-major telemetry shaped (steps, environments, features)."""
        features, hidden = self.gru(self.encoder(telemetry), hidden)
        return self.actor(features), self.critic(features).squeeze(-1), hidden
