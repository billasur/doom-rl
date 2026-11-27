"""Shared observation encoder and recurrent policy for training and export."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ACTIONS = np.zeros((20, 9), dtype=np.int64)
for index, buttons in enumerate((
    (), (0,), (1,), (2,), (3,), (4,), (5,), (0, 4), (0, 5), (0, 2), (0, 3),
    (6,), (0, 6), (4, 6), (5, 6), (2, 6), (3, 6), (7,), (0, 7), (8,),
)):
    ACTIONS[index, list(buttons)] = 1

# Edge-triggered buttons must have a released step between presses.
ACTION_MASK = torch.from_numpy((ACTIONS[:, None, 7:] * ACTIONS[None, :, 7:]).any(axis=-1))
RAY_TYPES = np.eye(6, dtype=np.float32)
ENTITY_TYPES = np.eye(12, dtype=np.float32)
VITAL_SCALE = np.array([100, 100, 8, *([50] * 5), *([1] * 6), 20, 20, 20], np.float32)
PRIV_SIZE = 10  # time, health, six keys, sectors and cells


def obs_to_tensor(obs, use_rgb=False):
    """Encode one contract observation as CPU tensors, with no batch dimensions."""
    rays, entities = obs["rays"], obs["visible_entities"]
    distance = rays["distance"][None]
    encoded = {
        "rays": torch.from_numpy(np.concatenate((distance, 1 - distance, RAY_TYPES[rays["hit_type"]].T))),
        "entities": torch.from_numpy(np.concatenate((entities["features"], ENTITY_TYPES[entities["entity_type"]]), axis=-1)),
        "mask": torch.from_numpy(entities["mask"].astype(bool)),
        "vitals": torch.from_numpy(obs["vitals"] / VITAL_SCALE),
    }
    if use_rgb:
        encoded["rgb"] = torch.from_numpy(obs["rgb"][::2, ::2].transpose(2, 0, 1).copy())
    return encoded


class Policy(nn.Module):
    def __init__(self, use_rgb=False, critic=True, aux=False):
        super().__init__()
        self.use_rgb = use_rgb
        self.register_buffer("action_mask", ACTION_MASK.clone(), persistent=False)
        self.ray = nn.Sequential(
            nn.Conv1d(8, 32, 5, stride=2, padding=2, padding_mode="circular"), nn.ReLU(),
            nn.Conv1d(32, 32, 5, stride=2, padding=2, padding_mode="circular"), nn.ReLU(),
            nn.AvgPool1d(2), nn.Flatten(), nn.Linear(256, 128), nn.ReLU(),
        )
        self.entity = nn.Sequential(nn.Linear(17, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        if use_rgb:
            self.rgb = nn.Sequential(
                nn.Conv2d(3, 8, 5, stride=4, padding=2), nn.ReLU(),
                nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((3, 4)), nn.Flatten(), nn.Linear(192, 32), nn.ReLU(),
            )
        self.input = nn.Sequential(nn.Linear(128 + 64 + 17 + len(ACTIONS) + 32 * use_rgb, 256), nn.ReLU())
        self.gru = nn.GRU(256, 256)
        self.actor = nn.Linear(256, len(ACTIONS))
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        if critic:
            self.value = nn.Sequential(nn.Linear(256 + PRIV_SIZE, 128), nn.ReLU(), nn.Linear(128, 1))
        if critic and aux:
            self.aux = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 4))

    def initial_hidden(self, batch=1):
        return self.actor.weight.new_zeros(1, batch, 256)

    def forward(self, obs, prev, hidden, reset=None, privileged=None, predict_aux=False):
        """Time-major [T, B, ...] inputs. reset marks episode starts within chunks."""
        mask = obs["mask"].flatten(0, 1).unsqueeze(-1)
        entities = self.entity(obs["entities"].flatten(0, 1)).masked_fill(~mask, 0)
        pooled = [entities.amax(1), entities.sum(1) / mask.sum(1).clamp_min(1)]
        inputs = [self.ray(obs["rays"].flatten(0, 1)), *pooled, obs["vitals"].flatten(0, 1),
                  F.one_hot(prev.flatten(), len(ACTIONS)).float()]
        if self.use_rgb:
            inputs.append(self.rgb(obs["rgb"].flatten(0, 1).float() / 255))
        encoded = self.input(torch.cat(inputs, dim=-1)).reshape(*prev.shape, 256)

        # Use fused GRU calls between resets. Reset only the sequences ending at each boundary.
        if reset is None:
            output, hidden = self.gru(encoded, hidden)
        else:
            boundaries = (reset[1:].any(dim=1).nonzero().flatten() + 1).tolist()
            pieces = []
            for start, end in zip([0, *boundaries], [*boundaries, len(encoded)]):
                hidden = hidden * (~reset[start]).reshape(1, -1, 1)
                output, hidden = self.gru(encoded[start:end], hidden)
                pieces.append(output)
            output = torch.cat(pieces)

        logits = self.actor(output).masked_fill(self.action_mask[prev], -1e9)
        value = None
        if privileged is not None:
            value = self.value(torch.cat((output, privileged), dim=-1)).squeeze(-1)
        if predict_aux:
            return logits, value, hidden, self.aux(output)
        return logits, value, hidden


