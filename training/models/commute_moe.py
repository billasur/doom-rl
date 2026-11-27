"""Trainable telemetry gate over two frozen recurrent categorical policies."""

import math

import torch
from torch import nn

from models.v1_spatial_net import (
    CATEGORICAL_HISTORY_SIZE, RAYS, TELEMETRY_SIZE, USE_ACTIONS, ActionHistory,
    SpatialActorCritic, checkpoint_features, mask_use,
)

MOE_KEY = "_moe_version"
EXPERT_TEMPERATURES = "_expert_temperatures"
TEMPERATURE_KEY = "_policy_temperature"
GATE_WIDTH = TELEMETRY_SIZE + CATEGORICAL_HISTORY_SIZE


class FrozenExpertGate(nn.Module):
    def __init__(self, observation_space, action_space, experts, temperatures):
        super().__init__()
        self.temperatures = tuple(float(value) for value in temperatures)
        if len(experts) != 2 or len(self.temperatures) != 2 or any(
            not math.isfinite(value) or value < 0 for value in self.temperatures
        ):
            raise ValueError("Expected two frozen experts with finite nonnegative temperatures")
        self.experts = nn.ModuleList()
        for weights in experts:
            features = checkpoint_features(weights)
            if not features["categorical"]:
                raise ValueError("The gate requires two categorical spatial experts")
            expert = SpatialActorCritic(observation_space, action_space, **features)
            expert.load_state_dict(weights)
            self.experts.append(expert.eval().requires_grad_(False))
        self.depth = any(expert.depth_encoder is not None for expert in self.experts)
        self.visits = any(expert.visit_bias is not None for expert in self.experts)
        self.gate = nn.Sequential(nn.Linear(GATE_WIDTH, 64), nn.Tanh(), nn.Linear(64, 3))
        with torch.no_grad():
            self.gate[-1].weight.zero_()
            self.gate[-1].bias.zero_()
        expert = self.experts[0]
        scale = torch.cat((
            torch.ones(RAYS), torch.full((RAYS,), expert.ray_type.num_embeddings),
            expert.entity_scale.repeat(32), torch.full((32,), expert.entity_type.num_embeddings),
            torch.ones(32), expert.vitals_scale, torch.ones(CATEGORICAL_HISTORY_SIZE),
        ))
        self.register_buffer("feature_scale", scale, persistent=False)

    def history(self):
        return ActionHistory(categorical=True, visits=self.visits)

    def train(self, mode=True):
        super().train(mode)
        self.experts.eval()
        return self

    def gate_outputs(self, packed):
        scores = self.gate((packed[..., :GATE_WIDTH] / self.feature_scale).clamp(-1, 1))
        return scores[..., :2], scores[..., 2]

    @torch.no_grad()
    def expert_log_probabilities(self, packed, state=None, starts=None):
        # The packed previous action is the same action both experts actually experienced.
        prev_used = packed[..., TELEMETRY_SIZE + USE_ACTIONS[0]:TELEMETRY_SIZE + USE_ACTIONS[-1] + 1].bool().any(-1)
        if state is None:
            state = (None, None)
        probabilities, next_state = [], []
        for expert, temperature, memory in zip(self.experts, self.temperatures, state):
            if memory is not None and starts is not None:
                memory = memory.reset_where(starts)
            visit_end = GATE_WIDTH + int(expert.visit_bias is not None)
            expert_input = packed[..., :visit_end]
            if expert.depth_encoder is not None:
                depth_start = GATE_WIDTH + int(self.visits)
                expert_input = (packed if depth_start == visit_end
                                else torch.cat((expert_input, packed[..., depth_start:]), dim=-1))
            logits, _, _, memory = expert(expert_input, memory)
            scores = mask_use(logits / (temperature or 1.0), prev_used)
            if temperature == 0:
                scores = torch.full_like(scores, -1e9).scatter(-1, scores.argmax(-1, keepdim=True), 0.0)
            probabilities.append(scores.log_softmax(dim=-1))
            next_state.append(memory)
        return torch.stack(probabilities, dim=-2), tuple(next_state)

    def mixture(self, packed, expert_logs):
        routes, value = self.gate_outputs(packed)
        logits = torch.logsumexp(routes.log_softmax(dim=-1)[..., None] + expert_logs, dim=-2)
        previous_use = packed[..., TELEMETRY_SIZE+USE_ACTIONS[0]:TELEMETRY_SIZE+USE_ACTIONS[-1]+1].bool().any(-1)
        return mask_use(logits, previous_use), value

    def checkpoint(self, temperature=1.0):
        weights = {key: value.detach().cpu() for key, value in self.state_dict().items()}
        weights[MOE_KEY] = torch.tensor(1)
        weights[EXPERT_TEMPERATURES] = torch.tensor(self.temperatures, dtype=torch.float64)
        weights[TEMPERATURE_KEY] = torch.tensor(temperature, dtype=torch.float64)
        return weights

    @classmethod
    def from_checkpoint(cls, observation_space, action_space, weights):
        if weights[MOE_KEY].item() != 1:
            raise ValueError("Unsupported frozen-expert gate checkpoint version")
        prefixes = ("experts.0.", "experts.1.")
        experts = [{key.removeprefix(prefix): value for key, value in weights.items()
                    if key.startswith(prefix)} for prefix in prefixes]
        model = cls(observation_space, action_space, experts, weights[EXPERT_TEMPERATURES].tolist())
        model.load_state_dict({key: value for key, value in weights.items()
                               if key not in (MOE_KEY, EXPERT_TEMPERATURES, TEMPERATURE_KEY)})
        return model
