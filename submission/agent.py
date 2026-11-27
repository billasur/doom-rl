"""Finalist's standalone inference entry point."""
from pathlib import Path

import numpy as np
import torch

from submission.cross_team_models.commute_moe import TEMPERATURE_KEY
from submission.cross_team_models.cross_gate import HOME_HISTORY, UNION, add_pixels
from submission.cross_team_models.final_gate import FinalGate
from submission.cross_team_models.v1_spatial_net import pack_policy

ROUTE_TEMPERATURE_KEY = "_route_temperature"

V2_WEIGHTS_PATH = Path(__file__).with_name("v2_weights.pt")


V1_WEIGHTS_PATH = V2_WEIGHTS_PATH


class DoomAgent:
    def __init__(self, observation_space, action_space, device):
        requested_device = torch.device(device)
        # This compact policy meets the scorer budget on one CPU and needs no CUDA context.
        self.device = torch.device("cpu") if requested_device.type == "cuda" else requested_device
        self.spaces = (observation_space, action_space)
        self.depth = True
        self.spatial = self.categorical = True
        self.model = None
        if self.device.type == "cpu":
            torch.set_num_threads(1)
        if V2_WEIGHTS_PATH.is_file():
            self.load_checkpoint(V2_WEIGHTS_PATH)

    def load_checkpoint(self, path):
        weights = torch.load(path, map_location=self.device, weights_only=True)
        self.temperature = float(weights[TEMPERATURE_KEY])
        self.route_temperature = float(weights.pop(ROUTE_TEMPERATURE_KEY, 1.0))
        self.model = FinalGate.from_checkpoint(self.spaces, weights).to(self.device).eval()
        self.reset()

    def reset(self):
        self.state = self.context = None
        self.previous = 0
        self.history = self.model.history()

    @torch.inference_mode()
    def act(self, observation):
        base = pack_policy(observation, self.history, True)
        if self.model.rgb_expert:
            packed = add_pixels(base, self.previous, observation["rgb"])
        else:
            # The released router has no RGB expert, so skip packing the frame.
            packed = np.concatenate((base, [self.previous]), dtype=np.float32)
        packed = torch.from_numpy(packed).to(self.device)[None]
        experts, self.state = self.model.expert_log_probabilities(packed, self.state)
        routes, _, self.context = self.model.gate_outputs(packed, self.context)
        if self.model.mode == "hard":
            choice = routes.argmax(-1) if self.temperature == 0 else torch.multinomial((routes / self.route_temperature).softmax(-1), 1).flatten()
            probabilities = experts[torch.arange(len(experts), device=self.device), choice]
        else:
            probabilities = ((routes / self.model.tau).softmax(-1)[..., None] * experts).sum(-2)
        logits = self.model.action_logits(probabilities, packed)
        action = logits.argmax(-1) if self.temperature == 0 else torch.multinomial((logits / self.temperature).softmax(-1), 1)
        self.previous = int(action.item())
        self.history.action = int(HOME_HISTORY[self.previous])
        return UNION[self.previous].copy()
