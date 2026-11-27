"""Finalist's standalone inference entry point."""
from pathlib import Path

import torch

from models.commute_moe import TEMPERATURE_KEY
from models.cross_gate import HOME_HISTORY, UNION, add_pixels
from models.final_gate import FinalGate
from models.v1_spatial_net import pack_policy

V2_WEIGHTS_PATH = Path(__file__).with_name("v2_weights.pt")


V1_WEIGHTS_PATH = V2_WEIGHTS_PATH


class DoomAgent:
    def __init__(self, observation_space, action_space, device):
        self.device = torch.device(device)
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
        self.model = FinalGate.from_checkpoint(self.spaces, weights).to(self.device).eval()
        self.reset()

    def reset(self):
        self.state = self.context = None
        self.previous = 0
        self.history = self.model.history()

    @torch.inference_mode()
    def act(self, observation):
        base = pack_policy(observation, self.history, True)
        packed = torch.from_numpy(add_pixels(base, self.previous, observation["rgb"])).to(self.device)[None]
        experts, self.state = self.model.expert_log_probabilities(packed, self.state)
        routes, _, self.context = self.model.gate_outputs(packed, self.context)
        if self.model.mode == "hard":
            choice = routes.argmax(-1) if self.temperature == 0 else torch.multinomial(routes.softmax(-1), 1).flatten()
            probabilities = experts[torch.arange(len(experts), device=self.device), choice]
        else:
            probabilities = ((routes / self.model.tau).softmax(-1)[..., None] * experts).sum(-2)
        logits = self.model.action_logits(probabilities, packed)
        action = logits.argmax(-1) if self.temperature == 0 else torch.multinomial((logits / self.temperature).softmax(-1), 1)
        self.previous = int(action.item())
        self.history.action = int(HOME_HISTORY[self.previous])
        return UNION[self.previous].copy()
