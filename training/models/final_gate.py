"""Frozen specialists with a recurrent sharp neural router on legal telemetry."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.commute_moe import GATE_WIDTH, EXPERT_TEMPERATURES, TEMPERATURE_KEY, MOE_KEY
from models.cross_gate import UNION, add_pixels, native_indices
from models.sanskar_policy import ACTIONS, Policy, VITAL_SCALE
from models.v1_spatial_net import (
    ACTION_TABLE, DEPTH_SIZE, RAYS, ENTITIES, TELEMETRY_SIZE, ActionHistory,
    SpatialActorCritic, checkpoint_features,
)

META = "_final_router"
ENTITY_START = 2 * RAYS + ENTITIES * 5
FEATURE_WIDTH = 17 + 6 + 8 + 12 + 8 + len(UNION)


class FinalGate(nn.Module):
    def __init__(self, spaces, weights, temperatures, mode="soft", tau=.20, architecture="gru", *, rgb_expert=True):
        super().__init__()
        if len(weights) not in (2, 3) or len(weights) != len(temperatures):
            raise ValueError("Expected two or three frozen experts and their temperatures")
        if any(not math.isfinite(value) or value <= 0 for value in temperatures):
            raise ValueError("Frozen expert temperatures must be positive and finite")
        if mode not in ("soft", "hard") or tau not in (.20, .35) or architecture not in ("gru", "mlp"):
            raise ValueError("Unknown router mode, temperature or architecture")
        self.mode, self.tau, self.architecture = mode, tau, architecture
        self.temperatures = tuple(temperatures)
        self.rgb_expert = rgb_expert
        self.home_count = len(weights) - int(rgb_expert)
        homes = []
        for state in weights[:self.home_count]:
            features = checkpoint_features(state)
            if not features["categorical"]:
                raise ValueError("Home experts must use the native categorical actions")
            model = SpatialActorCritic(*spaces, **features)
            model.load_state_dict(state)
            homes.append(model)
        if self.rgb_expert:
            sanskar = Policy(use_rgb=True, critic=False)
            sanskar.load_state_dict(weights[-1])
            sanskar.action_mask.zero_()
            homes.append(sanskar)
        self.experts = nn.ModuleList(homes).eval().requires_grad_(False)
        self.depth = True
        self.visits = any(expert.visit_bias is not None for expert in homes[:self.home_count])
        self.previous_column = GATE_WIDTH + int(self.visits) + DEPTH_SIZE
        self.context_width = 64 if architecture == "gru" else len(weights)
        self.router = (nn.GRUCell(FEATURE_WIDTH, 64) if architecture == "gru"
                       else nn.Sequential(nn.Linear(FEATURE_WIDTH + self.context_width, 64), nn.Tanh()))
        self.head = nn.Linear(64, len(weights) + 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[len(weights) - 1] = 1.2
        for name, array in (
            ("home_columns", native_indices(ACTION_TABLE, UNION)),
            ("away_columns", native_indices(ACTIONS, UNION)),
            ("away_history", native_indices(UNION, ACTIONS)),
            ("buttons", UNION.copy()), ("vital_divisor", VITAL_SCALE),
        ):
            self.register_buffer(name, torch.as_tensor(array), persistent=False)

    def history(self):
        return ActionHistory(categorical=True, visits=self.visits)

    def features(self, packed):
        hits = F.one_hot(packed[..., RAYS:2 * RAYS].long(), 6).float().mean(-2)
        rays = packed[..., :RAYS].unflatten(-1, (8, 8)).mean(-1)
        types = F.one_hot(packed[..., ENTITY_START:ENTITY_START + ENTITIES].long(), 12).float()
        visible = packed[..., ENTITY_START + ENTITIES:ENTITY_START + 2 * ENTITIES, None]
        entities = (types * visible).mean(-2)
        vitals = (packed[..., TELEMETRY_SIZE - 17:TELEMETRY_SIZE] / self.vital_divisor).clamp(-1, 1)
        depth = packed[..., self.previous_column - DEPTH_SIZE:self.previous_column].unflatten(-1, (8, 32)).mean(-1)
        action = F.one_hot(packed[..., self.previous_column].long(), len(UNION)).float()
        return torch.cat((vitals, hits, rays, entities, depth, action), dim=-1)

    def gate_outputs(self, packed, context=None):
        features = self.features(packed)
        if context is None:
            context = features.new_zeros((*features.shape[:-1], self.context_width))
        if self.architecture == "gru":
            hidden = self.router(features, context)
        else:
            hidden = self.router(torch.cat((features, context), dim=-1))
        output = self.head(hidden)
        scores, value = output[..., :-1], output[..., -1]
        next_context = hidden if self.architecture == "gru" else (scores / self.routing_temperature).softmax(-1)
        return scores, value, next_context

    @property
    def routing_temperature(self):
        return self.tau if self.mode == "soft" else 1.0

    def edge_mask(self, packed):
        previous = self.buttons[packed[..., self.previous_column].long()][..., 7:].bool()
        return (previous[..., None, :] & self.buttons[..., 7:].bool()).any(-1)

    def action_logits(self, probabilities, packed):
        return probabilities.clamp_min(1e-30).log().masked_fill((probabilities <= 0) | self.edge_mask(packed), -1e9)

    @torch.no_grad()
    def expert_log_probabilities(self, packed, state=None, starts=None):
        memory=(None,) * len(self.experts) if state is None else state
        probs=[]
        next_state=[]
        previous=packed[:,self.previous_column].long()
        for i,expert in enumerate(self.experts[:self.home_count]):
            hidden=memory[i]
            if hidden is not None and starts is not None:
                hidden=hidden.reset_where(starts)
            base=packed[:,:GATE_WIDTH+int(expert.visit_bias is not None)]
            if expert.depth_encoder is not None:
                base=torch.cat((base,packed[:,self.previous_column-DEPTH_SIZE:self.previous_column]),dim=-1)
            logits,_,_,hidden=expert(base,hidden)
            t=self.temperatures[i]
            p=(logits/t).softmax(-1) if t else F.one_hot(logits.argmax(-1),len(ACTION_TABLE)).float()
            probs.append(self.project(p,self.home_columns))
            next_state.append(hidden)
        if not self.rgb_expert:
            return torch.stack(probs, dim=-2), tuple(next_state)
        batch=len(packed)
        distances=packed[:,:RAYS].unsqueeze(1)
        ray_types=F.one_hot(packed[:,RAYS:2*RAYS].long(),6).transpose(1,2)
        entity_start=2*RAYS+ENTITIES*5
        entities=packed[:,2*RAYS:entity_start].reshape(batch,ENTITIES,5)
        types=F.one_hot(packed[:,entity_start:entity_start+ENTITIES].long(),12)
        obs={
            "rays":torch.cat((distances,1-distances,ray_types),1)[None],
            "entities":torch.cat((entities,types),-1)[None],
            "mask":packed[:,entity_start+ENTITIES:entity_start+2*ENTITIES].bool()[None],
            "vitals":(packed[:,TELEMETRY_SIZE-17:TELEMETRY_SIZE]/self.vital_divisor)[None],
            "rgb":packed[:,self.previous_column+1:].reshape(1,batch,3,60,80),
        }
        expert=self.experts[-1]
        hidden=memory[-1] if memory[-1] is not None else expert.initial_hidden(batch)
        if starts is not None:
            hidden=hidden*(~starts)[None,:,None]
        logits,_,hidden=expert(obs,self.away_history[previous][None],hidden)
        probs.append(self.project((logits[0]/self.temperatures[-1]).softmax(-1),self.away_columns))
        next_state.append(hidden)
        return torch.stack(probs,dim=-2),tuple(next_state)

    def project(self, probabilities, columns):
        result=probabilities.new_zeros(*probabilities.shape[:-1],len(UNION))
        return result.index_add(-1,columns,probabilities)

    def checkpoint(self, temperature=1.0):
        weights = {key: value.detach().cpu() for key, value in self.state_dict().items()}
        weights[MOE_KEY] = torch.tensor(3 if self.rgb_expert else 4)
        weights[EXPERT_TEMPERATURES] = torch.tensor(self.temperatures, dtype=torch.float64)
        weights[TEMPERATURE_KEY] = torch.tensor(temperature, dtype=torch.float64)
        weights[META] = torch.tensor([self.mode == "hard", self.tau, self.architecture == "mlp"], dtype=torch.float64)
        return weights

    @classmethod
    def from_checkpoint(cls, spaces, weights):
        version = weights[MOE_KEY].item()
        if version not in (3, 4):
            raise ValueError("Unsupported final router checkpoint version")
        hard, tau, mlp = weights[META].tolist()
        temperatures = weights[EXPERT_TEMPERATURES].tolist()
        experts = []
        for i in range(len(temperatures)):
            prefix = f"experts.{i}."
            experts.append({key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)})
        model = cls(spaces, experts, temperatures, "hard" if hard else "soft", tau, "mlp" if mlp else "gru",
                    rgb_expert=version == 3)
        metadata = (MOE_KEY, EXPERT_TEMPERATURES, TEMPERATURE_KEY, META)
        model.load_state_dict({key: value for key, value in weights.items() if key not in metadata})
        return model
