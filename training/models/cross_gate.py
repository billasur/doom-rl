"""A recurrent three-expert probability mixture on the union of nine-button actions."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.commute_moe import FrozenExpertGate, GATE_WIDTH, EXPERT_TEMPERATURES, TEMPERATURE_KEY, MOE_KEY
from models.sanskar_policy import ACTIONS, Policy, VITAL_SCALE
from models.v1_spatial_net import ACTION_TABLE, DEPTH_SIZE, RAYS, ENTITIES, TELEMETRY_SIZE, pack_policy

# Keep the two home experts' indices stable. Deduplicate by buttons, never by integer index.
UNION = np.asarray(list(dict.fromkeys(tuple(a) for a in np.concatenate((ACTION_TABLE, ACTIONS)))), dtype=np.int8)
UNION.setflags(write=False)
RGB_SIZE = 3 * 60 * 80


def native_indices(source, destination):
    # Only used for encoding action history in a frozen expert's native vocabulary.
    return (source[:,None,:] != destination[None,:,:]).sum(-1).argmin(-1)


HOME_HISTORY = native_indices(UNION, ACTION_TABLE)
UNION_INDEX = {tuple(a):i for i,a in enumerate(UNION)}


def add_pixels(base, previous, rgb):
    image = rgb[::2, ::2].transpose(2,0,1)
    return np.concatenate((base, [previous], image.reshape(-1)), dtype=np.float32)


class CrossGate(FrozenExpertGate):
    def __init__(self, observation_space, action_space, experts, temperatures):
        if len(experts)!=3 or len(temperatures)!=3 or not math.isfinite(temperatures[2]) or temperatures[2]<=0:
            raise ValueError("Expected three frozen experts and their operating temperatures")
        super().__init__(observation_space, action_space, experts[:2], temperatures[:2])
        self.temperatures = tuple(temperatures)
        teammate = Policy(use_rgb=True, critic=False)
        teammate.load_state_dict(experts[2])
        self.experts.append(teammate.eval().requires_grad_(False))
        self.depth = True  # The environment's render_buffers flag serves both RGB and depth.
        self.previous_column = GATE_WIDTH + int(self.visits) + DEPTH_SIZE
        self.gate[-1] = nn.Linear(64,4)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        for name, array in (
            ("home_columns",native_indices(ACTION_TABLE, UNION)),
            ("away_columns",native_indices(ACTIONS, UNION)),
            ("away_history",native_indices(UNION, ACTIONS)),
            ("buttons",UNION.copy()),("vital_divisor",VITAL_SCALE),
        ):
            self.register_buffer(name,torch.as_tensor(array),persistent=False)
        # USE is masked on the blended policy. The teammate's native cycle mask stays active.
        teammate.action_mask.copy_(torch.as_tensor((ACTIONS[:, None, 8] * ACTIONS[None, :, 8]).astype(bool)))

    def gate_outputs(self, packed):
        output=self.gate((packed[..., :GATE_WIDTH]/self.feature_scale).clamp(-1,1))
        return output[..., :3], output[..., 3]

    @torch.no_grad()
    def expert_log_probabilities(self, packed, state=None, starts=None):
        memory=(None,None,None) if state is None else state
        probs=[]
        next_state=[]
        previous=packed[:,self.previous_column].long()
        for i,expert in enumerate(self.experts[:2]):
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
        expert=self.experts[2]
        hidden=memory[2] if memory[2] is not None else expert.initial_hidden(batch)
        if starts is not None:
            hidden=hidden*(~starts)[None,:,None]
        logits,_,hidden=expert(obs,self.away_history[previous][None],hidden)
        probs.append(self.project((logits[0]/self.temperatures[2]).softmax(-1),self.away_columns))
        next_state.append(hidden)
        return torch.stack(probs,dim=-2),tuple(next_state)

    def project(self, probabilities, columns):
        result=probabilities.new_zeros(*probabilities.shape[:-1],len(UNION))
        return result.index_add(-1,columns,probabilities)

    def mixture(self, packed, expert_probabilities):
        routes,value=self.gate_outputs(packed)
        probabilities=(routes.softmax(-1)[...,None]*expert_probabilities).sum(-2)
        previous=packed[...,self.previous_column].long()
        repeat_edge=(self.buttons[previous][...,None,7:].bool() & self.buttons[:,7:].bool()).any(-1)
        legal=(probabilities>0) & ~repeat_edge
        logs=probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log().masked_fill(~legal,-torch.inf)
        return logs,value

    def checkpoint(self,temperature=1.0):
        result=super().checkpoint(temperature)
        result[MOE_KEY]=torch.tensor(2)
        return result

    @classmethod
    def from_checkpoint(cls,obs,act,weights):
        if weights[MOE_KEY].item()!=2:
            raise ValueError("Expected a three-expert checkpoint")
        experts=[]
        for i in range(3):
            prefix=f"experts.{i}."
            experts.append({k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)})
        model=cls(obs,act,experts,weights[EXPERT_TEMPERATURES].tolist())
        model.load_state_dict({k:v for k,v in weights.items() if k not in (MOE_KEY,EXPERT_TEMPERATURES,TEMPERATURE_KEY)})
        return model
