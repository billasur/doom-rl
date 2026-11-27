"""Keep the referee RGB pixels and executed action in the PPO collector."""
import gymnasium as gym
import numpy as np
from models.cross_gate import RGB_SIZE, UNION_INDEX, HOME_HISTORY, add_pixels
from train.train_v1 import CurriculumEnv


class CrossCurriculum(CurriculumEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        width=self.observation_space.shape[0]+1+RGB_SIZE
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,(width,),dtype=np.float32)

    def reset(self,**kwargs):
        base,info=super().reset(**kwargs)
        return add_pixels(base,0,self.env.unwrapped.render()),info

    def step(self,action):
        # CompactTelemetry can encode only the home action table in its native history.
        # The simulator receives the actual union action, including NEXT_WEAPON.
        idx=UNION_INDEX[tuple(action)]
        self.env.history.action=int(HOME_HISTORY[idx])
        obs,reward,terminated,truncated,info=self.env.env.step(action)
        packed=add_pixels(self.env.observation(obs),idx,obs["rgb"])
        return packed,reward,terminated,truncated,self.annotate(info)
