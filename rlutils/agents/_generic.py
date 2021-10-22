"""
DESCRIPTION
"""

import torch


class Agent:
    def __init__(self, env, hyperparameters):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.P = hyperparameters 
        self._reward = None
        assert len(self.env.action_space.shape) in {0,1}

    def __str__(self):
        P = "\n".join([f"| - {k} = {v}" for k, v in self.P.items()])
        return f"\n| {self.__class__.__name__} in {self.env} with hyperparameters:\n{P}\n"

    @property            
    def reward(self): return self._reward
    
    @reward.setter    
    def reward(self, r):  
        assert "reward_components" in self.P, "Agent class is incompatible with intrinsic reward"
        assert self.P["reward_components"] is not None, "Must enable decomposition to use intrinsic reward"
        assert r.m == self.P["reward_components"], "Unexpected number of reward components"
        self._reward = r  

    def per_timestep(self, state, action, reward, next_state, done): pass

    def per_episode(self): return {"logs": {}}