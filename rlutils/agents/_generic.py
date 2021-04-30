"""
DESCRIPTION
"""

import torch


class Agent:
    def __init__(self, env, hyperparameters):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.P = hyperparameters 
        assert len(self.env.action_space.shape) in {0,1}

    def __str__(self):
        P = "\n".join([f"| - {k} = {v}" for k, v in self.P.items()])
        return f"\n| {self.__class__.__name__} in {self.env} with hyperparameters:\n{P}\n"