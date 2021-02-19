from ..common.networks import SequentialNetwork

import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F


DEFAULT_HYPERPARAMETERS = {
    "method": "uniform",
}


class RandomAgent:
    def __init__(self, 
                 action_space, 
                 hyperparameters=DEFAULT_HYPERPARAMETERS
                 ):
        self.action_space = action_space
        self.P = hyperparameters

    def act(self, state, explore=True):
        """Random action selection."""
        return self.action_space.sample(), {}

    def per_timestep(self, state, action, reward, next_state): return
    def per_episode(self): return