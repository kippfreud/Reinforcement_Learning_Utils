"""
DESCRIPTION
"""

import torch


class RandomAgent:
    def __init__(self, env, hyperparameters):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.P = hyperparameters
        self.action_space = env.action_space
        self.a_last = self.action_space.sample()

    def act(self, state, explore=True):
        """Random action selection."""
        a = self.action_space.sample()
        if self.P["inertia"] > 0:
            a += self.P["inertia"] * (self.a_last - a)
            self.a_last = a
        return a, {}

    def per_timestep(self, state, action, reward, next_state): return
    def per_episode(self): return