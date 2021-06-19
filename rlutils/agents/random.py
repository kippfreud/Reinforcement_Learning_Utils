"""
DESCRIPTION
"""

from ._generic import Agent


class RandomAgent(Agent):
    def __init__(self, env, hyperparameters):
        Agent.__init__(self, env, hyperparameters)
        self.a_last = self.env.action_space.sample()

    def act(self, state, explore=True, do_extra=False):
        """Random action selection."""
        a = self.env.action_space.sample()
        if self.P["inertia"] > 0:
            a += self.P["inertia"] * (self.a_last - a)
            self.a_last = a
        return a, {}