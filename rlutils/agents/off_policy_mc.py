from ._generic import Agent

import numpy as np


class OffPolicyMCAgent(Agent):
    def __init__(self, env, hyperparameters): 
        """
        DESCRIPTION
        """
        Agent.__init__(self, env, hyperparameters)
        self.action_shape = self.env.action_space.shape[:-1]
        self.Q = np.zeros(tuple(list(self.env.observation_space.shape) + list(self.action_shape)))
        self.C = self.Q.copy()
        self.pi = np.zeros(tuple(list(self.env.observation_space.shape) + [len(self.action_shape)]), dtype=int)
        self.exploratory_action_prob = (1 - self.P["epsilon"]) / np.prod(self.action_shape)
        self.ep_transitions = []

    def act_behaviour(self, state, explore=True, do_extra=False): 
        if np.random.rand() > self.P["epsilon"]:
            idx = tuple(np.random.randint(n) for n in self.action_shape)
        else:
            idx = tuple(self.pi[tuple(state)])
        return self.env.action_space[idx], {}

    def act_target(self, state, explore=True, do_extra=False):
        idx = tuple(self.pi[tuple(state)])
        return self.env.action_space[idx], {}

    def update_on_episode(self):
        """xxx"""
        g, w = 0, 1
        for s, a, r in reversed(self.ep_transitions):
            sa = tuple(list(s) + list(a)); s = tuple(s) # Tuples for indexing.
            g = r + (self.P["gamma"] * g)
            self.C[sa] += w
            self.Q[sa] += (w / self.C[sa]) * (g - self.Q[sa])
            greedy_action_last = self.env.action_space[tuple(self.pi[s])]
            self.pi[s] = np.unravel_index(self.Q[s].argmax(), self.Q[s].shape)
            if any(a != self.env.action_space[tuple(self.pi[s])]): break # Break if action taken is not greedy.
            # Compute b(a | s)
            if any(a != greedy_action_last): b_a_s = self.exploratory_action_prob
            else: b_a_s = self.P["epsilon"] + self.exploratory_action_prob
            w /= b_a_s