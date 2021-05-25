"""
DESCRIPTION
"""

from ._generic import Agent
from ..common.networks import SequentialNetwork

import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F


class ReinforceAgent(Agent):
    def __init__(self, env, hyperparameters):
        Agent.__init__(self, env, hyperparameters)
        # Create pi network (and V if using advantage baselining).
        if len(self.env.observation_space.shape) > 1: raise NotImplementedError()
        self.pi = SequentialNetwork(code=self.P["net_pi"], input_shape=self.env.observation_space.shape[0], output_size=self.env.action_space.n, lr=self.P["lr_pi"]).to(self.device)
        if self.P["baseline"] == "adv":
            self.V = SequentialNetwork(code=self.P["net_V"], input_shape=self.env.observation_space.shape[0], output_size=1, lr=self.P["lr_V"]).to(self.device)
        else: self.V = None
        # Small float used to prevent div/0 errors.
        self.eps = np.finfo(np.float32).eps.item() 
        # Tracking variables.
        self.ep_predictions = [] # Log prob actions (and value).
        self.ep_rewards = []

    def act(self, state, explore=True, do_extra=False):
        """Probabilistic action selection."""
        state = state.to(self.device)
        if self.V is not None: action_probs, value = self.pi(state), self.V(state)
        else: action_probs = self.pi(state)
        dist = Categorical(action_probs) # Categorical action distribution.
        action = dist.sample()
        extra = {"pi": action_probs.cpu().detach().numpy()} if do_extra else {}
        if self.V is not None: 
            self.ep_predictions.append((dist.log_prob(action), value[0]))
            if do_extra: extra["V"] = value[0].item()
        else: 
            self.ep_predictions.append(dist.log_prob(action))
        return action.item(), extra

    def update_on_episode(self):
        """Use the latest episode of experience to update the policy (and value) network parameters."""
        assert len(self.ep_predictions) == len(self.ep_rewards), \
        "Need to store rewards using agent.rewards.append(reward) on each timestep."        
        g, returns = 0, []
        for reward in self.ep_rewards[::-1]: # Loop backwards through the episode.
            g = reward + (self.P["gamma"] * g)
            returns.insert(0, g)
        returns = torch.tensor(returns, device=self.device)
        if self.V is not None: 
            # Update value in the direction of advantage using Huber loss.
            log_probs, values = (torch.cat(x) for x in zip(*self.ep_predictions))
            value_loss = F.smooth_l1_loss(values, returns)
            self.V.optimise(value_loss)
        else: log_probs, values, value_loss = torch.cat(self.ep_predictions), None, 0
        # Update policy in the direction of log_prob(a) * delta.
        policy_loss = (-log_probs * self.baseline(returns, values)).sum()
        self.pi.optimise(policy_loss)
        return policy_loss.item(), value_loss.item()

    def baseline(self, returns, values):
        """Apply baselining to returns to improve update stability."""
        if self.P["baseline"] == "off":  return returns # No baselining.
        elif self.P["baseline"] == "Z":   return (returns - returns.mean()) / (returns.std() + self.eps) # Z-normalisation.
        elif self.P["baseline"] == "adv": return (returns - values).detach() # Advantage (subtract value prediction).
        else: raise NotImplementedError("Baseline method not recognised.")

    def per_timestep(self, state, action, reward, next_state, done):
        """Operations to perform on each timestep during training."""
        self.ep_rewards.append(reward)

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        policy_loss, value_loss = self.update_on_episode() 
        del self.ep_predictions[:]; del self.ep_rewards[:] 
        return {"logs":{"policy_loss": policy_loss, "value_loss": value_loss}}