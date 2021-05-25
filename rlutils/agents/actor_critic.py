"""
DESCRIPTION
"""

from ._generic import Agent
from ..common.networks import SequentialNetwork

import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F


class ActorCriticAgent(Agent):
    def __init__(self, env, hyperparameters):
        Agent.__init__(self, env, hyperparameters)
        self.eps = np.finfo(np.float32).eps.item() # Small float used to prevent div/0 errors.
        # Create pi and V networks.
        if len(self.env.observation_space.shape) > 1: raise NotImplementedError()
        self.pi = SequentialNetwork(code=self.P["net_pi"], input_shape=self.env.observation_space.shape[0], output_size=self.env.action_space.n, lr=self.P["lr_pi"]).to(self.device)
        self.V = SequentialNetwork(code=self.P["net_V"], input_shape=self.env.observation_space.shape[0], output_size=1, lr=self.P["lr_V"]).to(self.device)
        # Tracking variables.
        self.last_l_v = None # Log probability of action and value for previous timestep.
        self.ep_losses = []

    def act(self, state, explore=True, do_extra=False):
        """Probabilistic action selection."""
        action_probs, value = self.pi(state), self.V(state)
        dist = Categorical(action_probs) 
        action = dist.sample()
        self.last_l_v = (dist.log_prob(action), value[0])
        return action.item(), {"pi": action_probs.cpu().detach().numpy()[0], "V": value[0].item()} if do_extra else {}

    def update_on_transition(self, next_state, reward, done):
        """Use the latest transition to update the policy and value network parameters."""
        assert self.last_l_v is not None, "Need to store prediction on each timestep. Ensure using agent.act()."        
        log_prob, value = self.last_l_v
        self.last_l_v = None # To prevent using again.
        # Get value prediction for next state.
        if done: next_value = 0 # Handle terminal.
        else: next_value = self.V(next_state)[0]
        # Calculate TD target and error.
        td_target = reward + (self.P["gamma"] * next_value)
        td_error = (td_target - value).item() 
        # Update value in the direction of TD error using MSE loss (NOTE: seems to outperform Huber on CartPole!)
        value_loss = F.mse_loss(value, td_target)
        self.V.optimise(value_loss)
        # Update policy in the direction of log_prob(a) * TD error.
        policy_loss = -log_prob * td_error
        self.pi.optimise(policy_loss)
        return policy_loss.item(), value_loss.item()

    def per_timestep(self, state, action, reward, next_state, done):
        """Operations to perform on each timestep during training."""
        self.ep_losses.append(self.update_on_transition(next_state, torch.tensor([reward], device=self.device, dtype=torch.float), done))

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        policy_loss, value_loss = np.mean(self.ep_losses, axis=0)
        del self.ep_losses[:]
        return {"logs":{"policy_loss": policy_loss, "value_loss": value_loss}}