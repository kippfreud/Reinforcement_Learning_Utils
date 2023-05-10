from ..common.networks import SequentialNetwork

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F


DEFAULT_HYPERPARAMETERS = {
    "lr_pi": 1e-4,
    "lr_V": 1e-3,
    "gamma": 0.99,
    "baseline": "adv",
}


class ReinforceAgent:
    def __init__(self, 
                 state_shape, 
                 num_actions,
                 hyperparameters=DEFAULT_HYPERPARAMETERS,
                 device=None
                 ):
        self.device = device
        if self.device is None:
            print("WARNING: Device not specified, defaulting to best available device.")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.P = hyperparameters 
        self.eps = np.finfo(np.float32).eps.item() # Small float used to prevent div/0 errors.
        # Create pi network (and V if using advantage baselining).
        if len(state_shape) > 1: preset_pi, preset_V = "CartPolePi_Pixels", "CartPoleV_Pixels"
        else: preset_pi, preset_V = "CartPolePi_Vector", "CartPoleV_Vector"
        self.pi = SequentialNetwork(preset=preset_pi, input_shape=state_shape, output_size=num_actions).to(self.device)
        self.optimiser_pi = optim.Adam(self.pi.parameters(), lr=self.P["lr_pi"])
        if self.P["baseline"] == "adv":
            self.V = SequentialNetwork(preset=preset_V, input_shape=state_shape, output_size=1).to(self.device)
            self.optimiser_V = optim.Adam(self.V.parameters(), lr=self.P["lr_V"])
        else: self.V = None
        # Tracking variables.
        self.ep_predictions = [] # Log prob actions (and value).
        self.ep_rewards = []

    def act(self, state, explore=True):
        """Probabilistic action selection."""
        state = state.to(self.device)
        if self.V is not None: action_probs, value = self.pi(state), self.V(state)
        else: action_probs = self.pi(state)
        dist = Categorical(action_probs) # Categorical action distribution.
        action = dist.sample()
        extra = {"pi": action_probs.cpu().detach().numpy()}
        if self.V is not None: 
            self.ep_predictions.append((dist.log_prob(action), value[0]))
            extra["V"] = value[0].item()
        else: 
            self.ep_predictions.append(dist.log_prob(action))
        return action, extra

    def update_on_episode(self):
        """Use the latest episode of experience to update the policy (and value) network parameters."""
        assert len(self.ep_predictions) == len(self.ep_rewards), \
        "Need to store rewards using agent.rewards.append(reward) on each timestep."        
        g, returns = 0, []
        for reward in self.ep_rewards[::-1]: # Loop backwards through the episode.
            g = reward + (self.P["gamma"] * g)
            returns.insert(0, g)
        returns = torch.tensor(returns, device=self.device)
        # Zero gradient buffers of all parameters.
        if self.V is not None: 
            # Update value in the direction of advantage using MSE loss.
            self.optimiser_V.zero_grad()
            log_probs, values = (torch.cat(x) for x in zip(*self.ep_predictions))
            value_loss = F.mse_loss(values, returns)
            value_loss.backward()
            # for param in self.V.parameters(): param.grad.data.clamp_(-1, 1) 
            self.optimiser_V.step()
        else: log_probs, values, value_loss = torch.cat(self.ep_predictions), None, 0
        # Update policy in the direction of log_prob(a) * delta.
        self.optimiser_pi.zero_grad()
        policy_loss = (-log_probs * self.baseline(returns, values)).sum()
        policy_loss.backward() 
        self.optimiser_pi.step()
        return policy_loss.item(), value_loss.item()

    def baseline(self, returns, values):
        """Apply baselining to returns to improve update stability."""
        b = self.P["baseline"]
        if b == "off": return returns # No baselining.
        if b == "Z": 
            # Z-normalisation.
            return (returns - returns.mean()) / (returns.std() + self.eps)
        if b == "adv":
            # Advantage (subtract value prediction).
            return (returns - values).detach()
        else: raise NotImplementedError("Baseline method not recognised.")

    def per_timestep(self, state, action, reward, next_state):
        """Operations to perform on each timestep during training."""
        self.ep_rewards.append(reward)

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        policy_loss, value_loss = self.update_on_episode() 
        del self.ep_predictions[:]; del self.ep_rewards[:] 
        return {"logs":{"policy_loss": policy_loss, "value_loss": value_loss}}