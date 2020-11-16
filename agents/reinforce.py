import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F


DEFAULT_HYPERPARAMETERS = {
    "lr": 1e-2,
    "gamma": 0.99,
    "baseline": "Z",
}


class ReinforceAgent:
    def __init__(self, 
                 state_shape, 
                 num_actions,
                 hyperparameters=None
                 ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hyperparameters is not None:
            self.P = hyperparameters # Store hyperparameter dictionary.
        else:
            self.P = DEFAULT_HYPERPARAMETERS # Adopt defaults.
        self.have_value_net = self.P["baseline"] in ("adv")
        if self.P["baseline"] == "adv":
            # Create multi-headed policy and value network.
            if len(state_shape) > 1: raise NotImplementedError()
            else: preset = "CartPoleReinforceVectorWithBaseline"
            from common.networks import MultiHeadedNetwork
            self.net = MultiHeadedNetwork(preset=preset, state_shape=state_shape, num_actions=num_actions).to(self.device)        
        else:     
            # Create policy network only.
            if len(state_shape) > 1: preset = "CartPoleReinforcePixels"
            else: preset = "CartPoleReinforceVector"
            from common.networks import SequentialNetwork
            self.net = SequentialNetwork(preset=preset, state_shape=state_shape, num_actions=num_actions).to(self.device)
        self.optimiser = optim.Adam(self.net.parameters(), lr=self.P["lr"])
        self.ep_predictions = [] # Log prob actions (and value).
        self.ep_rewards = []
        self.eps = np.finfo(np.float32).eps.item() # Small float used to prevent div/0 errors.

    def act(self, state):
        """Probabilistic action selection."""
        if self.have_value_net: action_probs, value = self.net(state)
        else: action_probs = self.net(state)
        dist = Categorical(action_probs) # Categorical action distribution.
        action = dist.sample()
        if self.have_value_net: self.ep_predictions.append((dist.log_prob(action), value[0]))
        else: self.ep_predictions.append(dist.log_prob(action))
        return action

    def update_on_episode(self):
        """Use the latest episode of experience update the pi network parameters."""
        assert len(self.ep_predictions) == len(self.ep_rewards), \
        "Need to store rewards using agent.rewards.append(reward) on each timestep."        
        g, returns = 0, []
        for r in self.ep_rewards[::-1]: # Loop backwards through the episode.
            g = r + (self.P["gamma"] * g)
            returns.insert(0, g)
        returns = torch.tensor(returns, device=self.device)
        # Zero gradient buffers of all parameters.
        self.optimiser.zero_grad() 
        if self.have_value_net: 
            log_probs, values = (torch.cat(x) for x in zip(*self.ep_predictions))
            # Update value in the direction of advantage.
            value_loss = F.smooth_l1_loss(values, returns).sum()
            value_loss.backward(retain_graph=True) # Need to retain when going backward() twice through the same graph.
        else: log_probs, values, value_loss = torch.cat(self.ep_predictions), None, 0
        # Update policy in the direction of delta * log_prob(a).
        policy_loss = (-log_probs * self.baseline(returns, values)).sum()
        policy_loss.backward() 
        # Run backprop using autograd engine and update parameters of self.net.
        self.optimiser.step()
        # Empty the memory.
        del self.ep_predictions[:]; del self.ep_rewards[:] 
        return policy_loss, value_loss

    def baseline(self, returns, values):
        """Apply baselining to returns to improve update stability."""
        b = self.P["baseline"]
        if b == "off": return returns # No baselining.
        if b == "Z": 
            # Z-normalisation.
            return (returns - returns.mean()) / (returns.std() + self.eps)
        if b == "adv":
            # Advantage (subtract value prediction).
            return returns - values