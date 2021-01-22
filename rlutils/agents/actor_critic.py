from ..common.networks import SequentialNetwork

import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F


DEFAULT_HYPERPARAMETERS = {
    "lr_pi": 1e-4,
    "lr_V": 1e-3,
    "gamma": 0.99,
}


class ActorCriticAgent:
    def __init__(self, 
                 state_shape, 
                 num_actions,
                 hyperparameters=DEFAULT_HYPERPARAMETERS
                 ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.P = hyperparameters 
        self.eps = np.finfo(np.float32).eps.item() # Small float used to prevent div/0 errors.
        # Create pi and V networks.
        if len(state_shape) > 1: raise NotImplementedError()
            # preset_pi, preset_V = "CartPolePi_Pixels", "CartPoleV_Pixels"
        else:
            net_code = [(state_shape[0], 64), "R", (64, 128), "R"]
            net_code_pi = net_code + [(128, num_actions), "S"]
            net_code_V = net_code + [(128, 1)] 
        self.pi = SequentialNetwork(code=net_code_pi, lr=self.P["lr_pi"]).to(self.device)
        self.V = SequentialNetwork(code=net_code_V, lr=self.P["lr_V"]).to(self.device)
        # Tracking variables.
        self.last_s_l_v = None # State, log prob action and value.
        self.ep_losses = []

    def act(self, state, explore=True):
        """Probabilistic action selection."""
        action_probs, value = self.pi(state), self.V(state)
        dist = Categorical(action_probs) # Categorical action distribution.
        action = dist.sample()
        self.last_s_l_v = (state, dist.log_prob(action), value[0])
        return action, {"pi": action_probs.detach().numpy(), "V": value[0].item()}

    def update_on_transition(self, next_state, reward):
        """Use the latest transition to update the policy and value network parameters."""
        assert self.last_s_l_v is not None, \
        "Need to store prediction on each timestep. Ensure using agent.act()."        
        state, log_prob, value = self.last_s_l_v
        # Get value prediction for next state.
        if next_state is not None: next_value = self.V(next_state)[0]
        else: next_value = 0 # Handle terminal.
        # Calculate TD target and error.
        td_target = reward + (self.P["gamma"] * next_value)
        td_error = (td_target - value).item() 
        # Update value in the direction of TD error using MSE loss.
        value_loss = F.mse_loss(value, td_target)
        self.V.optimise(value_loss)
        # Update policy in the direction of log_prob(a) * TD error.
        policy_loss = -log_prob * td_error
        self.pi.optimise(policy_loss)
        return policy_loss.item(), value_loss.item()

    def per_timestep(self, state, action, reward, next_state):
        """Operations to perform on each timestep during training."""
        self.ep_losses.append(self.update_on_transition(next_state, torch.tensor([reward], device=self.device)))
        self.last_s_l_v = None 

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        policy_loss, value_loss = np.mean(self.ep_losses, axis=0)
        del self.ep_losses[:]
        return {"logs":{"policy_loss": policy_loss, "value_loss": value_loss}}