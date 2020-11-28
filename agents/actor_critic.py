import numpy as np
import torch
import torch.optim as optim
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
                 hyperparameters=None
                 ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hyperparameters is not None:
            self.P = hyperparameters # Store hyperparameter dictionary.
        else:
            self.P = DEFAULT_HYPERPARAMETERS # Adopt defaults.
        if len(state_shape) > 1: preset_pi, preset_V = "CartPolePi_Pixels", "CartPoleV_Pixels"
        else: preset_pi, preset_V = "CartPolePi_Vector", "CartPoleV_Vector"
        from common.networks import SequentialNetwork
        self.pi = SequentialNetwork(preset=preset_pi, state_shape=state_shape, num_actions=num_actions).to(self.device)
        self.optimiser_pi = optim.Adam(self.pi.parameters(), lr=self.P["lr_pi"])
        self.V = SequentialNetwork(preset=preset_V, state_shape=state_shape, num_actions=num_actions).to(self.device)
        self.optimiser_V = optim.Adam(self.V.parameters(), lr=self.P["lr_V"])
        self.eps = np.finfo(np.float32).eps.item() # Small float used to prevent div/0 errors.
        # Tracking variables.
        self.last_s_l_v = None # State, log prob action and value.
        self.ep_losses = []

    def act(self, state):
        """Probabilistic action selection."""
        action_probs, value = self.pi(state), self.V(state)
        dist = Categorical(action_probs) # Categorical action distribution.
        action = dist.sample()
        self.last_s_l_v = (state, dist.log_prob(action), value[0])
        return action

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
        td_error = (td_target - value).item() # NOTE: Is this the "advantage"?
        # Zero gradient buffers of all parameters.
        self.optimiser_pi.zero_grad(); self.optimiser_V.zero_grad()
        # Update value in the direction of TD error.
        value_loss = F.mse_loss(value, td_target)
        value_loss.backward()
        # for param in self.V.parameters(): param.grad.data.clamp_(-1, 1) 
        self.optimiser_V.step()
        # Update policy in the direction of log_prob(a) * TD error.
        policy_loss = -log_prob * td_error
        policy_loss.backward() 
        self.optimiser_pi.step()
        
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