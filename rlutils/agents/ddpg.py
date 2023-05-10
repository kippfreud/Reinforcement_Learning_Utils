from ..common.networks import SequentialNetwork
from ..common.memory import ReplayMemory
from ..common.exploration import OUNoise

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F 


DEFAULT_HYPERPARAMETERS = {
    "replay_capacity": 50000,
    "batch_size": 128,
    "lr_pi": 1e-4,
    "lr_Q": 1e-3,
    "gamma": 0.99,
    "tau": 1e-2,
    "noise_params": (0., 0.15, 0.1, 0.1, 100000)
}


class DdpgAgent:
    def __init__(self, 
                 state_shape,
                 action_space, 
                 hyperparameters=DEFAULT_HYPERPARAMETERS,
                 device=None,
                 ):
        self.device = device
        if self.device is None:
            print("WARNING: Device not specified, defaulting to best available device.")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.P = hyperparameters
        # Create pi and Q networks.
        if len(state_shape) > 1: raise NotImplementedError()
        else: preset_pi, preset_Q = "PendulumPi_Vector", "PendulumQ_Vector"
        num_actions = action_space.shape[0]
        self.pi = SequentialNetwork(preset=preset_pi, input_shape=state_shape, output_size=num_actions).to(self.device)
        self.optimiser_pi = optim.Adam(self.pi.parameters(), lr=self.P["lr_pi"])
        self.pi_target = SequentialNetwork(preset=preset_pi, input_shape=state_shape, output_size=num_actions).to(self.device)
        self.pi_target.load_state_dict(self.pi.state_dict()) # Clone.
        self.pi_target.eval() # Turn off training mode for target net.
        # Action is an *input* to the Q network here.
        self.Q = SequentialNetwork(preset=preset_Q, input_shape=(state_shape[0]+num_actions,), output_size=1).to(self.device)
        self.optimiser_Q = optim.Adam(self.Q.parameters(), lr=self.P["lr_Q"])
        self.Q_target = SequentialNetwork(preset=preset_Q, input_shape=(state_shape[0]+num_actions,), output_size=1).to(self.device)
        self.Q_target.load_state_dict(self.Q.state_dict()) 
        self.Q_target.eval()
        # Create replay memory.
        self.memory = ReplayMemory(self.P["replay_capacity"]) 
        # Create noise process for exploration.
        self.noise = OUNoise(action_space, *self.P["noise_params"])
        # Tracking variables.   
        self.total_t = 0 # Used for noise decay.
        self.ep_losses = []  
    
    def act(self, state, explore=True):
        """Deterministic action selection plus additive noise."""
        state = state.to(self.device)
        action = self.pi(state).cpu().detach().numpy()[0]
        if explore: return self.noise.get_action(action, self.total_t), {}
        else: return action, {"Qa":None}

    def update_on_batch(self):
        """Use a random batch from the replay memory to update the pi and Q network parameters."""
        if len(self.memory) < self.P["batch_size"]: return
        # Sample a batch and transpose it (see https://stackoverflow.com/a/19343/3343043).
        batch = self.memory.element(*zip(*self.memory.sample(self.P["batch_size"])))
        states = torch.cat(batch.state).to(self.device)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        # Compute Q(s, a) by running each concatenated s, a pair through self.Q.
        Q_values = self.Q(self.sa_concat(states, actions)).squeeze()
        # Identify nonterminal states (note that replay memory elements are initialised to None).
        nonterminal_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        nonterminal_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(self.device)
        # Use target Q network to compute Q_target(s', a') for each nonterminal next state.    
        # a' is chosen using the target pi network.
        next_Q_values = torch.zeros(self.P["batch_size"], device=self.device)
        nonterminal_next_actions = self.pi_target(nonterminal_next_states)
        next_Q_values[nonterminal_mask] = self.Q_target(self.sa_concat(nonterminal_next_states, nonterminal_next_actions.detach())).squeeze()
        # Compute target = reward + discounted Q_target(s', a').
        Q_targets = rewards + (self.P["gamma"] * next_Q_values)
        # Zero gradient buffers of all parameters.
        self.optimiser_pi.zero_grad(); self.optimiser_Q.zero_grad()
        # Update value in the direction of TD error using MSE loss. 
        # value_loss = F.mse_loss(Q_values, Q_targets)
        value_loss = F.smooth_l1_loss(Q_values, Q_targets)
        value_loss.backward() 
        for param in self.Q.parameters():
            param.grad.data.clamp_(-1, 1) # Implement gradient clipping.
        self.optimiser_Q.step()
        # Update policy in the direction of increasing value according to self.Q (the policy gradient).
        policy_loss = -self.Q(self.sa_concat(states, self.pi(states))).mean()
        policy_loss.backward()
        self.optimiser_pi.step()
        # Perform soft updates on targets.
        for target_param, param in zip(self.pi_target.parameters(), self.pi.parameters()):
            target_param.data.copy_(param.data * self.P["tau"] + target_param.data * (1.0 - self.P["tau"]))
        for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
            target_param.data.copy_(param.data * self.P["tau"] + target_param.data * (1.0 - self.P["tau"]))
        return policy_loss.item(), value_loss.item()

    def sa_concat(_, states, actions):
        """Concatenate states and actions into a single input vector for Q networks."""
        return torch.cat([states, actions.float()], 1).float()

    def per_timestep(self, state, action, reward, next_state):
        """Operations to perform on each timestep during training."""
        self.memory.add(state, torch.tensor([action], device=self.device), torch.tensor([reward], device=self.device), next_state)
        losses = self.update_on_batch()
        if losses: self.ep_losses.append(losses)
        self.total_t += 1

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        if self.ep_losses: mean_policy_loss, mean_value_loss = np.mean(self.ep_losses, axis=0)
        else: mean_policy_loss, mean_value_loss = 0., 0.
        del self.ep_losses[:]; #self.noise.reset(); self.total_t = 0
        return {"logs":{"sigma": self.noise.sigma, "policy_loss": mean_policy_loss, "value_loss": mean_value_loss}}