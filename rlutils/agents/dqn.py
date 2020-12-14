from ..common.networks import SequentialNetwork
from ..common.memory import ReplayMemory

import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F


DEFAULT_HYPERPARAMETERS = {
    "replay_capacity": 10000,
    "batch_size": 128,
    "lr_Q": 1e-3,
    "gamma": 0.99,
    "epsilon_start": 0.9,
    "epsilon_end": 0.05,
    "epsilon_decay": 10000,
    "updates_between_target_clone": 2000
}


class DqnAgent:
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
        # Create Q network.
        if len(state_shape) > 1: preset = "CartPoleQ_Pixels"
        else: preset = "CartPoleQ_Vector"
        self.Q = SequentialNetwork(preset=preset, input_shape=state_shape, output_size=num_actions).to(self.device)
        self.Q_target = SequentialNetwork(preset=preset, input_shape=state_shape, output_size=num_actions).to(self.device)
        self.Q_target.load_state_dict(self.Q.state_dict()) # Clone.
        self.Q_target.eval() # Turn off training mode for target net.
        self.num_actions = num_actions
        self.optimiser = optim.Adam(self.Q.parameters(), lr=self.P["lr_Q"])
        # Create replay memory.
        self.memory = ReplayMemory(self.P["replay_capacity"])
        # Tracking variables.
        self.epsilon = self.P["epsilon_start"]
        self.total_t = 0 # Used for epsilon decay.
        self.updates_since_target_clone = 0
        self.ep_losses = []

    def act(self, state, explore=True):
        """Epsilon-greedy action selection."""
        state = state.to(self.device)
        Q = self.Q(state)
        extra = {"Q": Q.cpu().detach().numpy()}
        if (not explore) or random.random() > self.epsilon:
            # Return action with highest Q value.
            return Q.max(1)[1].view(1, 1), extra
        else:
            # Return a random action.
            return torch.tensor([[random.randrange(self.num_actions)]], device=self.device, dtype=torch.long), extra

    def update_on_batch(self):
        """Use a random batch from the replay memory to update the Q network parameters."""
        if len(self.memory) < self.P["batch_size"]: return 
        # Sample a batch and transpose it (see https://stackoverflow.com/a/19343/3343043).
        batch = self.memory.element(*zip(*self.memory.sample(self.P["batch_size"])))
        states = torch.cat(batch.state).to(self.device)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        # Compute Q(s, a) by running each s through self.Q, then selecting the corresponding column.
        Q_values = self.Q(states).gather(1, actions)
        # Identify nonterminal states (note that replay memory elements are initialised to None).
        nonterminal_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        nonterminal_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(self.device)
        # Use target network to compute Q_target(s', a') for each nonterminal next state.
        # a' is chosen to be the maximising action from s'.
        next_Q_values = torch.zeros(self.P["batch_size"], device=self.device)
        next_Q_values[nonterminal_mask] = self.Q_target(nonterminal_next_states).max(1)[0].detach()
        # Compute target = reward + discounted Q_target(s', a').
        Q_targets = rewards + (self.P["gamma"] * next_Q_values)
        # Zero gradient buffers of all parameters.
        self.optimiser.zero_grad() 
        # Update value in the direction of TD error using Huber loss. 
        # See https://en.wikipedia.org/wiki/Huber_loss.
        loss = F.smooth_l1_loss(Q_values, Q_targets.unsqueeze(1))
        loss.backward() 
        for param in self.Q.parameters():
            param.grad.data.clamp_(-1, 1) # Implement gradient clipping.
        self.optimiser.step()
        # Periodically clone target.
        self.updates_since_target_clone += 1
        if self.updates_since_target_clone >= self.P["updates_between_target_clone"]:
            self.Q_target.load_state_dict(self.Q.state_dict())
            self.updates_since_target_clone = 0
        return loss.item()

    def per_timestep(self, state, action, reward, next_state):
        """Operations to perform on each timestep during training."""
        self.memory.add(state, action, torch.tensor([reward], device=self.device), next_state)
        loss = self.update_on_batch()
        if loss: self.ep_losses.append(loss)
        self.epsilon = self.P["epsilon_end"] + (self.P["epsilon_start"] - self.P["epsilon_end"]) * \
                       np.exp(-1 * self.total_t / self.P["epsilon_decay"])
        self.total_t += 1

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        if self.ep_losses: mean_loss = np.mean(self.ep_losses)
        else: mean_loss = 0.
        del self.ep_losses[:]
        return {"logs":{"epsilon": self.epsilon, "value_loss": mean_loss}}