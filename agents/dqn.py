from common.networks import SequentialNetwork

import random
import numpy as np
from collections import namedtuple
import torch
import torch.optim as optim
import torch.nn.functional as F


DEFAULT_HYPERPARAMETERS = {
    "replay_capacity": 10000,
    "batch_size": 128,
    "lr": 1e-3,
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
                 hyperparameters=None
                 ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hyperparameters is not None:
            self.P = hyperparameters # Store hyperparameter dictionary.
        else:
            self.P = DEFAULT_HYPERPARAMETERS # Adopt defaults.
        # Create Q network.
        if len(state_shape) > 1: preset = "CartPoleQ_Pixels"
        else: preset = "CartPoleQ_Vector"
        self.Q = SequentialNetwork(preset=preset, state_shape=state_shape, num_actions=num_actions).to(self.device)
        self.Q_target = SequentialNetwork(preset=preset, state_shape=state_shape, num_actions=num_actions).to(self.device)
        self.Q_target.load_state_dict(self.Q.state_dict()) # Clone.
        self.Q_target.eval() # Turn off training mode for target net.
        self.num_actions = num_actions
        self.optimiser = optim.Adam(self.Q.parameters(), lr=self.P["lr"])
        self.memory = ReplayMemory(self.P["replay_capacity"])
        self.epsilon = self.P["epsilon_start"]
        # Tracking variables.
        self.step_count = 0
        self.updates_since_target_clone = 0

    def act(self, state):
        """Epsilon-greedy action selection for during learning."""
        self.epsilon = self.P["epsilon_end"] + (self.P["epsilon_start"] - self.P["epsilon_end"]) * \
                       np.exp(-1 * self.step_count / self.P["epsilon_decay"])
        self.step_count += 1
        if random.random() > self.epsilon:
            # Return action with highest Q value.
            return self.Q(state).max(1)[1].view(1, 1)
        else:
            # Return a random action.
            return torch.tensor([[random.randrange(self.num_actions)]], device=self.device, dtype=torch.long)

    def update_on_batch(self):
        """Use a random batch from the replay memory to update the Q network parameters."""
        if len(self.memory) < self.P["batch_size"]: return 
        # Sample a batch and transpose it (see https://stackoverflow.com/a/19343/3343043).
        batch = Transition(*zip(*self.memory.sample(self.P["batch_size"])))
        # Separate out into tensors for states, actions and rewards.
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        # Compute Q(s, a) by running each s through self.Q, then selecting the corresponding column.
        Q_values = self.Q(states).gather(1, actions)
        # Identify nonterminal states (note that replay memory elements are initialised to None).
        nonterminal_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        nonterminal_next_states = torch.cat([s for s in batch.next_state if s is not None])
        # Use self.Q_target network to compute Q(s', a') for each next state.
        # a' is chosen to be the maximising action from s'.
        next_Q_values = torch.zeros(self.P["batch_size"], device=self.device)
        next_Q_values[nonterminal_mask] = self.Q_target(nonterminal_next_states).max(1)[0].detach()
        # Compute target = reward + discounted Q(s', a').
        Q_targets = rewards + (self.P["gamma"] * next_Q_values)
        # Zero gradient buffers of all parameters.
        self.optimiser.zero_grad() 
        # Compute loss. This is the Huber loss https://en.wikipedia.org/wiki/Huber_loss.
        loss = F.smooth_l1_loss(Q_values, Q_targets.unsqueeze(1))
        # Run backprop using autograd engine and update parameters.
        loss.backward() 
        for param in self.Q.parameters():
            param.grad.data.clamp_(-1, 1) # NOTE: Does this implement reward clipping?
        self.optimiser.step()
        self.updates_since_target_clone += 1
        # Periodically clone target.
        if self.updates_since_target_clone >= self.P["updates_between_target_clone"]:
            self.Q_target.load_state_dict(self.Q.state_dict())
            self.updates_since_target_clone = 0
        return loss


# Named tuple representing a transition in the environment.
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    """Class for storing transitions for learning."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self): return len(self.memory) # Length is length of memory list.

    def add(self, *args):
        """Saves a transition."""
        # Extend memory if capacity not yet reached.
        if len(self.memory) < self.capacity: self.memory.append(None) 
        # Overwrite current entry at this position.
        self.memory[self.position] = Transition(*args)
        # Increment position, cycling back to the beginning if needed.
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Retrieve a random sample of transitions."""
        return random.sample(self.memory, batch_size)