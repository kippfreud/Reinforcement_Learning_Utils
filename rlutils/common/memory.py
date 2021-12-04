import torch
import numpy as np
from collections import namedtuple
import random

# Structure of a memory element.
element = namedtuple('element', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.element = element
        self.clear()

    def __len__(self): return len(self.memory) # Length is length of memory list.

    def clear(self): self.memory = []; self.position = 0

    def add(self, state, action, reward, next_state, done):
        """Save a transition."""
        # Extend memory if capacity not yet reached.
        if len(self.memory) < self.capacity: self.memory.append(None) 
        # Overwrite current entry at this position.
        if type(action) == int: action_dtype = torch.int64
        elif type(action) == np.ndarray: action_dtype = torch.float
        self.memory[self.position] = self.element(
            state, 
            torch.tensor([action], device=state.device, dtype=action_dtype),
            torch.tensor([reward], device=state.device, dtype=torch.float), 
            next_state,
            torch.tensor([done], device=state.device, dtype=torch.bool)
        )
        # Increment position, cycling back to the beginning if needed.
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, mode="uniform", keep_terminal_next=False):
        """Retrieve a random sample of transitions and refactor.
        See https://stackoverflow.com/a/19343/3343043."""
        if len(self) < batch_size: return None, None, None, None, None
        if mode == "uniform": batch = self._uniform(batch_size)
        if mode == "prioritised": batch = self._prioritised(batch_size)
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        next_states = torch.cat(batch.next_state)
        if keep_terminal_next: nonterminal_mask = None
        else: 
            nonterminal_mask = ~torch.cat(batch.done)
            next_states = next_states[nonterminal_mask]
        return states, actions, rewards, nonterminal_mask, next_states

    def _uniform(self, batch_size): return self.element(*zip(*random.sample(self.memory, batch_size)))
    
    def _prioritised(self, batch_size): raise NotImplementedError()


# TODO: Unify.
class PpoMemory:
    def __init__(self): self.clear()

    def clear(self):
        self.state = []
        self.action = []
        self.log_prob = []
        self.reward = []
        self.done = []