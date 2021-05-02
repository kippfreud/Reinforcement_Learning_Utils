from collections import namedtuple
import random

# TODO: element_with_done should be default for all algorithms.

# Structure of a memory element.
element = namedtuple('element', ('state', 'action', 'reward', 'next_state'))
element_with_done = namedtuple('element', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity, include_done=False):
        self.capacity = capacity
        self.element = element_with_done if include_done else element 
        self.memory = []
        self.position = 0

    def __len__(self): return len(self.memory) # Length is length of memory list.

    def add(self, *args):
        """Save a transition."""
        # Extend memory if capacity not yet reached.
        if len(self.memory) < self.capacity: self.memory.append(None) 
        # Overwrite current entry at this position.
        self.memory[self.position] = self.element(*args)
        # Increment position, cycling back to the beginning if needed.
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Retrieve a random sample of transitions."""
        return random.sample(self.memory, batch_size)