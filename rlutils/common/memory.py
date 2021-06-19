from collections import namedtuple
import random

# Structure of a memory element.
element = namedtuple('element', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.element = element
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

# TODO: Unify.
class PpoMemory:
    def __init__(self):
        self.state = []
        self.action = []
        self.log_prob = []
        self.reward = []
        self.done = []

    def clear(self):
        del self.state[:]
        del self.action[:]
        del self.log_prob[:]
        del self.reward[:]
        del self.done[:]