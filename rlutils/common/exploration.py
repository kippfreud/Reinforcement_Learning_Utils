import numpy as np

# TODO: Work with Torch tensors rather than NumPy arrays.

class OUNoise(object):
    """
    Time-correlated noise for continuous actions using the Ornstein-Ulhenbeck process.
    Taken from https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
    """
    def __init__(self, action_space, mu, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=1000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma 
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = -1 # action_space.low # NOTE: Just use [-1,1] if applying NormaliseActionWrapper to env.
        self.high         = 1 # action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx

    def decay(self, k):
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, k / self.decay_period)
    
    def get_action(self, action):
        self.evolve_state()
        return np.clip(action + self.state, self.low, self.high)


class UniformNoise(object):
    """
    Weighted averaging with random uniform noise. Use sigma as parameter for consistency with above.
    """
    def __init__(self, action_space, max_sigma=1, min_sigma=0, decay_period=1000):
        self.action_space = action_space
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period

    def decay(self, k):
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, k / self.decay_period)

    def get_action(self, action):
        action_rand = self.action_space.sample()
        return (action * (1-self.sigma)) + (action_rand * self.sigma)