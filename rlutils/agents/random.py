import torch


DEFAULT_HYPERPARAMETERS = {
    "method": "uniform",
    "inertia": 0,
    "gamma": 0.99
}


class RandomAgent:
    def __init__(self, 
                 action_space, 
                 hyperparameters=DEFAULT_HYPERPARAMETERS
                 ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space = action_space
        self.P = hyperparameters
        self.a_last = self.action_space.sample()

    def act(self, state, explore=True):
        """Random action selection."""
        a = self.action_space.sample()
        if self.P["inertia"] > 0:
            a += self.P["inertia"] * (self.a_last - a)
            self.a_last = a
        return a, {}

    def per_timestep(self, state, action, reward, next_state): return
    def per_episode(self): return