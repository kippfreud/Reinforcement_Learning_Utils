import gym


class NormaliseActionWrapper(gym.ActionWrapper):
    """
    For environments with continuous action spaces.
    Maps normalised actions in [0,1] into the range used by the environment.
    """
    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)


class CustomRewardWrapper(gym.Wrapper): 
    """
    Enables implementation of a custom reward function.
    NOTE: uses next_state not current one!
    """
    def __init__(self, env, R):
        super().__init__(env)
        self.env = env
        self.R = R
        
    def step(self, action):
        next_state, _, done, info = self.env.step(action)
        reward, done = self.R(next_state, action, done)
        return next_state, reward, done, info