"""
DESCRIPTION
"""

from ._generic import Agent
import numpy as np


class StableBaselinesAgent(Agent):
    def __init__(self, env, hyperparameters):
        Agent.__init__(self, env, hyperparameters)

        if self.P["model_class"] == "dqn":
            from stable_baselines3 import DQN
            self.model = DQN('MlpPolicy', env, verbose=self.P["verbose"])
            self.model_class = DQN

        elif self.P["model_class"] == "a2c":
            from stable_baselines3 import A2C
            from stable_baselines3.a2c import MlpPolicy
            self.model = A2C(MlpPolicy, env, verbose=self.P["verbose"])
            self.model_class = A2C

        elif self.P["model_class"] == "ddpg":
            from stable_baselines3 import DDPG
            from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
            self.model = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=self.P["verbose"])        
            self.model_class = DDPG

        elif self.P["model_class"] == "td3":
            from stable_baselines3 import TD3
            from stable_baselines3.td3.policies import MlpPolicy
            from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
            self.model = TD3(MlpPolicy, env, action_noise=action_noise, verbose=self.P["verbose"])        
            self.model_class = TD3

        elif self.P["model_class"] == "ppo":
            from stable_baselines3 import PPO
            from stable_baselines3.ppo import MlpPolicy
            self.model = PPO(MlpPolicy, env, verbose=self.P["verbose"])        
            self.model_class = PPO

        elif self.P["model_class"] == "sac":
            from stable_baselines3 import SAC
            from stable_baselines3.sac import MlpPolicy
            self.model = SAC(MlpPolicy, env, verbose=self.P["verbose"])
            self.model_class = SAC

        else: raise NotImplementedError()

    def act(self, state, explore=True, do_extra=False): 
        action, _ = self.model.predict(state[0].numpy(), deterministic=(not explore))
        return action, {}

    # Defer to inbuilt methods from Stable Baselines.
    def train(self, parameters): self.model.learn(**parameters)
    def save(self, path): self.model.save(path)
    def load(self, path): self.model = self.model_class.load(path)

    # The regular learning methods shouldn't be used.
    def per_timestep(self, state, action, reward, next_state): raise NotImplementedError("Use agent.train to train!")
    def per_episode(self): raise NotImplementedError("Use agent.train to train!")