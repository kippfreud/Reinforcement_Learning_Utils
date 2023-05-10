from rlutils.common.deployment import train
from rlutils.common.env_wrappers import NormaliseActionWrapper, CustomRewardWrapper

import gym
import torch

train_parameters = {
    # * = Not used by StableBaselinesAgent.
    "project_name": "lunar_lander",
    "env": "LunarLanderContinuous-v2",
    "model": "ddpg",
    "num_episodes": 1000, # *
    "max_timesteps_per_episode": 500, # *
    "wandb_monitor": False, # *
    "render_freq": 1,
    "save_video": False,
    "save_final_agent": False,
}

# ===================================
# EXPERIMENT WITH CUSTOM REWARD FUNCTIONS
from custom_reward_experiment import R
# ===================================

# Make environment with wrappers.
env = CustomRewardWrapper(
      NormaliseActionWrapper(
      gym.make(train_parameters["env"])), R=R)
# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Make DdpgAgent.
if train_parameters["model"] == "ddpg":
    agent_parameters = {
        "replay_capacity": 50000,
        "batch_size": 128,
        "lr_pi": 1e-4,
        "lr_Q": 1e-3,
        "gamma": 0.99,
        "tau": 1e-2,
        "noise_params": (0., 0.15, 0.5, 0.01, 300000)
    }
    from rlutils.agents.ddpg import *
    agent = DdpgAgent(env.observation_space.shape, env.action_space, agent_parameters, device)

# Make StableBaselinesAgent.
elif train_parameters["model"] == "stable_baselines":
    agent_parameters = {
        "model_class": "sac",
        "verbose": True
    }
    del train_parameters["num_episodes"], train_parameters["max_timesteps_per_episode"]
    train_parameters["sb_parameters"] = {
        "total_timesteps": int(5e5)
    }
    from rlutils.agents.stable_baselines import *
    agent = StableBaselinesAgent(env, agent_parameters)

train(agent, env, train_parameters)