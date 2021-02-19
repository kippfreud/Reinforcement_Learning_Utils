from rlutils.common.deployment import train
from rlutils.common.env_wrappers import NormaliseActionWrapper

import gym

train_parameters = {
    "project_name": "pendulum",
    "env": "Pendulum-v0",
    "model": "sac",
    "num_episodes": 100,
    "max_timesteps_per_episode": 500,
    # "from_pixels": False,
    "wandb_monitor": True,
    "render_freq": 0,
    "save_video": False,
    "save_final_agent": False,
}

# Make environment.
env = NormaliseActionWrapper(gym.make(train_parameters["env"]))

# Make DdpgAgent.
if train_parameters["model"] in ("ddpg","td3"):
    agent_parameters = {
        "replay_capacity": 10000,
        "batch_size": 128,
        "lr_pi": 1e-4,
        "lr_Q": 1e-3,
        "gamma": 0.99,
        "tau": 0.01,
        "noise_params": (0., 0.15, 0.5, 0.01, 50000),
        "td3": train_parameters["model"] == "td3",
        "td3_noise_std": 0.2,
        "td3_noise_clip": 0.5,
        "td3_policy_freq": 2
    }
    from rlutils.agents.ddpg import *
    agent = DdpgAgent(env.observation_space.shape, env.action_space, agent_parameters)

# Make SacAgent.
if train_parameters["model"] == "sac":
    agent_parameters = {
        "replay_capacity": 10000,
        "batch_size": 64,
        "lr_pi": 5e-4,
        "lr_Q": 1e-3,
        "gamma": 0.99,
        "alpha": 0.2,
        "tau": 0.005,
    }
    from rlutils.agents.sac import *
    agent = SacAgent(env.observation_space.shape, env.action_space.shape[0], agent_parameters)

run_name = train(agent, env, train_parameters)