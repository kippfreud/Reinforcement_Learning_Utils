from rlutils.common.deployment import train
from rlutils.common.env_wrappers import NormalizedEnv

import gym

train_parameters = {
    "project_name": "lunar_lander",
    "env": "LunarLanderContinuous-v2",
    "model": "ddpg",
    "num_episodes": 1000,
    "max_timesteps_per_episode": 500,
    # "from_pixels": False,
    "wandb_monitor": False,
    "render_freq": 0
    "save_video":False
    "save_final_agent": False,
}

# Make environment.
env = NormalizedEnv(gym.make(train_parameters["env"]))

# Make DqnAgent.
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
    agent = DdpgAgent(env.observation_space.shape, env.action_space, agent_parameters)

run_name = train(agent, env, train_parameters, None)