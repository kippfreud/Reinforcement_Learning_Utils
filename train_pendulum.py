from rlutils.common.deployment import train
from rlutils.common.env_wrappers import NormalizedEnv

import gym
from joblib import dump

train_parameters = {
    "project_name": "pendulum",
    "env": "Pendulum-v0",
    "model": "ddpg",
    "num_episodes": 100,
    "max_timesteps_per_episode": 500,
    # "from_pixels": False,
    "wandb_monitor": True,
    "save_trained_agent": False,
    "render": False
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
    }
    from rlutils.agents.ddpg import *
    agent = DdpgAgent(env.observation_space.shape, env.action_space, agent_parameters)

run_name = train(agent, env, train_parameters, None)
if train_parameters["save_trained_agent"]:
    if not run_name: run_name = "unnamed_run"
    dump(agent, f"{run_name}.joblib") # Save agent using wandb run name.