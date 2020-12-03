from rlutils.common.training import train
import gym
from joblib import dump

train_parameters = {
    "project_name": "pendulum",
    "env": "Pendulum-v0",
    "model": "ddpg",
    "num_episodes": 2000,
    "max_timesteps_per_episode": 500,
    "from_pixels": False,
    "wandb_monitor": False,
    "render": False
}

# Make environment.
env = gym.make(train_parameters["env"]).unwrapped

# Make DdpgAgent.
if train_parameters["model"] == "ddpg":
    from rlutils.agents.actor_critic import *
    agent = ActorCriticAgent( env.observation_space.shape, env.action_space)


run_name = train(agent, env, train_parameters, None)
if not run_name: run_name = "unnamed_run"
# dump(agent, f"{run_name}.joblib") # Save agent using wandb run name.