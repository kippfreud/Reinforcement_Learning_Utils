import gym
import rlutils
from rlutils.common.env_wrappers import NormaliseActionWrapper

train_parameters = {
    "project_name": "pendulum",
    "env": "Pendulum-v0",
    "agent": "ppo",
    "wandb_monitor": True,
    "num_episodes": 1000,
    "render_freq": 0,
}

# Make environment.
env = NormaliseActionWrapper(gym.make(train_parameters["env"]))

agent_parameters = {}
agent_parameters["ppo"] = {
    "lr_pi": 3e-4,       
    "lr_V": 1e-3,
    "num_steps_per_update": 80, # Number of gradient steps per update.
    "baseline": "Z", # Baselining method: either "off", "Z" or "adv".
    "epsilon": 0.2, # Clip ratio for policy update.
    "noise_params": ("norm", 0.6, 0.1, 0.05, 20000), # Initial std, final std, decay rate, decay freq (timesteps).
}

a = train_parameters["agent"]
agent = rlutils.make(a, env, agent_parameters[a])
print(agent)
_, rn = rlutils.train(agent, train_parameters)

