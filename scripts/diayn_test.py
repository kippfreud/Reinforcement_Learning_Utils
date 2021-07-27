import gym
import rlutils
from rlutils.common.env_wrappers import NormaliseActionWrapper

# Make environment.
env = NormaliseActionWrapper(gym.make("Pendulum-v0"))

agent_parameters = {
    "lr_pi": 3e-4,       
    "lr_V": 1e-3,
    "num_steps_per_update": 80, # Number of gradient steps per update.
    "baseline": "Z", # Baselining method: either "off", "Z" or "adv".
    "epsilon": 0.2, # Clip ratio for policy update.
    "noise_params": ("norm", 0.6, 0.1, 0.05, 20000), # Initial std, final std, decay rate, decay freq (timesteps).
}

agent = rlutils.make("diayn", env, agent_parameters)