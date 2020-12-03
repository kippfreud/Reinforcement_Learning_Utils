import Reinforcement_Learning_Utils.rlutils as rl
import gym
from joblib import load

parameters = {
    "run_name": "cosmic-yogurt-123",
    # "run_name": "giddy-terrain-127",
    "state_dim_names": ["pos", "vel", "ang", "vel_ang"],
    "action_dim_names": ["action_idx"],
    "num_episodes": 2000,
    "max_timesteps_per_episode": 500,

    "render": True
}

# Load agent and environment into the observer class.
observer = rl.Observer(agent=load(f"{parameters['run_name']}.joblib"),
                       env=gym.make("CartPole-v1")
                       )

observer.observe(parameters)

"""
dataset = observer.add_continuous_actions(mapping)
dataset = observer.add_discounted_sums(dims=["reward"], gamma=0.99)
dataset = observer.add_custom_dims(function, dim_names=[""])
dataset = observer.add_derivatives(dims=["xxx"])
observer.to_csv()
"""