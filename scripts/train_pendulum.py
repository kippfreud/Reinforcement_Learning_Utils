import gym
import rlutils
from rlutils.common.env_wrappers import NormaliseActionWrapper

train_parameters = {
    "project_name": "pendulum",
    "env": "Pendulum-v0",
    "agent": "steve",
    "num_episodes": 100,
    "episode_time_limit": 200,
    "from_pixels": False,
    "wandb_monitor": True,
    "render_freq": 0,
    "video_save_freq": 0,
    "observe_freq": 0,
    "save_final_agent": 0,
}

# Make environment.
env = gym.make(train_parameters["env"]).unwrapped # Needed to impose custom time limit.

if train_parameters["agent"] in ("ddpg","td3","sac","steve"):
    env = NormaliseActionWrapper(env) # Actions in [-1, 1]
if train_parameters["agent"] in ("simple_model_based","steve"):
    from rlutils.specific.Pendulum import reward_function # Provide reward function.

agent_parameters = {}
agent_parameters["ddpg"] = {
    "replay_capacity": 10000,
    "batch_size": 128,
    "lr_pi": 1e-4,
    "lr_Q": 1e-3,
    "gamma": 0.99,
    "tau": 0.01,
    "noise_params": ("ou", 0., 0.15, 0.5, 0.01, 100),
}
agent_parameters["td3"] = {**agent_parameters["ddpg"], **{ 
    "td3": True,
    "td3_noise_std": 0.2,
    "td3_noise_clip": 0.5,
    "td3_policy_freq": 2
}}    
agent_parameters["sac"] = {
    "replay_capacity": 10000,
    "batch_size": 64,
    "lr_pi": 5e-4,
    "lr_Q": 1e-3,
    "gamma": 0.99,
    "alpha": 0.2,
    "tau": 0.005,
}
agent_parameters["simple_model_based"] = {
    "random_mode_only": True,
    "reward_function": reward_function,
}
agent_parameters["steve"] = {
    "reward_function": reward_function,
    "ddpg_parameters": agent_parameters["td3"]
}

a = train_parameters["agent"]
agent = rlutils.make(a, env, agent_parameters[a])
print(agent)
rlutils.train(agent, train_parameters, observer=rlutils.Observer(state_dims=3, action_dims=1))