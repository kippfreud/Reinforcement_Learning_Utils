import gym
import numpy as np
import rlutils
from rlutils.common.env_wrappers import NormaliseActionWrapper
from rlutils.specific.Pendulum import reward_function

train_parameters = {
    # "project_name":         "pendulum",
    # "env":                  "Pendulum-v0",
    # "state_dims":           ["cos_theta","sin_theta","theta_dot"],

    "project_name":         "mountaincar",
    "env":                  "MountainCarContinuous-v0",
    "state_dims":           ["pos","vel"],

    "agent":                "diayn",
    "num_episodes":         1000,
    "episode_time_limit":   200,
    "from_pixels":          False,
    "wandb_monitor":        True,
    "render_freq":          0,
    "video_save_freq":      0,
    "video_to_wandb":       True,
    "observe_freq":         0,
    "checkpoint_freq":      100,
}

# Make environment.
env = gym.make(train_parameters["env"]).unwrapped # Needed to impose custom time limit.

if train_parameters["agent"] in ("ddpg","td3","sac","diayn","steve"):
    env = NormaliseActionWrapper(env) # Actions in [-1, 1]

agent_parameters = {}
agent_parameters["ddpg"] = {
    "replay_capacity":      5000,
    "batch_size":           32,
    "lr_pi":                1e-4,
    "lr_Q":                 1e-3,
    "gamma":                0.99,
    "tau":                  0.01,
    "noise_params":         ("ou", 0., 0.15, 0.5, 0.01, 100),
}
agent_parameters["td3"] = {**agent_parameters["ddpg"], **{ 
    "td3": True,
    "td3_noise_std":        0.2,
    "td3_noise_clip":       0.5,
    "td3_policy_freq":      2
}}    
agent_parameters["sac"] = {
    "replay_capacity":      5000,
    "batch_size":           32,
    "lr_pi":                1e-4,
    "lr_Q":                 1e-3,
    "gamma":                0.99,
    "alpha":                0.2,
    "tau":                  0.01,
}
agent_parameters["diayn"] = {
    "num_skills":           20, 
    "sac_parameters":       {"batch_size": 128, "alpha": 0.1, "tau": 0.01}
}
agent_parameters["simple_model_based"] = {
    "random_mode_only":     False,
    "reward_function":      reward_function,
}
agent_parameters["steve"] = {
    "reward_function":      reward_function,
    "ddpg_parameters":      agent_parameters["td3"]
}

a = train_parameters["agent"]
agent = rlutils.make(a, env, agent_parameters[a])
print(agent)
obs = rlutils.Observer(state_dims=train_parameters["state_dims"], action_dims=1)
_, rn = rlutils.train(agent, train_parameters, observer=obs)

if train_parameters["observe_freq"]:
    obs.add_custom_dims(lambda x: np.array([np.arccos(x[2]) * np.sign(x[3])]), ["theta"])
    obs.add_future(["reward"], gamma=agent.P["gamma"], new_dims=["return"]) # Add return dim.
    obs.save(f"runs/{rn}_train.csv")