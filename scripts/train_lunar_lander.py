from rlutils.common.deployment import train

import gym

train_parameters = {
    # * = Not used by StableBaselinesAgent.
    "project_name": "lunar_lander",
    "env": "LunarLanderContinuous-v2",
    "model": "sac",
    "num_episodes": 1, # *
    "episode_time_limit": 200, # *
    "wandb_monitor": False, # *
    "render_freq": 0,
    "video_save_freq": 0,
    "save_final_agent": False,
}

# Make environment.
env = gym.make(train_parameters["env"])

# Make DdpgAgent.
if train_parameters["model"] in ("ddpg","td3"):
    agent_parameters = {
        "replay_capacity": 50000,
        "batch_size": 256,
        "lr_pi": 1e-4,
        "lr_Q": 1e-3,
        "gamma": 0.99,
        "tau": 0.01,
        "noise_params": (0., 0.15, 0.5, 0.01, 300000),
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
        "batch_size": 256,
        "lr_pi": 1e-4,
        "lr_Q": 1e-3,
        "gamma": 0.99,
        "alpha": 0.2,
        "tau": 0.01,
    }
    from rlutils.agents.sac import *
    agent = SacAgent(env.observation_space.shape, env.action_space.shape[0], agent_parameters)

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