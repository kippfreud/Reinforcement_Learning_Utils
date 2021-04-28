import gym
import rlutils

train_parameters = {
    "project_name": "pendulum",
    "env": "Pendulum-v0",
    "agent": "td3",
    "num_episodes": 100,
    "episode_time_limit": 500,
    "from_pixels": False,
    "wandb_monitor": False,
    "render_freq": 0,
    "video_save_freq": 0,
    "save_final_agent": False,
}

# Make environment.
env = rlutils.commmon.env_wrappers.NormaliseActionWrapper(gym.make(train_parameters["env"]))

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
agent = rlutils.agent(train_parameters["agent"], env, agent_parameters)
run_name = train(agent, env, train_parameters)