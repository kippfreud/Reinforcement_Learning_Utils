from rlutils.common.deployment import train

import gym

train_parameters = {
    "project_name": "cartpole",
    "env": "CartPole-v1",
    "model": "simple_model_based",
    "num_episodes": 200,
    "episode_time_limit": 500,
    "from_pixels": False,
    "wandb_monitor": True,
    "render_freq": 0,
    "video_save_freq": 0,
    "save_final_agent": True,
}

# Make environment.
env = gym.make(train_parameters["env"]).unwrapped

# Hand-specify reward function.
import math
def reward_function(state):
    theta_threshold_radians = 12 * 2 * math.pi / 360
    x_threshold = 2.4
    x, _, theta, _ = state
    done = bool(x < -x_threshold
                or x > x_threshold
                or theta < -theta_threshold_radians
                or theta > theta_threshold_radians)
    return 1 if not done else 0

# Make SimpleModelBasedAgent.
if train_parameters["model"] == "simple_model_based":
    from rlutils.agents.simple_model_based import *
    agent = SimpleModelBasedAgent(env.observation_space.shape, env.action_space, reward_function)

run_name = train(agent, env, train_parameters)