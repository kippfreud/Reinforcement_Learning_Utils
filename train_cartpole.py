from rlutils.common.deployment import train

import gym
from joblib import dump

train_parameters = {
    "project_name": "cartpole",
    "env": "CartPole-v1",
    "model": "dqn",
    "num_episodes": 2000,
    "max_timesteps_per_episode": 500,
    "from_pixels": False,
    "wandb_monitor": True,
    "save_trained_agent": False,
    "render": False
}

# Make environment.
env = gym.make(train_parameters["env"]).unwrapped
if train_parameters["from_pixels"]:
    # If from_pixels, set up screen processor.
    from rlutils.common.rendering import Renderer
    from rlutils.specific.CartPole import screen_processor # <<< NOTE: HARD CODED FOR CARTPOLE!
    env.reset()
    renderer = Renderer(env, screen_processor)
    state_shape = renderer.get_screen().shape
else: state_shape, renderer = env.observation_space.shape, None

# Make DqnAgent.
if train_parameters["model"] == "dqn":
    agent_parameters = {
        "replay_capacity": 10000,
        "batch_size": 128,
        "lr": 1e-3,
        "gamma": 0.99,
        "epsilon_start": 0.9,
        "epsilon_end": 0.05,
        "epsilon_decay": 10000,
        "updates_between_target_clone": 2000
    }
    from rlutils.agents.dqn import *
    agent = DqnAgent(state_shape, env.action_space.n, agent_parameters)

# Make ReinforceAgent.
elif train_parameters["model"] == "reinforce":
    agent_parameters = {
        "lr_pi": 1e-4,
        "lr_V": 1e-3,
        "gamma": 0.99,
        "baseline": "adv"
    }   
    from rlutils.agents.reinforce import *
    agent = ReinforceAgent(state_shape, env.action_space.n, agent_parameters)

# Make ActorCriticAgent.
elif train_parameters["model"] == "actor-critic":
    agent_parameters = {
        "lr_pi": 1e-4,
        "lr_V": 1e-3,
        "gamma": 0.99
    }   
    from rlutils.agents.actor_critic import *
    agent = ActorCriticAgent(state_shape, env.action_space.n, agent_parameters)

run_name = train(agent, env, train_parameters, renderer)
if train_parameters["save_trained_agent"]:
    if not run_name: run_name = "unnamed_run"
    dump(agent, f"{run_name}.joblib") # Save agent using wandb run name.