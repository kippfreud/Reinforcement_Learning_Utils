from rlutils.common.deployment import train

import gym

train_parameters = {
    "project_name": "pong",
    "env": "Pong-v0",
    "agent": "dqn",
    "num_episodes": int(5e4),
    "episode_time_limit": None,
    "from_pixels": True,
    "wandb_monitor": True,
    "render_freq": 0,
    "video_save_freq": 250,
    "save_final_agent": False,
}

# Make environment.
env = gym.make(train_parameters["env"]).unwrapped
if train_parameters["from_pixels"]:
    # If from_pixels, set up screen processor.
    from rlutils.common.rendering import Renderer
    from rlutils.specific.Pong import screen_processor # <<< NOTE: HARD CODED FOR PONG!
    env.reset()
    renderer = Renderer(env, screen_processor, mode="diff", params={"prev_alpha": 0.5})
    renderer.get(first=True); env.step(0); s = renderer.get(show=False)
    state_shape = s.shape
else: state_shape, renderer = env.observation_space.shape, None

# Make DqnAgent.
if train_parameters["model"] == "dqn":
    agent_parameters = {
        "replay_capacity": int(1e6),
        "batch_size": 32,
        "lr_Q": 1e-3,
        "gamma": 0.99,
        "epsilon_start": 1,
        "epsilon_end": 0.1,
        "epsilon_decay": int(1e6),
        "updates_between_target_clone": 10000
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
