import gym
import rlutils

train_parameters = {
    "project_name": "cartpole",
    "env": "CartPole-v1",
    "agent": "simple_model_based",
    "num_episodes": 2000,
    "episode_time_limit": 500,
    "from_pixels": False,
    "wandb_monitor": False,
    "render_freq": 0,
    "video_save_freq": 0,
    "save_final_agent": False,
}

# Make environment.
env = gym.make(train_parameters["env"]).unwrapped
if train_parameters["from_pixels"]:
    # If from_pixels, set up screen processor.
    from rlutils.common.rendering import Renderer
    from rlutils.specific.CartPole import screen_processor # <<< NOTE: HARD CODED FOR CARTPOLE!
    env.reset()
    renderer = Renderer(env, screen_processor, mode="diff")
    renderer.get(first=True); env.step(0); s = renderer.get(show=False)
    state_shape = s.shape
else: state_shape, renderer = env.observation_space.shape, None

if train_parameters["agent"] == "dqn":
    agent_parameters = {
        "replay_capacity": 50000,
        "batch_size": 32,
        "lr_Q": 1e-3,
        "epsilon_start": 1,
        "epsilon_end": 0.05,
        "epsilon_decay": 100000,
        "updates_between_target_clone": 2000
    }
elif train_parameters["agent"] == "reinforce":
    agent_parameters = {
        "lr_pi": 1e-4,
        "lr_V": 1e-3,
        "baseline": "adv"
    }
elif train_parameters["agent"] == "actor_critic":
    agent_parameters = {
        "lr_pi": 1e-4,
        "lr_V": 1e-3,
    }
elif train_parameters["agent"] == "simple_model_based":
    from rlutils.specific.CartPole import reward_function
    agent_parameters = {
        "reward_function": reward_function,
    }
agent = rlutils.agent(train_parameters["agent"], env, agent_parameters)
print(agent)
run_name = rlutils.train(agent, env, train_parameters, renderer)