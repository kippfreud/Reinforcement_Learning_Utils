import gym
import rlutils

train_parameters = {
    "project_name": "cartpole",
    "env": "CartPole-v1",
    "agent": "actor_critic",
    "num_episodes": 1500,
    "episode_time_limit": 500,
    "from_pixels": False,
    "wandb_monitor": True,
    "render_freq": 0,
    "video_save_freq": 0,
    "observe_freq": 0,
    "checkpoint_freq": 0,
}

# Make environment.
env = gym.make(train_parameters["env"]).unwrapped # Needed to impose custom time limit.
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
        "net_Q": [(None, 32), "R", (32, 32), "R", (32, None)],
        "replay_capacity": 10000,
        "batch_size": 32,
        "lr_Q": 1e-3,
        "epsilon_start": 1,
        "epsilon_end": 0.05,
        "epsilon_decay": 10000,
        "target_update": ("soft", 0.0005),
        "double": True
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
        "random_mode_only": False,
        "reward_function": reward_function
    }

agent = rlutils.make(train_parameters["agent"], env, agent_parameters)
print(agent)
obs = rlutils.Observer(state_dims=["pos","vel","ang","ang_vel"], action_dims=1)
_, rn = rlutils.train(agent, train_parameters, renderer, observer=obs)

if train_parameters["observe_freq"]:
    obs.add_future(["reward"], gamma=agent.P["gamma"], new_dims=["return"]) # Add return dim.
    obs.save(f"runs/{rn}_train.csv")