import gym
import wandb
import matplotlib.pyplot as plt

ENV = "CartPole-v1"
PROJECT_NAME = "cartpole"
WANDB = True

run_parameters = {
    "model": "DQN",
    "num_episodes": 2000,
    "max_timesteps_per_episode": 500,
    "from_pixels": False
}

# Make environment.
env = gym.make(ENV).unwrapped
if run_parameters["from_pixels"]:
    # If from_pixels, set up screen processor.
    from common.rendering import Renderer
    from specific.CartPole import screen_processor
    env.reset()
    renderer = Renderer(env, screen_processor)
    state_shape = renderer.get_screen().shape
else: state_shape = env.observation_space.shape

# Make DqnAgent.
if run_parameters["model"] == "DQN":
    model_parameters = {
        "replay_capacity": 10000,
        "batch_size": 128,
        "lr": 1e-3,
        "gamma": 0.99,
        "epsilon_start": 0.9,
        "epsilon_end": 0.05,
        "epsilon_decay": 10000,
        "updates_between_target_clone": 2000
    }
    from agents.dqn import *
    agent = DqnAgent(state_shape, env.action_space.n, model_parameters)

# Make ReinforceAgent.
elif run_parameters["model"] == "REINFORCE":
    model_parameters = {
        "lr_pi": 1e-4,
        "lr_V": 1e-3,
        "gamma": 0.99,
        "baseline": "off"
    }   
    from agents.reinforce import *
    agent = ReinforceAgent(state_shape, env.action_space.n, model_parameters)

# Make ActorCritcAgent.
elif run_parameters["model"] == "ActorCritic":
    model_parameters = {
        "lr_pi": 1e-4,
        "lr_V": 1e-3,
        "gamma": 0.99
    }   
    from agents.actor_critic import *
    agent = ActorCriticAgent(state_shape, env.action_space.n, model_parameters)

# Initialise weights and biases monitoring.
if WANDB: 
    wandb.init(project=PROJECT_NAME, config={**agent.P, **run_parameters})
    if run_parameters["model"] == "DQN": wandb.watch(agent.Q)
    elif run_parameters["model"] == "REINFORCE": wandb.watch(agent.pi)
    elif run_parameters["model"] == "ActorCritic": wandb.watch(agent.pi)

# Run training loop.
for ep in range(run_parameters["num_episodes"]):
    state, reward_sum = env.reset(), 0
    # Get state representation.
    if run_parameters["from_pixels"]: state, last_screen = renderer.get_delta(renderer.get_screen())
    else: state = torch.from_numpy(state).float().unsqueeze(0)
    # Iterate through timesteps.
    for t in range(run_parameters["max_timesteps_per_episode"]): 
        # Get action; advance state.
        action = agent.act(state)        
        next_state, reward, done, _ = env.step(action.item())
        if done: next_state = None
        # Get state representation.
        elif run_parameters["from_pixels"]: 
            next_state, last_screen = renderer.get_delta(last_screen)
            # plt.figure()
            # plt.imshow(renderer.to_numpy(next_state))
            # plt.show()
        else: 
            next_state = torch.from_numpy(next_state).float().unsqueeze(0)

        # ====== MODEL-SPECIFIC ======
        # DQN updates on every timestep.
        if run_parameters["model"] == "DQN":
            agent.memory.add(state, action, next_state, torch.tensor([reward], device=agent.device))
            value_loss = agent.update_on_batch()
        # REINFORCE requires us to store all rewards.
        elif run_parameters["model"] == "REINFORCE":
            agent.ep_rewards.append(reward)
        # ActorCritic updates on every timestep.
        elif run_parameters["model"] == "ActorCritic":
            policy_loss, value_loss = agent.update_on_transition(next_state, torch.tensor([reward], device=agent.device))
        # ============================

        # Update tracking variables and terminate episode if done.
        reward_sum += reward; state = next_state
        if done: break
    logs = {"reward": reward_sum}

    # ====== MODEL-SPECIFIC ======
    if run_parameters["model"] == "DQN":
        logs["epsilon"] = agent.epsilon
        logs["value_loss"] = value_loss
    elif run_parameters["model"] == "REINFORCE":
        policy_loss, value_loss = agent.update_on_episode() # Update happens here.
        logs["policy_loss"] = policy_loss
        logs["value_loss"] = value_loss
    elif run_parameters["model"] == "ActorCritic": 
        logs["policy_loss"] = policy_loss
        logs["value_loss"] = value_loss
    # ============================

    # Log to weights and biases.  
    if WANDB: wandb.log(logs)

# Clean up.
if run_parameters["from_pixels"]: renderer.close()
env.close()