from ..agents.stable_baselines import StableBaselinesAgent

import torch 
import gym
import numpy as np
from tqdm import tqdm


def train(agent, env, parameters, renderer=None, observer=None):
    return deploy(agent, env, parameters, True, renderer, observer)

def deploy(agent, env, parameters, train=False, renderer=None, observer=None):

    # Initialise weights and biases monitoring.
    if parameters["wandb_monitor"]: 
        assert not type(agent)==StableBaselinesAgent, "wandb monitoring not implemented for StableBaselinesAgent."
        import wandb
        run = wandb.init(project=parameters["project_name"], monitor_gym=True, config={**agent.P, **parameters})
        run_name = run.name
        if train:
            try:
                if parameters["model"] == "dqn": wandb.watch(agent.Q)
                elif parameters["model"] in ("reinforce","actor-critic","ddpg","td3"): wandb.watch(agent.pi)
            except: pass
    else:
        import time; run_name = time.strftime("%Y-%m-%d_%H-%M-%S")

    # Stable Baselines uses its own training and saving procedures.
    if train and type(agent)==StableBaselinesAgent:
        agent.train(parameters["sb_parameters"])
    else:
        # Iterate through episodes.
        for ep in tqdm(range(parameters["num_episodes"])):
            render_this_ep = parameters["render_freq"] > 0 and ep % parameters["render_freq"] == 0
            observe_this_ep = observer and parameters["observe_freq"] > 0 and ep % parameters["observe_freq"] == 0
            if parameters["save_video"] and render_this_ep: env = gym.wrappers.Monitor(env, f"./video/{run_name}/{ep}", force=True) # Record a new video every episode.
            state, reward_sum = env.reset(), 0
            
            # Get state representation.
            if renderer: state, last_screen = renderer.get_delta(renderer.get_screen())
            else: state = torch.from_numpy(state).float().unsqueeze(0)
            
            # Iterate through timesteps.
            for t in range(parameters["max_timesteps_per_episode"]): 
                
                # Get action and advance state.
                action, extra = agent.act(state, explore=train) # If not in training mode, turn exploration off.
                try: action_for_env = action.item() # If action is 1D, just extract its item().
                except: action_for_env = action # Otherwise, keep the whole vector.
                next_state, reward, done, info = env.step(action_for_env)

                # Send an observation to the observer if applicable.
                if observe_this_ep:
                    observer.observe(ep, t, state, action_for_env, next_state, reward, info, extra)

                # Render the environment if applicable.
                if render_this_ep: env.render()

                # Get state representation.
                if done: next_state = None
                elif renderer: next_state, last_screen = renderer.get_delta(last_screen)
                else: next_state = torch.from_numpy(next_state).float().unsqueeze(0)

                if train:
                    # Perform some agent-specific operations on each timestep.
                    agent.per_timestep(state, action, reward, next_state)

                # Update tracking variables and terminate episode if done.
                reward_sum += np.float64(reward).sum()
                if done: break
                state = next_state
            
            if train:
                # Perform some agent-specific operations on each episode.
                results = agent.per_episode()    
            else: results = {"logs":{}}  

            # Log to weights and biases.
            if parameters["wandb_monitor"]: 
                results["logs"]["reward_sum"] = reward_sum
                wandb.log(results["logs"])

        # Clean up.
        if renderer: renderer.close()
        env.close()

    # Save final agent if requested.
    if parameters["save_final_agent"]:
        if type(agent)==StableBaselinesAgent: 
            agent.save(f"saved_runs/{run_name}") 
        else:
            from joblib import dump
            dump(agent, f"saved_runs/{run_name}.joblib") 

    return run_name # Return run name for reference.