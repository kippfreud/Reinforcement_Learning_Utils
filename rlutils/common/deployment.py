from ..agents.stable_baselines import StableBaselinesAgent

import torch 
import gym
import numpy as np
from tqdm import tqdm


P_DEFAULT = {"num_episodes": 100, "render_freq": 1}

def train(agent, P=P_DEFAULT, renderer=None, observer=None):
    return deploy(agent, P, True, renderer, observer)

def deploy(agent, P=P_DEFAULT, train=False, renderer=None, observer=None, save_dir="runs"):

    do_extra = "do_extra" in P and P["do_extra"] # Whether or not to request extra predictions from the agent.
    do_wandb = "wandb_monitor" in P and P["wandb_monitor"]
    do_render = "render_freq" in P and P["render_freq"] > 0
    if "observe_freq" in P and P["observe_freq"] > 0:
        if observer: do_observe = True
        else: raise Exception("No observer provided!") 
    else: do_observe = False
    do_checkpoints = "checkpoint_freq" in P and P["checkpoint_freq"] > 0

    # Initialise weights and biases monitoring.
    if do_wandb: 
        assert not type(agent)==StableBaselinesAgent, "wandb monitoring not implemented for StableBaselinesAgent."
        import wandb
        run = wandb.init(project=P["project_name"], monitor_gym=True, config={**agent.P, **P})
        run_name = run.name
        # if train: # TODO: Weight monitoring causes an error with STEVE.
            # try: 
                # if type(agent.Q) == list: # Handling Q ensembles.
                    # for Q in agent.Q: wandb.watch(Q)
                # else:
                # wandb.watch(agent.Q)
            # except: pass
            # try: wandb.watch(agent.pi)
            # except: pass
    else:
        import time; run_name = time.strftime("%Y-%m-%d_%H-%M-%S")

    # Add wrappers to environment.
    if "episode_time_limit" in P and P["episode_time_limit"]: # Time limit.
        agent.env = gym.wrappers.TimeLimit(agent.env, P["episode_time_limit"])
    if "video_save_freq" in P and P["video_save_freq"] > 0: # Video recording. NOTE: Must put this last.
        agent.env = gym.wrappers.Monitor(agent.env, f"./video/{run_name}", video_callable=lambda ep: ep % P["video_save_freq"] == 0, force=True)

    # Create directory for saving.
    if do_observe or do_checkpoints: import os; os.makedirs(save_dir, exist_ok=True)

    # Stable Baselines uses its own training and saving procedures.
    if train and type(agent)==StableBaselinesAgent: agent.train(P["sb_parameters"])
    else:
        # Iterate through episodes.
        for ep in tqdm(range(P["num_episodes"])):
            render_this_ep = do_render and ep % P["render_freq"] == 0
            observe_this_ep = do_observe and ep % P["observe_freq"] == 0
            checkpoint_this_ep = do_checkpoints and (ep+1) % P["checkpoint_freq"] == 0
            state, reward_sum, t, done = agent.env.reset(), 0, 0, False
            
            # Get state representation.
            if renderer: state = renderer.get(first=True)
            else: state = torch.from_numpy(state).float().to(agent.device).unsqueeze(0)
            
            # Iterate through timesteps.
            while not done:
                
                # Get action and advance state.
                action, extra = agent.act(state, explore=train, do_extra=do_extra) # If not in training mode, turn exploration off.
                next_state, reward, done, info = agent.env.step(action)

                # Send an observation to the observer if applicable.
                if observe_this_ep: observer.observe(ep, t, state, action, next_state, reward, info, extra)

                # Render the environment if applicable.
                if render_this_ep: agent.env.render()

                # Get state representation.
                if renderer: next_state = renderer.get()
                else: next_state = torch.from_numpy(next_state).float().to(agent.device).unsqueeze(0)

                # Perform some agent-specific operations on each timestep if training.
                if train: agent.per_timestep(state, action, reward, next_state, done)

                # Update tracking variables.
                reward_sum += np.float64(reward).sum(); state = next_state; t += 1

            # Perform some agent-specific operations on each episode if training.
            if train: results = agent.per_episode()    
            else: results = {"logs": {}}  

            # Log to weights and biases if applicable.
            if do_wandb: results["logs"]["reward_sum"] = reward_sum; wandb.log(results["logs"])

            # Save current agent model if applicable.
            if checkpoint_this_ep:
                fname = f"{save_dir}/{run_name}_ep{ep+1}"
                if type(agent)==StableBaselinesAgent: agent.save(fname) 
                else: torch.save(agent, f"{fname}.agent")

        # Clean up.
        if renderer: renderer.close()
        agent.env.close()

    return run_name # Return run name for reference.