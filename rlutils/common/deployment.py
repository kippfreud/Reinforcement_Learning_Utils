from ..agents.stable_baselines import StableBaselinesAgent

import torch 
import gym
import numpy as np
from tqdm import tqdm


P_DEFAULT = {"num_episodes": 100, "render_freq": 1}

def train(agent, P=P_DEFAULT, renderer=None, observer=None):
    return deploy(agent, P, True, renderer, observer)

def deploy(agent, P=P_DEFAULT, train=False, renderer=None, observer=None):

    # Initialise weights and biases monitoring.
    if "wandb_monitor" in P and P["wandb_monitor"]: 
        assert not type(agent)==StableBaselinesAgent, "wandb monitoring not implemented for StableBaselinesAgent."
        import wandb
        run = wandb.init(project=P["project_name"], monitor_gym=True, config={**agent.P, **P})
        run_name = run.name
        if train:
            try: wandb.watch(agent.Q)
            except: pass
            try: wandb.watch(agent.pi)
            except: pass
    else:
        import time; run_name = time.strftime("%Y-%m-%d_%H-%M-%S")

    # Add wrappers to environment.
    if "episode_time_limit" in P and P["episode_time_limit"]: # Time limit.
        agent.env = gym.wrappers.TimeLimit(agent.env, P["episode_time_limit"])
    if "video_save_freq" in P and P["video_save_freq"] > 0: # Video recording. NOTE: Must put this last.
        agent.env = gym.wrappers.Monitor(agent.env, f"./video/{run_name}", video_callable=lambda ep: ep % P["video_save_freq"] == 0, force=True)

    do_extra = "do_extra" in P and P["do_extra"]

    # Stable Baselines uses its own training and saving procedures.
    if train and type(agent)==StableBaselinesAgent: agent.train(P["sb_parameters"])
    else:
        # Iterate through episodes.
        for ep in tqdm(range(P["num_episodes"])):
            render_this_ep = "render_freq" in P and P["render_freq"] > 0 and ep % P["render_freq"] == 0
            observe_this_ep = observer and P["observe_freq"] > 0 and ep % P["observe_freq"] == 0
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
                if observe_this_ep:
                    observer.observe(ep, t, state, action, next_state, reward, info, extra)

                # Render the environment if applicable.
                if render_this_ep: agent.env.render()

                # Get state representation.
                if done: next_state = None
                elif renderer: next_state = renderer.get()
                else: next_state = torch.from_numpy(next_state).float().to(agent.device).unsqueeze(0)

                if train:
                    # Perform some agent-specific operations on each timestep.
                    agent.per_timestep(state, action, reward, next_state)

                # Update tracking variables.
                reward_sum += np.float64(reward).sum()
                state = next_state
                t += 1

            if train:
                # Perform some agent-specific operations on each episode.
                results = agent.per_episode()    
            else: results = {"logs":{}}  

            # Log to weights and biases.
            if "wandb_monitor" in P and P["wandb_monitor"]: 
                results["logs"]["reward_sum"] = reward_sum
                wandb.log(results["logs"])

        # Clean up.
        if renderer: renderer.close()
        agent.env.close()

    # Save final agent if requested.
    if "save_final_agent" in P and P["save_final_agent"]:
        if type(agent)==StableBaselinesAgent: 
            agent.save(f"saved_runs/{run_name}") 
        else:
            torch.save(agent, f"saved_runs/{run_name}.agent")

    return run_name # Return run name for reference.