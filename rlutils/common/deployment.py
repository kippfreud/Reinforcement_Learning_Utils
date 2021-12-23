from ..agents.stable_baselines import StableBaselinesAgent

import torch 
import numpy as np
from tqdm import tqdm
from gym import wrappers

""""
TODO: Repeated calls with persistent run_id causes Monitor wrapper to be re-applied! Possible solutions:
- Unwrap on agent.env.close()
- Never actually wrap agent.env, but create copy in here which does have wrappers
"""

P_DEFAULT = {"num_episodes": int(1e6), "render_freq": 1}


def train(agent, P=P_DEFAULT, renderer=None, observers=[], run_id=None, save_dir="agents"):
    return deploy(agent, P, True, renderer, observers, run_id, save_dir)

def deploy(agent, P=P_DEFAULT, train=False, renderer=None, observers=[], run_id=None, save_dir="agents"):

    do_extra = "do_extra" in P and P["do_extra"] # Whether or not to request extra predictions from the agent.
    do_wandb = "wandb_monitor" in P and P["wandb_monitor"]
    do_render = "render_freq" in P and P["render_freq"] > 0
    do_checkpoints = "checkpoint_freq" in P and P["checkpoint_freq"] > 0    
    if do_checkpoints: import os; os.makedirs(save_dir, exist_ok=True) # Create directory for saving.
    # do_reward_decomposition = "reward_components" in agent.P and agent.P["reward_components"] is not None

    if do_wandb: 
        # Initialise Weights & Biases monitoring.
        assert not type(agent)==StableBaselinesAgent, "wandb monitoring not implemented for StableBaselinesAgent."
        import wandb
        if run_id is None: run_id, resume = wandb.util.generate_id(), "never"
        else: resume = "must"
        run = wandb.init(
            project=P["project_name"], 
            id=run_id, 
            resume=resume, 
            monitor_gym="video_to_wandb" in P and P["video_to_wandb"],
            config={**agent.P, **P})
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
        import time; run_id, run_name = None, time.strftime("%Y-%m-%d_%H-%M-%S")

    # Tell observers what the run name is.
    for o in observers: o.run_names.append(run_name) 

    # Add wrappers to environment.
    if "episode_time_limit" in P and P["episode_time_limit"]: # Time limit.
        agent.env = wrappers.TimeLimit(agent.env, P["episode_time_limit"])
    if "video_freq" in P and P["video_freq"] > 0: # Video recording. NOTE: Must put this last.
        agent.env = wrappers.Monitor(agent.env, f"./video/{run_name}", video_callable=lambda ep: ep % P["video_freq"] == 0, force=True)

    # Stable Baselines uses its own training and saving procedures.
    if train and type(agent)==StableBaselinesAgent: agent.train(P["sb_parameters"])
    else:
        # Iterate through episodes.
        state = agent.env.reset()
        for ep in tqdm(range(P["num_episodes"])):
            render_this_ep = do_render and (ep+1) % P["render_freq"] == 0
            checkpoint_this_ep = do_checkpoints and ((ep+1) == P["num_episodes"] or (ep+1) % P["checkpoint_freq"] == 0)
            
            # Get state in PyTorch format expected by agent.
            state_torch = renderer.get(first=True) if renderer else torch.from_numpy(state).float().to(agent.device).unsqueeze(0)
            
            # Iterate through timesteps.
            t = 0; done = False; reward_sum = 0
            while not done:
                
                # Get action and advance state.
                action, extra = agent.act(state_torch, explore=train, do_extra=do_extra) # If not in training mode, turn exploration off.
                next_state, reward, done, info = agent.env.step(action)
                next_state_torch = renderer.get() if renderer else torch.from_numpy(next_state).float().to(agent.device).unsqueeze(0)
                reward_sum += (sum(extra["reward_components"]) if "reward_components" in extra else np.float64(reward).sum()) 
                
                # Perform some agent-specific operations on each timestep if training.
                if train: agent.per_timestep(state_torch, action, reward, next_state_torch, done)
                                # info["reward_components"] if do_reward_decomposition else reward, 

                # Send all information relating to the current timestep to to the observers.
                for o in observers: o.per_timestep(ep, t, state, action, next_state, reward, done, info, extra)

                # Render the environment if applicable.
                if render_this_ep: agent.env.render()

                state = next_state; state_torch = next_state_torch; t += 1
            
            state = agent.env.reset() # NOTE: PbRL observer requires env.reset() here.
                    
            # Perform some agent- and observer-specific operations on each episode, which may create logs.
            logs = {"reward_sum": reward_sum}
            if train: logs.update(agent.per_episode())    
            elif hasattr(agent, "per_episode_deploy"): logs.update(agent.per_episode_deploy())   
            for o in observers: logs.update(o.per_episode(ep))

            # Send logs to Weights & Biases if applicable.
            if do_wandb: wandb.log(logs)

            # Periodic save-outs of checkpoints.
            if checkpoint_this_ep: agent.save(f"{save_dir}/{run_name}_ep{ep+1}") 

        # Clean up.
        if renderer: renderer.close()
        agent.env.close()

    return run_id, run_name # Return run ID and name for reference.


# TODO: Move elsewhere.
class SumLogger:
    def __init__(self, name, info_or_extra, key): 
        self.name, self.info_or_extra, self.key = name, info_or_extra, key
        self.run_names = []
    def per_timestep(self,ep, t, state, action, next_state, reward, done, info, extra):
        if self.info_or_extra == "info": c = info[self.key]
        elif self.info_or_extra == "extra": c = extra[self.key]
        if t == 0: self.sums = np.array(c)
        else: self.sums += c
    def per_episode(self, ep): 
        return {f"{self.name}_{c}": r for c, r in enumerate(self.sums)}