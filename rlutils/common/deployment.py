import torch 
from tqdm import tqdm


def train(agent, env, parameters, renderer=None, observer=None):
    return deploy(agent, env, parameters, True, renderer, observer)

def deploy(agent, env, parameters, train=False, renderer=None, observer=None):

    # Initialise weights and biases monitoring.
    if parameters["wandb_monitor"]: 
        import wandb
        run = wandb.init(project=parameters["project_name"], config={**agent.P, **parameters})
        if train:
            if parameters["model"] == "dqn": wandb.watch(agent.Q)
            elif parameters["model"] == "reinforce": wandb.watch(agent.pi)
            elif parameters["model"] == "actor-critic": wandb.watch(agent.pi)
            elif parameters["model"] == "ddpg": wandb.watch(agent.pi)

    # Iterate through episodes.
    for ep in tqdm(range(parameters["num_episodes"])):
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
            next_state, reward, done, _ = env.step(action_for_env)
            if parameters["render_freq"] > 0 and ep % parameters["render_freq"] == 0: env.render()
            if done: next_state = None
            
            # Get state representation.
            elif renderer: next_state, last_screen = renderer.get_delta(last_screen)
            else: next_state = torch.from_numpy(next_state).float().unsqueeze(0)
            
            if train:
                # Perform some agent-specific operations on each timestep.
                agent.per_timestep(state, action, reward, next_state)

            if observer:
                # Send an observation to the observer.
                observer.observe(ep, t, state, action, reward, next_state, extra)

            # Update tracking variables and terminate episode if done.
            reward_sum += reward; state = next_state
            if done: break
        
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
        from joblib import dump
        if parameters["wandb_monitor"]: run_name = run.name # Using wandb run name if possible.
        else: import time; run_name = "untitled_" + time.strftime("%Y-%m-%d_%H-%M-%S")
        dump(agent, f"{run_name}.joblib") 

    return run_name # Return run name for reference.