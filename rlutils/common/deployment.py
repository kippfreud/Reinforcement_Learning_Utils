import torch # <<< NOTE: Would be good to get rid of this requirement. 
from tqdm import tqdm


def train(agent, env, parameters, renderer=None, observer=None):
    return deploy(agent, env, parameters, True, renderer, observer)

def deploy(agent, env, parameters, train=False, renderer=None, observer=None):

    # Initialise weights and biases monitoring.
    if parameters["wandb_monitor"]: 
        import wandb
        run = wandb.init(project=parameters["project_name"], config={**agent.P, **parameters})
        if train:
            if parameters["model"] == "DQN": wandb.watch(agent.Q)
            elif parameters["model"] == "REINFORCE": wandb.watch(agent.pi)
            elif parameters["model"] == "ActorCritic": wandb.watch(agent.pi)
        
    # Iterate through episodes.
    for ep in tqdm(range(parameters["num_episodes"])):
        state, reward_sum = env.reset(), 0
        
        # Get state representation.
        if renderer: state, last_screen = renderer.get_delta(renderer.get_screen())
        else: state = torch.from_numpy(state).float().unsqueeze(0)
        
        # Iterate through timesteps.
        for t in range(parameters["max_timesteps_per_episode"]): 
            
            # Get action and advance state.
            action, extra = agent.act(state)   
            next_state, reward, done, _ = env.step(action.item())
            if parameters["render"]: env.render()
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

    # Return wandb run name for reference.
    if parameters["wandb_monitor"]: return run.name 