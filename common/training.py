import torch # <<< NOTE: Would be good to get rid of this requirement. 
from tqdm import tqdm

def train(agent, env, train_parameters, renderer=None):

    # Initialise weights and biases monitoring.
    if train_parameters["wandb_monitor"]: 
        import wandb
        wandb.init(project=train_parameters["project_name"], config={**agent.P, **train_parameters})
        if train_parameters["model"] == "DQN": wandb.watch(agent.Q)
        elif train_parameters["model"] == "REINFORCE": wandb.watch(agent.pi)
        elif train_parameters["model"] == "ActorCritic": wandb.watch(agent.pi)
    
    # Iterate through episodes.
    for ep in tqdm(range(train_parameters["num_episodes"])):
        state, reward_sum = env.reset(), 0
        
        # Get state representation.
        if train_parameters["from_pixels"]: state, last_screen = renderer.get_delta(renderer.get_screen())
        else: state = torch.from_numpy(state).float().unsqueeze(0)
        
        # Iterate through timesteps.
        for t in range(train_parameters["max_timesteps_per_episode"]): 
            
            # Get action and advance state.
            action = agent.act(state)        
            next_state, reward, done, _ = env.step(action.item())
            if train_parameters["render"]: env.render()
            if done: next_state = None
            
            # Get state representation.
            elif train_parameters["from_pixels"]: 
                next_state, last_screen = renderer.get_delta(last_screen)
            else: 
                next_state = torch.from_numpy(next_state).float().unsqueeze(0)
            
            # Perform some agent-specific operations on each timestep.
            results = agent.per_timestep(state, action, reward, next_state)

            # Update tracking variables and terminate episode if done.
            reward_sum += reward; state = next_state
            if done: break
        
        # Perform some agent-specific operations on each episode.
        results = agent.per_episode()        

        # Log to weights and biases.
        if train_parameters["wandb_monitor"]: 
            results["logs"]["reward_sum"] = reward_sum
            wandb.log(results["logs"])

    # Clean up.
    if train_parameters["from_pixels"]: renderer.close()
    env.close()