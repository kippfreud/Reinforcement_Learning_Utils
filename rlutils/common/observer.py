"""
This is a class for collecting observational data of a trained agent.
"""
import torch # <<< NOTE: Would be good to get rid of this requirement. 
import numpy as np
import pandas as pd
from tqdm import tqdm


PROTECTED_DIM_NAMES = {"ep", "time", "reward", "pi", "Q", "V"}


class Observer:
    def __init__(self, agent, env, renderer=None):
        if renderer: raise NotImplementedError("Observer currently only works with vector states.")
        self.agent, self.env, self.renderer = agent, env, renderer

    def observe(self, observe_parameters):

        # Check for protected dim_names.
        illegal = PROTECTED_DIM_NAMES & set(observe_parameters["state_dim_names"] + observe_parameters["action_dim_names"])
        if illegal: raise Exception(f"dim_names {illegal} illegal because they are protected.")

        # List the dimensions in the dataset to be constructed
        dim_names = ["ep", "time"] \
                  + observe_parameters["state_dim_names"] \
                  + observe_parameters["action_dim_names"] \
                  + ["reward"] \
        
        dataset = []

        # Iterate through episodes.
        for ep in tqdm(range(observe_parameters["num_episodes"])):
            state, reward_sum = self.env.reset(), 0
            
            # Get state representation.
            # if self.renderer: state, last_screen = self.renderer.get_delta(self.renderer.get_screen())
            # else: 
            state = torch.from_numpy(state).float().unsqueeze(0)
            
            # Iterate through timesteps.
            for t in range(observe_parameters["max_timesteps_per_episode"]): 
                
                # Get action and advance state.
                action, extra = self.agent.act(state)        
                next_state, reward, done, _ = self.env.step(action.item())
                if observe_parameters["render"]: self.env.render()
                if done: next_state = None
                
                # Get state representation.
                # elif self.renderer: next_state, last_screen = self.renderer.get_delta(last_screen)
                else: next_state = torch.from_numpy(next_state).float().unsqueeze(0)

                # Construct the observation for this timestep.
                observation = [ep, t] \
                            + list(state.numpy().flatten()) \
                            + list(action.numpy().flatten()) \
                            + [reward]
                for k,v in extra.items():
                    if type(v) == np.ndarray: v = list(v.flatten())
                    else: v = [v]
                    observation += v
                    # Get dim_names for the extra dictionary. 
                    if len(dataset) == 0:
                        if k in ("action_probs", "Q"):
                            dim_names += [f"{k}_{i}" for i in range(len(v))]
                        else:
                            dim_names += [k]

                        print(dim_names)


                dataset.append(observation)

                

                """
                ep
                time
                state
                action
                reward
                extra 
                """

                # print(observation)
                
                # Terminate episode if done.
                state = next_state
                if done: break

        # return dataset