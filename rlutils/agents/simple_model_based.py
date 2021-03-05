"""
Simple model-based agent for discrete action spaces.
One component of the architecture from:
    "Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning"
"""

from ..common.networks import SequentialNetwork
from ..common.memory import ReplayMemory

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_HYPERPARAMETERS = {  
    "replay_capacity": 2000,
    "random_replay_capacity": 2000, 
    "batch_size": 256,
    "batch_ratio": 0.9, # Proportion of on-policy transitions.
    "steps_between_update": 10,
    "lr_model": 1e-3,
    "num_rollouts": 50,
    "rollout_horizon": 20,
    "gamma": 0.99,
}

class SimpleModelBasedAgent:
    def __init__(self, 
                 state_shape, 
                 action_space,
                 reward_function,
                 hyperparameters=DEFAULT_HYPERPARAMETERS
                 ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space = action_space
        self.reward_function = reward_function
        self.P = hyperparameters 
        # Create model network.
        if len(state_shape) > 1: raise NotImplementedError()
        else:
            net_code = [(state_shape[0]+1, 32), "R", (32, 64), "R", (64, state_shape[0])]
        self.model = SequentialNetwork(code=net_code, lr=self.P["lr_model"]).to(self.device)
        # Create replay memory in two components: one for on-policy transitions and one for random transitions.
        self.memory = ReplayMemory(self.P["replay_capacity"]) 
        self.random_memory = ReplayMemory(self.P["random_replay_capacity"]) 
        self.batch_split = (round(self.P["batch_size"] * self.P["batch_ratio"]), round(self.P["batch_size"] * (1-self.P["batch_ratio"])))
        # Tracking variables.
        self.random_mode = True
        self.total_t = 0 # Used for steps_between_update.
        self.ep_losses = []

    def act(self, state, explore=True):
        """Either random or model-based action selection."""
        if self.random_mode: action, extra = self.action_space.sample(), {}
        else: 
            returns, first_actions = self._model_rollout(state)
            best_rollout = np.argmax(returns)
            action, extra = first_actions[best_rollout], {"G": returns[best_rollout]}
        return action, extra

    def update_on_batch(self):
        """Use a random batch from the replay memory to update the model network parameters."""

        # TODO: NORMALISATION

        if self.random_mode: # During random mode, just sample from random memory.   
            if len(self.random_memory) < self.P["batch_size"]: return 
            batch = self.random_memory.sample(self.P["batch_size"])
        else: # After random mode, sample from both memories according to self.batch_split.
            if len(self.memory) < self.batch_split[0]: return 
            batch = list(self.memory.sample(self.batch_split[0])) + list(self.random_memory.sample(self.batch_split[1]))
        states_and_actions = torch.cat(tuple(torch.cat((x.state, torch.Tensor([[x.action]]).to(self.device)), dim=-1) for x in batch), dim=0)
        next_states = torch.cat(tuple(x.next_state for x in batch)).to(self.device)
        # Update model in the direction of the true change in state using MSE loss.
        target = next_states - states_and_actions[:,:-1]
        prediction = self.model(states_and_actions)
        loss = F.mse_loss(prediction, target)
        self.model.optimise(loss)
        return loss.item()

    def per_timestep(self, state, action, reward, next_state):
        """Operations to perform on each timestep during training."""
        state = state.to(self.device)
        #action = torch.tensor([action]).float().to(self.device)
        reward = torch.tensor([reward]).float().to(self.device)
        if self.random_mode and len(self.random_memory) >= self.P["random_replay_capacity"]: 
            self.random_mode = False
            print("Random data collection complete.")
        if next_state != None: 
            if self.random_mode: self.random_memory.add(state, action, reward, next_state)
            else: self.memory.add(state, action, reward, next_state)
        if self.total_t % self.P["steps_between_update"] == 0:
            loss = self.update_on_batch()
            if loss: self.ep_losses.append(loss)
        self.total_t += 1

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        if self.ep_losses: mean_loss = np.mean(self.ep_losses)
        else: mean_loss = 0.
        del self.ep_losses[:]
        return {"logs":{"model_loss": mean_loss, "random_mode": int(self.random_mode)}}

    def _model_rollout(self, state): 
        """Use model and reward function to generate and evaluate rollouts with random action selection.
        Then select the first action from the rollout with maximum return."""
        returns = []; first_actions = []
        for _ in range(self.P["num_rollouts"]):
            rollout_state, rollout_return = state[0].detach().clone().to(self.device), 0
            for t in range(self.P["rollout_horizon"]):
                rollout_action = self.action_space.sample() # Random action selection.
                if t == 0: first_actions.append(rollout_action)               
                rollout_state += self.model(torch.cat((rollout_state, torch.Tensor([rollout_action]).to(self.device))).to(self.device))
                rollout_return += (self.P["gamma"] ** t) * self.reward_function(rollout_state)                
            returns.append(rollout_return)
        return returns, first_actions