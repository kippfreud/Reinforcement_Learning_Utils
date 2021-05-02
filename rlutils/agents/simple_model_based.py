"""
Simple model-based agent for discrete and continuous action spaces.
Adapted from the model-based component of the architecture from:
    "Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning"
"""

from ._generic import Agent
from ..common.networks import SequentialNetwork
from ..common.memory import ReplayMemory

import numpy as np
import torch
import torch.nn.functional as F


class SimpleModelBasedAgent(Agent):
    def __init__(self, env, hyperparameters):
        assert "reward_function" in hyperparameters, "Need to provide reward function."
        Agent.__init__(self, env, hyperparameters)
        # Establish whether action space is discrete or continuous.
        self.continuous_actions = len(self.env.action_space.shape) > 0
        # Create model network.
        if len(self.env.observation_space.shape) > 1: raise NotImplementedError()
        else:
            self.state_dim = self.env.observation_space.shape[0]
            action_dim = self.env.action_space.shape[0] if self.continuous_actions else 1 
            net_code = [(self.state_dim+action_dim, 32), "R", (32, 64), "R", (64, self.state_dim)]
        self.model = SequentialNetwork(code=net_code, lr=self.P["lr_model"]).to(self.device)
        # Create replay memory in two components: one for random transitions one for on-policy transitions.
        self.random_memory = ReplayMemory(self.P["random_replay_capacity"]) 
        if not self.P["random_mode_only"]:
            self.memory = ReplayMemory(self.P["replay_capacity"]) 
            self.batch_split = (round(self.P["batch_size"] * self.P["batch_ratio"]), round(self.P["batch_size"] * (1-self.P["batch_ratio"])))
        # Tracking variables.
        self.random_mode = True
        self.total_t = 0 # Used for model_freq.
        self.ep_losses = []

    def act(self, state, explore=True, do_extra=False):
        """Either random or model-based action selection."""
        extra = {}
        if self.random_mode: action = self.env.action_space.sample()
        else: 
            returns, first_actions = self._model_rollout(state)
            best_rollout = np.argmax(returns)
            action = first_actions[best_rollout]
            if do_extra: extra["g_pred"] = returns[best_rollout]
        if do_extra: extra["next_state_pred"] = self.predict(state, action).numpy()
        return action, extra

    def predict(self, state, action):
        """Use model to predict the next state given a single state-action pair."""
        return state[0] + self.model(torch.cat((state[0], torch.Tensor(action if self.continuous_actions else [action]).to(self.device))).to(self.device)).detach()

    def update_on_batch(self):
        """Use a random batch from the replay memory to update the model network parameters."""

        # TODO: Batch normalisation of state dimensions.

        if self.random_mode: # During random mode, just sample from random memory.   
            if len(self.random_memory) < self.P["batch_size"]: return 
            batch = self.random_memory.sample(self.P["batch_size"])
        else: # After random mode, sample from both memories according to self.batch_split.
            if len(self.memory) < self.batch_split[0]: return 
            batch = list(self.memory.sample(self.batch_split[0])) + list(self.random_memory.sample(self.batch_split[1]))
        # Update model in the direction of the true change in state using MSE loss.
        states_and_actions = torch.cat(tuple(torch.cat((x.state, torch.Tensor([x.action]).to(self.device)), dim=-1) for x in batch), dim=0).to(self.device)
        next_states = torch.cat(tuple(x.next_state for x in batch)).to(self.device)
        target = next_states - states_and_actions[:,:self.state_dim]
        prediction = self.model(states_and_actions)
        loss = F.mse_loss(prediction, target)
        self.model.optimise(loss)
        return loss.item()

    def per_timestep(self, state, action, reward, next_state, done):
        """Operations to perform on each timestep during training."""
        state = state.to(self.device)
        # action = torch.tensor([action]).float().to(self.device)
        reward = torch.tensor([reward]).float().to(self.device)
        if not self.P["random_mode_only"] and self.random_mode and len(self.random_memory) >= self.P["random_replay_capacity"]: 
            self.random_mode = False
            print("Random data collection complete.")
        if not self.continuous_actions: action = [action]
        if self.random_mode: self.random_memory.add(state, action, reward, next_state)
        else: self.memory.add(state, action, reward, next_state)
        if self.total_t % self.P["model_freq"] == 0:
            loss = self.update_on_batch()
            if loss: self.ep_losses.append(loss); print(len(self.random_memory), loss)
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
                rollout_action = self.env.action_space.sample() # Random action selection.
                if t == 0: first_actions.append(rollout_action)               
                rollout_state = self.predict(rollout_state, rollout_action)
                rollout_return += (self.P["gamma"] ** t) * self.P["reward_function"](rollout_state, rollout_action)                
            returns.append(rollout_return)
        return returns, first_actions    