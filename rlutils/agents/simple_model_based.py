from ._generic import Agent
from ..common.networks import SequentialNetwork
from ..common.memory import ReplayMemory
from ..common.utils import col_concat

import numpy as np
import torch
import torch.nn.functional as F


class SimpleModelBasedAgent(Agent):
    def __init__(self, env, hyperparameters):
        """
        Simple model-based agent for both discrete and continuous action spaces.
        Adapted from the model-based component of the architecture from:
            "Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning"
        TODO: With discrete actions, better to have per-action output nodes rather than providing action integer as an input.
        """
        assert "reward" in hyperparameters, f"{type(self).__name__} requires a reward function."
        Agent.__init__(self, env, hyperparameters)
        # Establish whether action space is continuous or discrete.
        self.continuous_actions = len(self.env.action_space.shape) > 0
        # Create model network.
        if len(self.env.observation_space.shape) > 1: raise NotImplementedError()
        else: self.state_dim, action_dim = self.env.observation_space.shape[0], (self.env.action_space.shape[0] if self.continuous_actions else 1) 
        self.model = SequentialNetwork(code=self.P["net_model"], input_shape=self.state_dim+action_dim, output_size=self.state_dim, lr=self.P["lr_model"]).to(self.device)
        # Create replay memory in two components: one for random transitions one for on-policy transitions.
        self.random_memory = ReplayMemory(self.P["num_random_steps"])
        if not self.P["random_mode_only"]:
            self.memory = ReplayMemory(self.P["replay_capacity"])
            self.batch_split = (round(self.P["batch_size"] * self.P["batch_ratio"]), round(self.P["batch_size"] * (1-self.P["batch_ratio"])))
        # Tracking variables.
        self.random_mode = True
        self.total_t = 0 # Used for model_freq.
        self.ep_losses = []

    def act(self, state, explore=True, do_extra=False):
        """Either random or model-based action selection."""
        with torch.no_grad():
            extra = {}
            if self.random_mode: action = torch.tensor([self.env.action_space.sample()])
            else: 
                returns, first_actions = self.rollout(state)
                best_rollout = np.argmax(returns)
                action = first_actions[best_rollout]
                if do_extra: extra["g_pred"] = returns[best_rollout]
            if do_extra: extra["next_state_pred"] = self.predict(state, action)[0].numpy()
            return action[0].numpy() if self.continuous_actions else action.item(), extra

    def predict(self, states, actions):
        """Use model to predict the next state given a single state-action pair."""
        return states + self.model(col_concat(states, actions.unsqueeze(1) if len(actions.shape) == 1 else actions))

    def update_on_batch(self):
        """Use a random batch from the replay memory to update the model network parameters."""
        # TODO: Batch normalisation of state dimensions.
        if self.random_mode: # During random mode, just sample from random memory.   
            states, actions, _, _, next_states = self.random_memory.sample(self.P["batch_size"], keep_terminal_next=True)
            if states is None: return 
        else: # After random mode, sample from both memories according to self.batch_split.
            states, actions, _, _, next_states = self.memory.sample(self.batch_split[0], keep_terminal_next=True)
            if states is None: return 
            states_r, actions_r, _, _, next_states_r = self.random_memory.sample(self.batch_split[1], keep_terminal_next=True)
            assert states_r is not None, "Random mode not long enough!"
            states = torch.cat((states, states_r), dim=0)
            actions = torch.cat((actions, actions_r), dim=0)
            next_states = torch.cat((next_states, next_states_r), dim=0)        
        # Update model in the direction of the true state derivatives using MSE loss.
        loss = F.mse_loss(self.predict(states, actions), next_states)
        self.model.optimise(loss)
        return loss.item()

    def per_timestep(self, state, action, reward, next_state, done):
        """Operations to perform on each timestep during training."""
        if not self.P["random_mode_only"] and self.random_mode and len(self.random_memory) >= self.P["num_random_steps"]: 
            self.random_mode = False
            print("Random data collection complete.")
        if self.random_mode: self.random_memory.add(state, action, reward, next_state, done)
        else: self.memory.add(state, action, reward, next_state, done)
        if self.total_t % self.P["model_freq"] == 0:
            loss = self.update_on_batch()
            if loss: self.ep_losses.append(loss)
        self.total_t += 1

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        if self.ep_losses: mean_loss = np.mean(self.ep_losses)
        else: mean_loss = 0.
        del self.ep_losses[:]
        return {"logs":{"model_loss": mean_loss, "random_mode": int(self.random_mode)}}

    def rollout(self, state): 
        """Use model and reward function to generate and evaluate rollouts with random action selection.
        Then select the first action from the rollout with maximum return."""
        returns = []; first_actions = []
        for _ in range(self.P["num_rollouts"]):
            rollout_state, rollout_return = state.detach().clone().to(self.device), 0
            for t in range(self.P["rollout_horizon"]):
                rollout_action = torch.tensor([self.env.action_space.sample()]) # Random action selection.
                if t == 0: first_actions.append(rollout_action)       
                rollout_state = self.predict(rollout_state, rollout_action)
                rollout_return += (self.P["gamma"] ** t) * self.P["reward"](rollout_state, rollout_action)           
            returns.append(rollout_return)
        return returns, first_actions    