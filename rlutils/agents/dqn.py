"""
DESCRIPTION
"""

from ._generic import Agent
from ..common.networks import SequentialNetwork
from ..common.memory import ReplayMemory

import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F


class DqnAgent(Agent):
    def __init__(self, env, hyperparameters, net_code=None):
        Agent.__init__(self, env, hyperparameters)
        # Create Q network.
        if net_code is None:
            if len(self.env.observation_space.shape) > 1: 
                net_preset = "CartPoleQ_Pixels"
                net_code, input_shape, output_size = None, self.env.observation_space.shape, self.env.action_space.n*self.P["reward_components"]
            else: 
                # From https://github.com/transedward/pytorch-dqn/blob/master/dqn_model.py.
                net_code = [(self.env.observation_space.shape[0], 256), "R", (256, 128), "R", (128, 64), "R", (64, self.env.action_space.n*self.P["reward_components"])]
                net_preset, input_shape, output_size = None, None, None
        self.Q = SequentialNetwork(code=net_code, preset=net_preset, input_shape=input_shape, output_size=output_size, lr=self.P["lr_Q"], clip_grads=True).to(self.device)
        self.Q_target = SequentialNetwork(code=net_code, preset=net_preset, input_shape=input_shape, output_size=output_size, eval_only=True).to(self.device)
        self.Q_target.load_state_dict(self.Q.state_dict()) # Clone.
        # Create replay memory.
        self.memory = ReplayMemory(self.P["replay_capacity"]) 
        # Tracking variables.
        self.epsilon = self.P["epsilon_start"]
        # self.total_t = 0 # Used for epsilon decay.
        self.epsilon_decay_per_timestep = (self.P["epsilon_start"] - self.P["epsilon_end"]) / self.P["epsilon_decay"]
        self.updates_since_target_clone = 0
        self.ep_losses = []

    def act(self, state, explore=True):
        """Epsilon-greedy action selection."""
        Q = self.Q(state).reshape(self.env.action_space.n,-1)
        # If using decomposed rewards, need to take sum.
        #
        # ===================
        #
        if self.P["reward_components"] > 1: Q = Q.sum(axis=1).reshape(-1,1)
        #
        # ===================
        #
        # Assemble epsilon-greedy action distribution.
        greedy = Q.max(0)[1].view(1, 1)
        if explore: action_probs = torch.ones((1, self.env.action_space.n), device=self.device) * self.epsilon / self.env.action_space.n
        else:       action_probs = torch.zeros((1, self.env.action_space.n), device=self.device)
        action_probs[0,greedy] += (1-action_probs.sum())
        dist = Categorical(action_probs) # Categorical action distribution.
        action = dist.sample()
        return action, {"pi": action_probs.cpu().detach().numpy()[0], "Q": Q.cpu().detach().numpy()}

    def update_on_batch(self):
        """Use a random batch from the replay memory to update the Q network parameters."""
        if len(self.memory) < self.P["batch_size"]: return 
        # Sample a batch and transpose it (see https://stackoverflow.com/a/19343/3343043).
        batch = self.memory.element(*zip(*self.memory.sample(self.P["batch_size"])))
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        # Compute Q(s, a) by running each s through self.Q, then selecting the corresponding column.
        #
        # ===================
        #
        if self.P["reward_components"] > 1: 
            Q_values = self.Q(states).reshape(self.P["batch_size"], self.env.action_space.n, -1)[torch.arange(self.P["batch_size"]), actions, :]
        #
        # ===================
        #
        else: Q_values = self.Q(states).gather(1, actions.reshape(-1,1)).squeeze()
        # Identify nonterminal states (note that replay memory elements are initialised to None).
        nonterminal_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        nonterminal_next_states = torch.cat([s for s in batch.next_state if s is not None])
        # Use target network to compute Q_target(s', a') for each nonterminal next state.
        # a' is chosen to be the maximising action from s'.
        next_Q_values = torch.zeros((self.P["batch_size"], self.P["reward_components"]), device=self.device)
        # 
        # ===================
        #
        Q_t_n = self.Q_target(nonterminal_next_states)
        if self.P["reward_components"] > 1: 
            Q_t_n = Q_t_n.reshape(Q_t_n.shape[0], self.env.action_space.n, -1)
            actions_next = Q_t_n.sum(axis=2).max(1)[1].detach()
        else: 
            actions_next = Q_t_n.max(1)[1].detach()
            Q_t_n = Q_t_n.unsqueeze(-1)
            rewards = rewards.unsqueeze(-1)
            Q_values = Q_values.unsqueeze(-1)
        #
        # ===================
        #
        next_Q_values[nonterminal_mask] = Q_t_n[torch.arange(Q_t_n.shape[0]), actions_next, :]        
        # Compute target = reward + discounted Q_target(s', a').
        Q_targets = rewards + (self.P["gamma"] * next_Q_values)
        # Update value in the direction of TD error using Huber loss. 
        # See https://en.wikipedia.org/wiki/Huber_loss.
        loss = F.smooth_l1_loss(Q_values, Q_targets)
        self.Q.optimise(loss)
        # Periodically clone target.
        self.updates_since_target_clone += 1
        if self.updates_since_target_clone >= self.P["updates_between_target_clone"]:
            self.Q_target.load_state_dict(self.Q.state_dict())
            self.updates_since_target_clone = 0
        return loss.item()

    def per_timestep(self, state, action, reward, next_state):
        """Operations to perform on each timestep during training."""
        self.memory.add(state, action, torch.tensor([reward], device=self.device, dtype=torch.float), next_state)
        loss = self.update_on_batch()
        if loss: self.ep_losses.append(loss)
        # Decay epsilon linearly as per Nature paper.
        self.epsilon = max(self.epsilon - self.epsilon_decay_per_timestep, self.P["epsilon_end"])

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        if self.ep_losses: mean_loss = np.mean(self.ep_losses)
        else: mean_loss = 0.
        del self.ep_losses[:]
        return {"logs":{"value_loss": mean_loss, "epsilon": self.epsilon}}