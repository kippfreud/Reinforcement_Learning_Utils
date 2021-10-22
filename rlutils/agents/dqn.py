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
    def __init__(self, env, hyperparameters):
        Agent.__init__(self, env, hyperparameters)
        # Create Q network.
        m = (1 if self.P["reward_components"] is None else self.P["reward_components"])
        if len(self.env.observation_space.shape) > 1: 
            net_preset = "CartPoleQ_Pixels"
            net_code, input_shape, output_size = None, self.env.observation_space.shape, self.env.action_space.n*m
        else: net_code, net_preset, input_shape, output_size = self.P["net_Q"], None, self.env.observation_space.shape[0], self.env.action_space.n*m
        self.Q = SequentialNetwork(code=net_code, preset=net_preset, input_shape=input_shape, output_size=output_size, lr=self.P["lr_Q"], clip_grads=True).to(self.device)
        self.Q_target = SequentialNetwork(code=net_code, preset=net_preset, input_shape=input_shape, output_size=output_size, eval_only=True).to(self.device)
        self.Q_target.load_state_dict(self.Q.state_dict()) # Clone.
        # Create replay memory.
        self.memory = ReplayMemory(self.P["replay_capacity"])
        # Initialise epsilon-greedy for exploration.
        # TODO: Move to exploration.py.
        self.epsilon = self.P["epsilon_start"]
        self.epsilon_decay_per_timestep = (self.P["epsilon_start"] - self.P["epsilon_end"]) / self.P["epsilon_decay"]
        # Tracking variables.
        if self.P["target_update"][0] == "hard": self.updates_since_target_clone = 0
        else: assert self.P["target_update"][0] == "soft"
        self.ep_losses = []

    def act(self, state, explore=True, do_extra=False):
        """Epsilon-greedy action selection."""
        Q = self.Q(state).reshape(self.env.action_space.n, -1)
        # If using reward decomposition, need to take sum.
        if self.P["reward_components"] is not None: 
            if self.reward is not None: Q *= self.reward.weights
            Q = Q.sum(axis=1).reshape(-1,1)
        # Assemble epsilon-greedy action distribution.
        greedy = Q.max(0)[1].view(1, 1)
        if explore: action_probs = torch.ones((1, self.env.action_space.n), device=self.device) * self.epsilon / self.env.action_space.n
        else:       action_probs = torch.zeros((1, self.env.action_space.n), device=self.device)
        action_probs[0,greedy] += (1-action_probs.sum())
        dist = Categorical(action_probs) # Categorical action distribution.
        action = dist.sample()
        if do_extra:
            extra = {"pi": action_probs.cpu().detach().numpy()[0], "Q": Q.cpu().detach().numpy()}
            if self.P["reward_components"] is not None and self.reward is not None: 
                extra["reward_components"] = (self.reward.phi(state, action)*self.reward.weights).cpu().detach().numpy()
        else: extra = {}
        return action.item(), extra

    def update_on_batch(self):
        """Use a random batch from the replay memory to update the Q network parameters."""
        if len(self.memory) < self.P["batch_size"]: return 
        # Sample a batch and transpose it (see https://stackoverflow.com/a/19343/3343043).
        batch = self.memory.element(*zip(*self.memory.sample(self.P["batch_size"])))
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        nonterminal_mask = ~torch.cat(batch.done)
        nonterminal_next_states = torch.cat(batch.next_state)[nonterminal_mask]
        # Use target network to compute Q_target(s', a') for each nonterminal next state.
        next_Q_values = torch.zeros((self.P["batch_size"], 1 if self.P["reward_components"] is None else self.P["reward_components"]), device=self.device)
        Q_t_n = self.Q_target(nonterminal_next_states)
        if self.P["reward_components"] is None: 
            # Compute Q(s, a) by running each s through self.Q, then selecting the corresponding column.
            Q_values = self.Q(states).gather(1, actions.reshape(-1,1))
            # In double DQN, a' is the Q-maximising action for self.Q. This decorrelation reduces overestimation bias.
            # In regular DQN, a' is the Q-maximising action for self.Q_target.
            nonterminal_next_actions = (self.Q(nonterminal_next_states) if self.P["double"] else Q_t_n).argmax(1).detach()
            Q_t_n = Q_t_n.unsqueeze(-1)
            rewards = rewards.unsqueeze(-1)
        else: 
            # Equivalent of above for decomposed reward.
            Q_values = self.Q(states).reshape(self.P["batch_size"], self.env.action_space.n, self.P["reward_components"])[torch.arange(self.P["batch_size"]), actions, :]
            Q_t_n = Q_t_n.reshape(Q_t_n.shape[0], self.env.action_space.n, self.P["reward_components"])
            Q_for_a_n = self.Q(nonterminal_next_states).reshape(*Q_t_n.shape) if self.P["double"] else Q_t_n
            if self.reward is not None: Q_for_a_n *= self.reward.weights; rewards = self.reward.phi(states, actions)
            nonterminal_next_actions = Q_for_a_n.sum(axis=2).argmax(1).detach()
        next_Q_values[nonterminal_mask] = Q_t_n[torch.arange(Q_t_n.shape[0]), nonterminal_next_actions, :]        
        # Compute target = reward + discounted Q_target(s', a').
        Q_targets = rewards + (self.P["gamma"] * next_Q_values)
        # Update value in the direction of TD error using Huber loss. 
        if self.P["reward_components"] is not None and self.reward is not None: 
            # NOTE: This creates a regular Bellman update. Disabling it prioritising learning all successor features equally.
            if True: Q_values *= self.reward.weights; Q_targets *= self.reward.weights 
        loss = F.smooth_l1_loss(Q_values, Q_targets)
        self.Q.optimise(loss)
        if self.P["target_update"][0] == "hard":
            # Perform periodic hard update on target.
            self.updates_since_target_clone += 1
            if self.updates_since_target_clone >= self.P["target_update"][1]:
                self.Q_target.load_state_dict(self.Q.state_dict())
                self.updates_since_target_clone = 0
        else: 
            # Perform soft (Polyak) update on target.
            for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
                target_param.data.copy_(param.data * self.P["target_update"][1] + target_param.data * (1.0 - self.P["target_update"][1]))
        return loss.item()

    def per_timestep(self, state, action, reward, next_state, done):
        """Operations to perform on each timestep during training."""
        self.memory.add(state, 
                        torch.tensor([action], device=self.device, dtype=torch.int64),
                        torch.tensor([reward], device=self.device, dtype=torch.float), 
                        next_state,
                        torch.tensor([done], device=self.device, dtype=torch.bool))
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