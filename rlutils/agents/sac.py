"""
DESCRIPTION
"""

from ._generic import Agent
from ..common.networks import SequentialNetwork
from ..common.memory import ReplayMemory

import numpy as np
import torch
import torch.nn.functional as F 
from torch.distributions.normal import Normal


class SacAgent(Agent):
    def __init__(self, env, hyperparameters):
        Agent.__init__(self, env, hyperparameters)
        # Create pi and Q networks.
        if len(self.env.observation_space.shape) > 1: raise NotImplementedError()
        # Policy outputs mean and standard deviation.
        self.pi = SequentialNetwork(code=self.P["net_pi"], input_shape=self.env.observation_space.shape[0], output_size=2*self.env.action_space.shape[0], lr=self.P["lr_pi"]).to(self.device)
        self.Q, self.Q_target = [], []
        for _ in range(2): # We have two Q networks, each with their corresponding targets.
            # Action is an *input* to the Q network here.
            Q = SequentialNetwork(code=self.P["net_Q"], input_shape=self.env.observation_space.shape[0]+self.env.action_space.shape[0], output_size=1, lr=self.P["lr_Q"], clip_grads=True).to(self.device)
            Q_target = SequentialNetwork(code=self.P["net_Q"], input_shape=self.env.observation_space.shape[0]+self.env.action_space.shape[0], output_size=1, eval_only=True).to(self.device)
            Q_target.load_state_dict(Q.state_dict()) # Clone.
            self.Q.append(Q); self.Q_target.append(Q_target)
        # Create replay memory.
        self.memory = ReplayMemory(self.P["replay_capacity"])
        # Tracking variables.   
        self.ep_losses = []  
    
    def act(self, state, explore=True, do_extra=False):
        """Probabilistic action selection from Gaussian parameterised by output of self.pi."""
        action, log_prob = self._pi_to_action_and_log_prob(self.pi(state))
        return action.cpu().detach().numpy()[0], {}

    def update_on_batch(self):
        """Use a random batch from the replay memory to update the pi and Q network parameters."""
        if len(self.memory) < self.P["batch_size"]: return
        # Sample a batch and transpose it (see https://stackoverflow.com/a/19343/3343043).
        batch = self.memory.element(*zip(*self.memory.sample(self.P["batch_size"])))
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        nonterminal_mask = ~torch.cat(batch.done)
        nonterminal_next_states = torch.cat(batch.next_state)[nonterminal_mask]
        # Select a' using the current pi network.
        nonterminal_next_actions, nonterminal_next_log_probs = self._pi_to_action_and_log_prob(self.pi(nonterminal_next_states))
        # Use target Q networks to compute Q_target(s', a') for each nonterminal next state and take the minimum value. This is the "clipped double Q trick".
        next_Q_values = torch.zeros(self.P["batch_size"], device=self.device)
        next_Q_values[nonterminal_mask] = torch.min(*(Q_target(_sa_concat(nonterminal_next_states, nonterminal_next_actions)) for Q_target in self.Q_target)).squeeze()       
        # Subtract entropy term, creating soft Q values.
        next_Q_values[nonterminal_mask] -= self.P["alpha"] * nonterminal_next_log_probs
        # Compute target = reward + discounted soft Q_target(s', a').
        Q_targets = (rewards + (self.P["gamma"] * next_Q_values)).detach()
        value_loss_sum = 0.
        for Q in self.Q:    
            # Update value in the direction of entropy-regularised TD error using Huber loss. 
            value_loss = F.smooth_l1_loss(Q(_sa_concat(states, actions)).squeeze(), Q_targets)
            Q.optimise(value_loss)
            value_loss_sum += value_loss.item()
        # Re-evaluate actions using the current pi network and get their values using the current Q networks. Again use the clipped double Q trick. 
        actions_new, log_probs_new = self._pi_to_action_and_log_prob(self.pi(states))
        Q_values_new = torch.min(*(Q(_sa_concat(states, actions_new)) for Q in self.Q))
        # Update policy in the direction of increasing value according to self.Q (the policy gradient), plus entropy regularisation.
        policy_loss = ((self.P["alpha"] * log_probs_new) - Q_values_new).mean()
        self.pi.optimise(policy_loss)
        # Perform soft (Polyak) updates on targets.
        for net, target in zip(self.Q, self.Q_target):
            for param, target_param in zip(net.parameters(), target.parameters()):
                target_param.data.copy_(param.data * self.P["tau"] + target_param.data * (1.0 - self.P["tau"]))
        return policy_loss.item(), value_loss_sum

    def per_timestep(self, state, action, reward, next_state, done):
        """Operations to perform on each timestep during training."""
        self.memory.add(state, 
                        torch.tensor([action], device=self.device, dtype=torch.float), 
                        torch.tensor([reward], device=self.device, dtype=torch.float), 
                        next_state, 
                        torch.tensor([done], device=self.device, dtype=torch.bool))
        losses = self.update_on_batch()
        if losses: self.ep_losses.append(losses)

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        if self.ep_losses: mean_policy_loss, mean_value_loss = np.nanmean(self.ep_losses, axis=0)
        else: mean_policy_loss, mean_value_loss = 0., 0.
        del self.ep_losses[:]
        return {"logs":{"policy_loss": mean_policy_loss, "value_loss": mean_value_loss}}

    def _pi_to_action_and_log_prob(self, pi): 
        """SAC uses the output of self.pi as the mean and log standard deviation of a squashed Gaussian,
        then generates an action by sampling from that distribution."""
        mu, log_std = torch.split(pi, self.env.action_space.shape[0], dim=1)
        log_std = torch.clamp(log_std, -20, 2)
        gaussian = Normal(mu, torch.exp(log_std))
        action_unsquashed = gaussian.rsample() # rsample() required to allow differentiation.
        action = torch.tanh(action_unsquashed)
        # Compute log_prob from Gaussian, then apply correction for Tanh squashing.
        log_prob = gaussian.log_prob(action_unsquashed).sum(axis=-1)
        log_prob -= (2 * (np.log(2) - action_unsquashed - F.softplus(-2 * action_unsquashed))).sum(axis=1)
        return action, log_prob

def _sa_concat(states, actions):
    """Concatenate states and actions into a single input vector for Q networks."""
    return torch.cat([states, actions], 1).float()