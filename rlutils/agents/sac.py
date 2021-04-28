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
        else: 
            net_code_pi = [(self.env.observation_space.shape[0], 256), "R", (256, 256), "R", (256, 2*self.env.action_space.shape[0])] # Mean and standard deviation.
            net_code_Q = [(self.env.observation_space.shape[0]+self.env.action_space.shape[0], 256), "R", (256, 256), "R", (256, 1)]
        self.pi = SequentialNetwork(code=net_code_pi, lr=self.P["lr_pi"]).to(self.device)
        self.Q, self.Q_target = self._make_Q(net_code_Q)
        self.Q2, self.Q2_target = self._make_Q(net_code_Q)
        # Create replay memory.
        self.memory = ReplayMemory(self.P["replay_capacity"]) 
        # Tracking variables.   
        self.ep_losses = []  
    
    def act(self, state, explore=True):
        """Probabilistic action selection from Gaussian parameterised by output of self.pi."""
        state = state.to(self.device)
        action, _ = self._pi_to_action_and_log_prob(self.pi(state))
        return action.cpu().detach().numpy()[0], {}

    def update_on_batch(self):
        """Use a random batch from the replay memory to update the pi and Q network parameters."""
        if len(self.memory) < self.P["batch_size"]: return
        # Sample a batch and transpose it (see https://stackoverflow.com/a/19343/3343043).
        batch = self.memory.element(*zip(*self.memory.sample(self.P["batch_size"])))
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        # Identify nonterminal states (note that replay memory elements are initialised to None).
        nonterminal_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        nonterminal_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(self.device)
        # Select a' using the current pi network.
        nonterminal_next_actions, nonterminal_next_log_probs = self._pi_to_action_and_log_prob(self.pi(nonterminal_next_states))
        # Use both target Q networks to compute Q_target(s', a') for each nonterminal next state and take the minimum value. 
        # This is the "clipped double Q trick".
        next_soft_Q_values = torch.zeros(self.P["batch_size"], device=self.device)
        next_soft_Q_values[nonterminal_mask] = torch.min(
            self.Q_target(_sa_concat(nonterminal_next_states, nonterminal_next_actions.detach())).squeeze(),
            self.Q2_target(_sa_concat(nonterminal_next_states, nonterminal_next_actions.detach())).squeeze())       
        # Subtract entropy term to create soft Q values.
        next_soft_Q_values[nonterminal_mask] -= self.P["alpha"] * nonterminal_next_log_probs
        # Compute target = reward + discounted soft Q_target(s', a').
        Q_targets = (rewards + (self.P["gamma"] * next_soft_Q_values)).detach()
        # Update value in the direction of entropy-regularised TD error. 
        Q_values = self.Q(_sa_concat(states, actions)).squeeze()
        value_loss = F.smooth_l1_loss(Q_values, Q_targets)
        self.Q.optimise(value_loss, retain_graph=True)
        Q2_values = self.Q2(_sa_concat(states, actions)).squeeze()
        value2_loss = F.smooth_l1_loss(Q2_values, Q_targets)
        self.Q2.optimise(value2_loss)
        # Re-evaluate actions using the current pi network and get their values using the current Q networks.
        actions_new, log_probs_new = self._pi_to_action_and_log_prob(self.pi(states))
        Q_values_new = torch.min( 
            self.Q(_sa_concat(states, actions_new.detach())).squeeze(),
            self.Q2(_sa_concat(states, actions_new.detach())).squeeze()) # Again use the clipped double Q trick.  
        # Update policy in the direction of increasing value according to self.Q (the policy gradient), plus entropy regularisation.
        policy_loss = ((self.P["alpha"] * log_probs_new) - Q_values_new).mean()
        self.pi.optimise(policy_loss)
        # Perform soft updates on targets.
        for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
            target_param.data.copy_(param.data * self.P["tau"] + target_param.data * (1.0 - self.P["tau"]))
        for target_param, param in zip(self.Q2_target.parameters(), self.Q2.parameters()):
            target_param.data.copy_(param.data * self.P["tau"] + target_param.data * (1.0 - self.P["tau"]))
        return policy_loss.item(), value_loss.item() + value2_loss.item()

    def per_timestep(self, state, action, reward, next_state):
        """Operations to perform on each timestep during training."""
        state = state.to(self.device)
        action = torch.tensor([action]).float().to(self.device)
        reward = torch.tensor([reward]).float().to(self.device)
        self.memory.add(state, action, reward, next_state)
        losses = self.update_on_batch()
        if losses: self.ep_losses.append(losses)

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        if self.ep_losses: mean_policy_loss, mean_value_loss = np.nanmean(self.ep_losses, axis=0)
        else: mean_policy_loss, mean_value_loss = 0., 0.
        del self.ep_losses[:]
        return {"logs":{"policy_loss": mean_policy_loss, "value_loss": mean_value_loss}}

    def _make_Q(self, net_code_Q):
        """Create Q network and target."""
        # Action is an *input* to the Q network here.
        Q = SequentialNetwork(code=net_code_Q, lr=self.P["lr_Q"], clip_grads=True).to(self.device)
        Q_target = SequentialNetwork(code=net_code_Q, eval_only=True).to(self.device)
        Q_target.load_state_dict(Q.state_dict()) # Clone.
        return Q, Q_target

    def _pi_to_action_and_log_prob(self, pi): 
        """SAC uses the output of self.pi as the mean and log standard deviation of a squashed Gaussian,
        then generates an action by sampling from that distribution."""
        mu, log_std = torch.split(pi, self.env.action_space.shape[0], dim=1)
        log_std = torch.clamp(log_std, -20, 2)
        gaussian = Normal(mu, torch.exp(log_std))
        action_unsquashed = gaussian.sample()
        action = torch.tanh(action_unsquashed)
        # Compute log_prob from Gaussian, then apply correction for Tanh squashing.
        log_prob = gaussian.log_prob(action_unsquashed).sum(axis=-1)
        log_prob -= (2 * (np.log(2) - action_unsquashed - F.softplus(-2 * action_unsquashed))).sum(axis=1)
        return action, log_prob

def _sa_concat(states, actions):
    """Concatenate states and actions into a single input vector for Q networks."""
    return torch.cat([states, actions], 1).float()