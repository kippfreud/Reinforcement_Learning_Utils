from ._generic import Agent
from ..common.networks import SequentialNetwork
from ..common.memory import ReplayMemory
from ..common.exploration import squashed_gaussian
from ..common.utils import col_concat

import numpy as np
import torch
import torch.nn.functional as F 


class SacAgent(Agent):
    def __init__(self, env, hyperparameters):
        """
        Soft actor-critic (SAC) agent for continuous action spaces.
        """
        Agent.__init__(self, env, hyperparameters)
        # NOTE:If the DIAYN algorithm is wrapped around SAC, the observation space is augmented by a one-hot skill vector.
        if "aug_obs_shape" in self.P: obs_shape = self.P["aug_obs_shape"]
        else: obs_shape = self.env.observation_space.shape
        # Create pi and Q networks.
        if len(obs_shape) > 1: raise NotImplementedError()
        if self.P["input_normaliser"] == "obs_lims": # TODO: Move this into _generic
            norm_pi = (self.env.observation_space,)
            norm_Q = (self.env.observation_space, self.env.action_space)
        else: norm_pi, norm_Q = None, None
        # Policy outputs mean and standard deviation.
        self.pi = SequentialNetwork(code=self.P["net_pi"], input_shape=obs_shape[0], output_size=2*self.env.action_space.shape[0], normaliser=norm_pi, lr=self.P["lr_pi"]).to(self.device)
        self.Q, self.Q_target = [], []
        for _ in range(2): # We have two Q networks, each with their corresponding targets.
            # Action is an *input* to the Q network here.
            Q = SequentialNetwork(code=self.P["net_Q"], input_shape=obs_shape[0]+self.env.action_space.shape[0], output_size=1, normaliser=norm_Q, lr=self.P["lr_Q"], clip_grads=True).to(self.device)
            Q_target = SequentialNetwork(code=self.P["net_Q"], input_shape=obs_shape[0]+self.env.action_space.shape[0], output_size=1, normaliser=norm_Q, eval_only=True).to(self.device)
            Q_target.load_state_dict(Q.state_dict()) # Clone.
            self.Q.append(Q); self.Q_target.append(Q_target)
        self.start()

    def start(self):
        # Create replay memory.
        self.memory = ReplayMemory(self.P["replay_capacity"])
        # Tracking variables.   
        self.total_t = 0 # Used for update_freq.
        self.ep_losses = []  
    
    def act(self, state, explore=True, do_extra=False):
        """Probabilistic action selection from Gaussian parameterised by output of self.pi."""
        action, log_prob = squashed_gaussian(self.pi(state))
        return action.cpu().detach().numpy()[0], {}

    def update_on_batch(self, diayn_batch=None):
        """Use a random batch from the replay memory to update the pi and Q network parameters."""
        # If the DIAYN algorithm is wrapped around SAC, the batch will be given.
        states, actions, rewards, nonterminal_mask, nonterminal_next_states = self.memory.sample(self.P["batch_size"]) if diayn_batch is None else diayn_batch
        if states is None: return 
        # Select a' using the current pi network.
        nonterminal_next_actions, nonterminal_next_log_probs = squashed_gaussian(self.pi(nonterminal_next_states))
        # Use target Q networks to compute Q_target(s', a') for each nonterminal next state and take the minimum value. This is the "clipped double Q trick".
        next_Q_values = torch.zeros(self.P["batch_size"], device=self.device)
        next_Q_values[nonterminal_mask] = torch.min(*(Q_target(col_concat(nonterminal_next_states, nonterminal_next_actions)) for Q_target in self.Q_target)).squeeze()       
        # Subtract entropy term, creating soft Q values.
        next_Q_values[nonterminal_mask] -= self.P["alpha"] * nonterminal_next_log_probs
        # Compute target = reward + discounted soft Q_target(s', a').
        Q_targets = (rewards + (self.P["gamma"] * next_Q_values)).detach()
        value_loss_sum = 0.
        for Q in self.Q:    
            # Update value in the direction of entropy-regularised TD error using Huber loss. 
            value_loss = F.smooth_l1_loss(Q(col_concat(states, actions)).squeeze(), Q_targets)
            Q.optimise(value_loss)
            value_loss_sum += value_loss.item()
        # Re-evaluate actions using the current pi network and get their values using the current Q networks. Again use the clipped double Q trick. 
        actions_new, log_probs_new = squashed_gaussian(self.pi(states))
        Q_values_new = torch.min(*(Q(col_concat(states, actions_new)) for Q in self.Q))
        # Update policy in the direction of increasing value according to self.Q (the policy gradient), plus entropy regularisation.
        policy_loss = ((self.P["alpha"] * log_probs_new) - Q_values_new).mean()
        self.pi.optimise(policy_loss)
        # Perform soft (Polyak) updates on targets.
        for net, target in zip(self.Q, self.Q_target): target.polyak(net, tau=self.P["tau"])
        return policy_loss.item(), value_loss_sum

    def per_timestep(self, state, action, reward, next_state, done):
        """Operations to perform on each timestep during training."""
        self.memory.add(state, action, reward, next_state, done)  
        self.total_t += 1
        if self.total_t % self.P["update_freq"] == 0:
            losses = self.update_on_batch()
            if losses: self.ep_losses.append(losses)

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        if self.ep_losses: mean_policy_loss, mean_value_loss = np.nanmean(self.ep_losses, axis=0)
        else: mean_policy_loss, mean_value_loss = 0., 0.
        del self.ep_losses[:]
        return {"logs":{"policy_loss": mean_policy_loss, "value_loss": mean_value_loss}}