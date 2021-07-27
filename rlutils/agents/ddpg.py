"""
DESCRIPTION

TODO: Make TD3 inherit from DDPG, rather than complicating this class with if statements.
"""

from ._generic import Agent
from ..common.networks import SequentialNetwork
from ..common.memory import ReplayMemory
from ..common.exploration import OUNoise, UniformNoise
from ..common.utils import col_concat

import numpy as np
import torch
import torch.nn.functional as F 


class DdpgAgent(Agent):
    def __init__(self, env, hyperparameters):
        Agent.__init__(self, env, hyperparameters)
        # Create pi and Q networks.
        if len(self.env.observation_space.shape) > 1: raise NotImplementedError()
        self.pi = SequentialNetwork(code=self.P["net_pi"], input_shape=self.env.observation_space.shape[0], output_size=self.env.action_space.shape[0], lr=self.P["lr_pi"]).to(self.device)
        self.pi_target = SequentialNetwork(code=self.P["net_pi"], input_shape=self.env.observation_space.shape[0], output_size=self.env.action_space.shape[0], eval_only=True).to(self.device)
        self.pi_target.load_state_dict(self.pi.state_dict()) # Clone.
        self.Q, self.Q_target = [], []
        for _ in range(2 if self.P["td3"] else 1): # For TD3 we have two Q networks, each with their corresponding targets.
            # Action is an *input* to the Q network here.
            Q = SequentialNetwork(code=self.P["net_Q"], input_shape=self.env.observation_space.shape[0]+self.env.action_space.shape[0], output_size=1, lr=self.P["lr_Q"], clip_grads=True).to(self.device)
            Q_target = SequentialNetwork(code=self.P["net_Q"], input_shape=self.env.observation_space.shape[0]+self.env.action_space.shape[0], output_size=1, eval_only=True).to(self.device)
            Q_target.load_state_dict(Q.state_dict()) # Clone.
            self.Q.append(Q); self.Q_target.append(Q_target)
        # Create replay memory.
        self.memory = ReplayMemory(self.P["replay_capacity"])
        # Create noise process for exploration.
        if self.P["noise_params"][0] == "ou": self.noise = OUNoise(self.env.action_space, *self.P["noise_params"][1:])
        elif self.P["noise_params"][0] == "un": self.noise = UniformNoise(self.env.action_space, *self.P["noise_params"][1:])
        else: raise Exception()
        # Tracking variables.   
        self.total_ep = 0 # Used for noise decay.
        self.total_t = 0 # Used for policy update frequency for TD3.
        self.ep_losses = []  
    
    def act(self, state, explore=True, do_extra=False):
        """Deterministic action selection plus additive noise."""
        action_greedy = self.pi(state).cpu().detach().numpy()[0]
        action = self.noise.get_action(action_greedy) if explore else action_greedy
        if do_extra:
            sa = col_concat(state, torch.tensor([action], device=self.device, dtype=torch.float))
            sa_greedy = col_concat(state, torch.tensor([action_greedy], device=self.device, dtype=torch.float)) if explore else sa
            extra = {"action_greedy": action_greedy}
            for i, Q in zip(["", "2"], self.Q):
                extra[f"Q{i}"] = Q(sa).item(); extra[f"Q{i}_greedy"] = Q(sa_greedy).item()
        else: extra = {}       
        return action, extra 

    def update_on_batch(self, states=None, actions=None, Q_targets=None):
        """Use a random batch from the replay memory to update the pi and Q network parameters.
        If the STEVE algorithm is wrapped around DDPG, states, actions and Q_targets will be given."""
        if states is None:
            if len(self.memory) < self.P["batch_size"]: return
            # Sample a batch and transpose it (see https://stackoverflow.com/a/19343/3343043).
            batch = self.memory.element(*zip(*self.memory.sample(self.P["batch_size"])))
            states = torch.cat(batch.state)
            actions = torch.cat(batch.action)
            rewards = torch.cat(batch.reward)
            nonterminal_mask = ~torch.cat(batch.done)
            nonterminal_next_states = torch.cat(batch.next_state)[nonterminal_mask]
            # Select a' using the target pi network.
            nonterminal_next_actions = self.pi_target(nonterminal_next_states)
            if self.P["td3"]:
                # For TD3 we add clipped noise to a' to reduce overfitting.
                noise = (torch.randn_like(nonterminal_next_actions) * self.P["td3_noise_std"]
                        ).clamp(-self.P["td3_noise_clip"], self.P["td3_noise_clip"])
                nonterminal_next_actions = (nonterminal_next_actions + noise).clamp(-1, 1)
            # Use target Q networks to compute Q_target(s', a') for each nonterminal next state and take the minimum value. This is the "clipped double Q trick".
            next_Q_values = torch.zeros(self.P["batch_size"], device=self.device)
            next_Q_values[nonterminal_mask] = torch.min(*(Q_target(col_concat(nonterminal_next_states, nonterminal_next_actions)) for Q_target in self.Q_target)).squeeze()       
            # Compute target = reward + discounted Q_target(s', a').
            Q_targets = (rewards + (self.P["gamma"] * next_Q_values)).detach()
        value_loss_sum = 0.
        for Q in self.Q:    
            # Update value in the direction of TD error using Huber loss. 
            value_loss = F.smooth_l1_loss(Q(col_concat(states, actions)).squeeze(), Q_targets)
            Q.optimise(value_loss)
            value_loss_sum += value_loss.item()
        policy_loss = np.nan
        if (not self.P["td3"]) or (self.total_t % self.P["td3_policy_freq"] == 0): # For TD3, only update policy and targets every N timesteps.
            # Update policy in the direction of increasing value according to self.Q (the policy gradient).
            # This means that the policy function approximates the argmax() operation used in Q learning for discrete action spaces.
            policy_loss = -self.Q[0](col_concat(states, self.pi(states))).mean() # NOTE: Using first Q network only.
            self.pi.optimise(policy_loss)
            policy_loss = policy_loss.item()
        # Perform soft (Polyak) updates on targets.
        for net, target in zip([self.pi]+self.Q, [self.pi_target]+self.Q_target):
            for param, target_param in zip(net.parameters(), target.parameters()):
                target_param.data.copy_(param.data * self.P["tau"] + target_param.data * (1.0 - self.P["tau"]))
        return policy_loss, value_loss_sum

    def per_timestep(self, state, action, reward, next_state, done, suppress_update=False):
        """Operations to perform on each timestep during training."""
        self.memory.add(state, 
                        torch.tensor([action], device=self.device, dtype=torch.float), 
                        torch.tensor([reward], device=self.device, dtype=torch.float), 
                        next_state, 
                        torch.tensor([done], device=self.device, dtype=torch.bool))                
        if not suppress_update:
            losses = self.update_on_batch()
            if losses: self.ep_losses.append(losses)
        self.total_t += 1

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        if self.ep_losses: mean_policy_loss, mean_value_loss = np.nanmean(self.ep_losses, axis=0)
        else: mean_policy_loss, mean_value_loss = 0., 0.
        del self.ep_losses[:]; self.total_ep += 1
        self.noise.decay(self.total_ep)
        return {"logs":{"policy_loss": mean_policy_loss, "value_loss": mean_value_loss, "sigma": self.noise.sigma}}