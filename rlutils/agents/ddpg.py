"""
DESCRIPTION
"""

from ._generic import Agent
from ..common.env_wrappers import NormaliseActionWrapper
from ..common.networks import SequentialNetwork
from ..common.memory import ReplayMemory
from ..common.exploration import OUNoise, UniformNoise

import numpy as np
import torch
import torch.nn.functional as F 


class DdpgAgent(Agent):
    def __init__(self, env, hyperparameters, net_code_pi=None, net_code_Q=None):
        assert type(env) == NormaliseActionWrapper, "Action space must be normalised for DDPG."
        Agent.__init__(self, env, hyperparameters)
        # Create pi and Q networks.
        if net_code_pi is None:
            if len(self.env.observation_space.shape) > 1: raise NotImplementedError()
            else: 
                net_code_pi = [(self.env.observation_space.shape[0], 256), "R", (256, 256), "R", (256, self.env.action_space.shape[0]), "T"]
                net_code_Q = [(self.env.observation_space.shape[0]+self.env.action_space.shape[0], 256), "R", (256, 256), "R", (256, 1)]
        self.pi = SequentialNetwork(code=net_code_pi, lr=self.P["lr_pi"]).to(self.device)
        self.pi_target = SequentialNetwork(code=net_code_pi, eval_only=True).to(self.device)
        self.pi_target.load_state_dict(self.pi.state_dict()) # Clone.
        self.Q, self.Q_target = self._make_Q(net_code_Q)
        if self.P["td3"]:
            # For TD3 we have two Q networks, each with their corresponding targets.
            self.Q2, self.Q2_target = self._make_Q(net_code_Q)
        # Create replay memory.
        self.memory = ReplayMemory(self.P["replay_capacity"], include_done=True) # TODO: element_with_done should be default for all algorithms.
        # Create noise process for exploration.
        if self.P["noise_params"][0] == "ou": self.noise = OUNoise(self.env.action_space, *self.P["noise_params"][1:])
        if self.P["noise_params"][0] == "un": self.noise = UniformNoise(self.env.action_space, *self.P["noise_params"][1:])
        # Tracking variables.   
        self.total_ep = 0 # Used for noise decay.
        self.total_t = 0 # Used for policy update frequency for TD3.
        self.ep_losses = []  
    
    def act(self, state, explore=True, do_extra=False):
        """Deterministic action selection plus additive noise."""
        action_greedy = self.pi(state).cpu().detach().numpy()[0]
        action = self.noise.get_action(action_greedy) if explore else action_greedy
        if do_extra:
            sa = _sa_concat(state, torch.tensor([action], device=self.device, dtype=torch.float))
            sa_greedy = _sa_concat(state, torch.tensor([action_greedy], device=self.device, dtype=torch.float)) if explore else sa
            extra = {"action_greedy":action_greedy, "Q":self.Q(sa).item(), "Q_greedy":self.Q(sa_greedy).item()}
            if self.P["td3"]: extra["Q2"] = self.Q2(sa).item(); extra["Q2_greedy"] = self.Q2(sa_greedy).item()
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
            # Identify nonterminal states (note that replay memory elements are initialised to None).
            nonterminal_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
            nonterminal_next_states = torch.cat([s for s in batch.next_state if s is not None])
            # Select a' using the target pi network.
            nonterminal_next_actions = self.pi_target(nonterminal_next_states)
            if self.P["td3"]:
                # For TD3 we add clipped noise to a' to reduce overfitting.
                noise = (torch.randn_like(nonterminal_next_actions) * self.P["td3_noise_std"]
                        ).clamp(-self.P["td3_noise_clip"], self.P["td3_noise_clip"])
                nonterminal_next_actions = (nonterminal_next_actions + noise).clamp(-1, 1)
            # Use target Q network to compute Q_target(s', a') for each nonterminal next state.    
            next_Q_values = torch.zeros(self.P["batch_size"], device=self.device)
            next_Q_values[nonterminal_mask] = self.Q_target(_sa_concat(nonterminal_next_states, nonterminal_next_actions.detach())).squeeze()
            if self.P["td3"]: 
                # For TD3 we use both target Q networks and take the minimum value.
                # This is the "clipped double Q trick".
                next_Q2_values = torch.zeros(self.P["batch_size"], device=self.device)
                next_Q2_values[nonterminal_mask] = self.Q2_target(_sa_concat(nonterminal_next_states, nonterminal_next_actions.detach())).squeeze()
                next_Q_values = torch.min(next_Q_values, next_Q2_values)        
            # Compute target = reward + discounted Q_target(s', a').
            Q_targets = rewards + (self.P["gamma"] * next_Q_values)
        # Update value in the direction of TD error. 
        Q_values = self.Q(_sa_concat(states, actions)).squeeze()
        value_loss = F.smooth_l1_loss(Q_values, Q_targets)
        self.Q.optimise(value_loss)
        if self.P["td3"]: 
            # For TD3, do the same for the second Q network.
            Q2_values = self.Q2(_sa_concat(states, actions)).squeeze()
            value2_loss = F.smooth_l1_loss(Q2_values, Q_targets)
            self.Q2.optimise(value2_loss)
        policy_loss = np.nan
        if (not self.P["td3"]) or (self.total_t % self.P["td3_policy_freq"] == 0): 
            # For TD3, only update policy and targets every N timesteps.
            # Update policy in the direction of increasing value according to self.Q (the policy gradient).
            policy_loss = -self.Q(_sa_concat(states, self.pi(states))).mean()
            self.pi.optimise(policy_loss)
            policy_loss = policy_loss.item()
        # Perform soft updates on targets.
        for target_param, param in zip(self.pi_target.parameters(), self.pi.parameters()):
            target_param.data.copy_(param.data * self.P["tau"] + target_param.data * (1.0 - self.P["tau"]))
        for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
            target_param.data.copy_(param.data * self.P["tau"] + target_param.data * (1.0 - self.P["tau"]))
        if self.P["td3"]:
            for target_param, param in zip(self.Q2_target.parameters(), self.Q2.parameters()):
                target_param.data.copy_(param.data * self.P["tau"] + target_param.data * (1.0 - self.P["tau"]))
        return policy_loss, value_loss.item() + value2_loss.item() if self.P["td3"] else value_loss.item()

    def per_timestep(self, state, action, reward, next_state, done, do_update=True):
        """Operations to perform on each timestep during training."""
        self.memory.add(state, torch.tensor([action], device=self.device, dtype=torch.float), torch.tensor([reward], device=self.device, dtype=torch.float), next_state, done)                
        if do_update:
            losses = self.update_on_batch()
            if losses: self.ep_losses.append(losses)
        self.total_t += 1

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        if self.ep_losses: mean_policy_loss, mean_value_loss = np.nanmean(self.ep_losses, axis=0)
        else: mean_policy_loss, mean_value_loss = 0., 0.
        self.noise.decay(self.total_ep)
        del self.ep_losses[:]; self.total_ep += 1
        return {"logs":{"policy_loss": mean_policy_loss, "value_loss": mean_value_loss, "sigma": self.noise.sigma}}

    def _make_Q(self, net_code_Q):
        """Create Q network and target."""
        # Action is an *input* to the Q network here.
        Q = SequentialNetwork(code=net_code_Q, lr=self.P["lr_Q"], clip_grads=True).to(self.device)
        Q_target = SequentialNetwork(code=net_code_Q, eval_only=True).to(self.device)
        Q_target.load_state_dict(Q.state_dict()) # Clone.
        return Q, Q_target

def _sa_concat(states, actions):
    """Concatenate states and actions into a single input vector for Q networks."""
    return torch.cat([states, actions], 1).float()