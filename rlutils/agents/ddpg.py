from ..common.networks import SequentialNetwork
from ..common.memory import ReplayMemory
from ..common.exploration import OUNoise

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F 


DEFAULT_HYPERPARAMETERS = {
    "replay_capacity": 10000,
    "batch_size": 256,
    "lr_pi": 1e-4,
    "lr_Q": 1e-3,
    "gamma": 0.99,
    "tau": 0.01,
    "noise_params": (0., 0.15, 0.3, 0.3, 300000),
    "td3": True,
    "td3_noise_std": 0.2,
    "td3_noise_clip": 0.5,
    "td3_policy_freq": 2
}


class DdpgAgent:
    def __init__(self, 
                 state_shape,
                 action_space, 
                 hyperparameters=DEFAULT_HYPERPARAMETERS
                 ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.P = hyperparameters 
        # Create pi and Q networks.
        if len(state_shape) > 1: raise NotImplementedError()
        else: preset_pi, preset_Q = "PendulumPi_Vector", "PendulumQ_Vector"
        num_actions = action_space.shape[0]
        self.pi = SequentialNetwork(preset=preset_pi, input_shape=state_shape, output_size=num_actions).to(self.device)
        self.optimiser_pi = optim.Adam(self.pi.parameters(), lr=self.P["lr_pi"])
        self.pi_target = SequentialNetwork(preset=preset_pi, input_shape=state_shape, output_size=num_actions).to(self.device)
        self.pi_target.load_state_dict(self.pi.state_dict()) # Clone.
        self.pi_target.eval() # Turn off training mode for target net.
        self.Q, self.Q_target, self.optimiser_Q = self._make_Q(preset_Q, state_shape, num_actions)
        if self.P["td3"]:
            # For TD3 we have two Q networks, each with their corresponding targets.
            self.Q2, self.Q2_target, self.optimiser_Q2 = self._make_Q(preset_Q, state_shape, num_actions)
        # Create replay memory.
        self.memory = ReplayMemory(self.P["replay_capacity"]) 
        # Create noise process for exploration.
        self.noise = OUNoise(action_space, *self.P["noise_params"])
        # Tracking variables.   
        self.total_t = 0 # Used for noise decay (and policy update frequency for TD3).
        self.ep_losses = []  
    
    def _make_Q(self, preset_Q, state_shape, num_actions):
        """Create Q network, optimiser and target."""
        # Action is an *input* to the Q network here.
        Q = SequentialNetwork(preset=preset_Q, input_shape=(state_shape[0]+num_actions,), output_size=1).to(self.device)
        optimiser_Q = optim.Adam(Q.parameters(), lr=self.P["lr_Q"])
        Q_target = SequentialNetwork(preset=preset_Q, input_shape=(state_shape[0]+num_actions,), output_size=1).to(self.device)
        Q_target.load_state_dict(Q.state_dict()) # Clone.
        Q_target.eval()
        return Q, Q_target, optimiser_Q
    
    def act(self, state, explore=True):
        """Deterministic action selection plus additive noise."""
        action_greedy = self.pi(state).detach().numpy()[0]
        if explore: 
            action = self.noise.get_action(action_greedy, self.total_t)
            # Return greedy action and Q values in extra.
            sa = self.sa_concat(state, torch.FloatTensor([action], device=self.device))
            sa_greedy = self.sa_concat(state, torch.FloatTensor([action_greedy], device=self.device))
            extra = {"action_greedy":action_greedy, "Q":self.Q(sa).item(), "Q_greedy":self.Q(sa_greedy).item()}
            if self.P["td3"]:
                extra["Q2"] = self.Q2(sa).item(); extra["Q2_greedy"] = self.Q2(sa_greedy).item()
            return action, extra
        else: return action_greedy, {} 

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
        nonterminal_next_states = torch.cat([s for s in batch.next_state if s is not None])
        # Use target Q network to compute Q_target(s', a') for each nonterminal next state.    
        # a' is chosen using the target pi network.
        next_Q_values = torch.zeros(self.P["batch_size"], device=self.device)
        nonterminal_next_actions = self.pi_target(nonterminal_next_states)
        if self.P["td3"]:
            # For TD3 we add clipped noise to a' to reduce overfitting.
            noise = (torch.randn_like(nonterminal_next_actions) * self.P["td3_noise_std"]
                    ).clamp(-self.P["td3_noise_clip"], self.P["td3_noise_clip"])
            nonterminal_next_actions = (nonterminal_next_actions + noise).clamp(-1, 1)
        next_Q_values[nonterminal_mask] = self.Q_target(self.sa_concat(nonterminal_next_states, nonterminal_next_actions.detach())).squeeze()
        if self.P["td3"]: 
            # For TD3 we use both target Q networks and take the minimum value.
            next_Q2_values = torch.zeros(self.P["batch_size"], device=self.device)
            next_Q2_values[nonterminal_mask] = self.Q2_target(self.sa_concat(nonterminal_next_states, nonterminal_next_actions.detach())).squeeze()
            next_Q_values = torch.min(next_Q_values, next_Q2_values)        
        # Compute target = reward + discounted Q_target(s', a').
        Q_targets = rewards + (self.P["gamma"] * next_Q_values)#.detach()
        # Update value in the direction of TD error. 
        self.optimiser_Q.zero_grad()
        Q_values = self.Q(self.sa_concat(states, actions)).squeeze()
        value_loss = F.smooth_l1_loss(Q_values, Q_targets)
        value_loss.backward(retain_graph=True) 
        for param in self.Q.parameters():
            param.grad.data.clamp_(-1, 1) # Implement gradient clipping.
        self.optimiser_Q.step()
        if self.P["td3"]: 
            # For TD3, do the same for the second Q network.
            self.optimiser_Q2.zero_grad()
            Q2_values = self.Q2(self.sa_concat(states, actions)).squeeze()
            value2_loss = F.smooth_l1_loss(Q2_values, Q_targets)
            value2_loss.backward() 
            for param in self.Q2.parameters():
                param.grad.data.clamp_(-1, 1) # Implement gradient clipping.
            self.optimiser_Q2.step()
        policy_loss = np.nan
        if (not self.P["td3"]) or (self.total_t % self.P["td3_policy_freq"] == 0): 
            # For TD3, only update policy and targets every N timesteps.
            # Update policy in the direction of increasing value according to self.Q (the policy gradient).
            self.optimiser_pi.zero_grad()
            policy_loss = -self.Q(self.sa_concat(states, self.pi(states))).mean()
            policy_loss.backward()
            self.optimiser_pi.step()
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

    def sa_concat(_, states, actions):
        """Concatenate states and actions into a single input vector for Q networks."""
        return torch.cat([states, actions], 1).float()

    def per_timestep(self, state, action, reward, next_state):
        """Operations to perform on each timestep during training."""
        self.memory.add(state, torch.FloatTensor([action], device=self.device), torch.FloatTensor([reward], device=self.device), next_state)
        losses = self.update_on_batch()
        if losses: self.ep_losses.append(losses)
        self.total_t += 1

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        if self.ep_losses: mean_policy_loss, mean_value_loss = np.nanmean(self.ep_losses, axis=0)
        else: mean_policy_loss, mean_value_loss = 0., 0.
        del self.ep_losses[:]; #self.noise.reset(); self.total_t = 0
        return {"logs":{"sigma": self.noise.sigma, "policy_loss": mean_policy_loss, "value_loss": mean_value_loss}}