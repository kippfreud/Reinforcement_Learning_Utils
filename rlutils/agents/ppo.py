from ._generic import Agent
from ..common.networks import SequentialNetwork
from ..common.memory import PpoMemory

import numpy as np
import torch
from torch.distributions import MultivariateNormal, Categorical
import torch.nn.functional as F


class PpoAgent(Agent):
    def __init__(self, env, hyperparameters):
        """
        DESCRIPTION
        """
        Agent.__init__(self, env, hyperparameters)
        # Establish whether action space is continuous or discrete.
        self.continuous_actions = len(self.env.action_space.shape) > 0
        # Create pi and V networks.
        if len(self.env.observation_space.shape) > 1: raise NotImplementedError() 
        self.action_dim = self.env.action_space.shape[0] if self.continuous_actions else self.env.action_space.n
        self.pi, self.pi_new = [SequentialNetwork(code=(self.P["net_pi_cont"] if self.continuous_actions else self.P["net_pi_disc"]), 
            input_shape=self.env.observation_space.shape[0], output_size=self.action_dim, 
            lr=self.P["lr_pi"], eval_only=eval_only).to(self.device) for eval_only in (True, False)]
        self.pi_new.load_state_dict(self.pi.state_dict()) # Require a copy of pi to perform clipping.
        self.V = SequentialNetwork(code=self.P["net_V"], input_shape=self.env.observation_space.shape[0], output_size=1, lr=self.P["lr_V"]).to(self.device)
        # Create replay memory. Note that PPO is on-policy so this is cleared after each update.
        self.memory = PpoMemory()
        if self.continuous_actions:
            # TODO: Move to exploration.py.
            if self.P["noise_params"][0] == "norm": 
                # Create linearly decaying multivariate normal action noise for exploration.
                self.action_std = self.P["noise_params"][1]
                self.noise = torch.full((self.action_dim,), self.action_std**2).to(self.device)
            else: raise Exception()
            self.total_t = 0 # Used for noise decay.
        # Small float used to prevent div/0 errors.
        self.eps = np.finfo(np.float32).eps.item() 
        # Tracking variables.
        self.last_action, self.last_log_prob = None, None

    def act(self, state, explore=True, do_extra=False):
        """Deterministic action selection plus additive noise for continuous, probabilistic action selection for discrete."""
        if self.continuous_actions:
            action_greedy = self.pi(state)
            dist = MultivariateNormal(action_greedy, torch.diag(self.noise).unsqueeze(dim=0))
            extra = {"action_greedy": action_greedy}
        else:
            action_probs = self.pi(state)
            dist = Categorical(action_probs) 
            extra = {"pi": action_probs.cpu().detach().numpy()} if do_extra else {}
        action = dist.sample()
        if self.continuous_actions: action = torch.clamp(action, -1, 1)
        self.last_action = action
        self.last_log_prob = dist.log_prob(action)
        return action.cpu().detach().numpy()[0] if self.continuous_actions else action.item(), extra

    def update_on_experience(self):
        """Use the latest batch of experience to update the policy and value network parameters."""
        # Loop backwards through episodes to compute Monte Carlo returns.      
        g, returns = 0, []
        for reward, done in zip(reversed(self.memory.reward), reversed(self.memory.done)):
            if done: g = 0
            g = reward + (self.P["gamma"] * g)
            returns.insert(0, g)
        # Apply baselining.
        returns = self.baseline(torch.tensor(returns, dtype=torch.float32).to(self.device), None)
        # Get states, actions and log probs in Torch tensors.
        states = torch.squeeze(torch.stack(self.memory.state, dim=0)).detach().to(self.device)
        actions = torch.squeeze(torch.stack(self.memory.action, dim=0)).detach().to(self.device)
        log_probs = torch.squeeze(torch.stack(self.memory.log_prob, dim=0)).detach().to(self.device)
        # Update for multiple gradient steps.
        for _ in range(self.P["num_steps_per_update"]):
            log_probs_new, values, dist_entropy = self._evaluate(states, actions)
            # Compute the ratio pi / pi_old for constraining the update.
            ratios = torch.exp(log_probs_new - log_probs.detach())

            # Compute surrogate loss.
            advantages = returns - values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.P["epsilon"], 1+self.P["epsilon"]) * advantages
            
            # Update value using MSE loss.
            value_loss = F.mse_loss(values, returns)
            self.V.optimise(value_loss) 

            # Update 
            raise NotImplementedError("Is this implemented?")
            policy_loss = (-torch.min(surr1, surr2) - 0.01*dist_entropy).mean()
            self.pi_new.optimise(policy_loss) 

        # Copy new weights into policy and clear buffer.
        self.pi.load_state_dict(self.pi_new.state_dict())
        self.memory.clear()

        return policy_loss.item(), value_loss.item()

    def baseline(self, returns, values):
        """Apply baselining to returns to improve update stability."""
        if self.P["baseline"] == "off":  return returns # No baselining.
        elif self.P["baseline"] == "Z":   return (returns - returns.mean()) / (returns.std() + self.eps) # Z-normalisation.
        elif self.P["baseline"] == "adv": return (returns - values).detach() # Advantage (subtract value prediction).
        else: raise NotImplementedError("Baseline method not recognised.")

    def per_timestep(self, state, action, reward, next_state, done):
        """Operations to perform on each timestep during training."""
        assert self.last_action is not None
        self.memory.state.append(state)
        self.memory.action.append(self.last_action)
        self.memory.log_prob.append(self.last_log_prob)
        self.memory.reward.append(reward)
        self.memory.done.append(done)
        if self.continuous_actions and self.total_t % self.P["noise_params"][4] == 0: 
            # Decay action standard deviation linearly. 
            self.action_std = max(self.action_std - self.P["noise_params"][3], self.P["noise_params"][2])
            self.noise = torch.full((self.action_dim,), self.action_std**2).to(self.device)
        self.total_t += 1

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        policy_loss, value_loss = self.update_on_experience() 
        return {"policy_loss": policy_loss, "value_loss": value_loss}

    def _evaluate(self, states, actions):
        if self.continuous_actions:
            actions_greedy = self.pi_new(states)
            cov_mat = torch.diag_embed(self.noise.expand_as(actions_greedy)).to(self.device)
            dist = MultivariateNormal(actions_greedy, cov_mat)
            if self.action_dim == 1: # Correction for environments with one action dim.
                actions = actions.reshape(-1, self.action_dim)
        else: dist = Categorical(self.pi_new(states))    
        log_probs = dist.log_prob(actions)
        values = torch.squeeze(self.V(states))
        dist_entropy = dist.entropy()
        return log_probs, values, dist_entropy