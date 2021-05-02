"""
Stochastic ensemble value expansion (STEVE). From:
    "Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion"
NOTE: Currently requires reward function to be provided rather than learned.
"""

from .ddpg import DdpgAgent, _sa_concat # STEVE inherits from DDPG.
from ._default_hyperparameters import default_hyperparameters
from ..common.networks import SequentialNetwork

import numpy as np
import torch
import torch.nn.functional as F


class SteveAgent(DdpgAgent):
    def __init__(self, env, hyperparameters):
        assert "reward_function" in hyperparameters, "Need to provide reward function."
        # Overwrite default hyperparameters for DDPG.
        P = default_hyperparameters["ddpg"]
        for k, v in default_hyperparameters["steve"]["ddpg_parameters"].items(): P[k] = v
        for k, v in hyperparameters["ddpg_parameters"].items(): P[k] = v
        DdpgAgent.__init__(self, env, P)
        assert len(self.Q_target) > 1, "Need multiple Q networks to do variance for horizon = 0."
        # Add parameters specific to STEVE.
        for k, v in hyperparameters.items(): 
            if k != "ddpg_parameters": self.P[k] = v
        # Create an ensemble of model networks.
        if len(self.env.observation_space.shape) > 1: raise NotImplementedError()
        else:
            self.state_dim = self.env.observation_space.shape[0]
            net_code = [(self.state_dim+self.env.action_space.shape[0], 32), "R", (32, 64), "R", (64, self.state_dim)]
        self.models = [SequentialNetwork(code=net_code, lr=self.P["lr_model"]).to(self.device) for _ in range(self.P["num_models"])]
        # Parameters for action scaling.
        self.act_k = (self.env.action_space.high - self.env.action_space.low) / 2.
        self.act_b = (self.env.action_space.high + self.env.action_space.low) / 2.
        # Tracking variables.
        self.random_mode = True
        self.ep_losses_model = []
        self.ep_model_usage = []

    def act(self, state, explore=True, do_extra=False):
        """Either random or DDPG action selection."""
        extra = {}
        if self.random_mode: action = 2*np.random.rand(*self.env.action_space.shape) - 1 # Normalised in [-1,1].
        else: action, extra = DdpgAgent.act(self, state, explore, do_extra)
        if do_extra: extra["next_state_pred"] = self.predict(state, action).numpy()
        return action, extra

    def predict(self, state, action):
        """Use model to predict the next state given a single state-action pair."""
        return state[0] + self.model(torch.cat((state[0], torch.Tensor(action if self.continuous_actions else [action]).to(self.device))).to(self.device)).detach()

    def update_on_batch(self):
        """Use a random batch from the replay memory to update the model, pi and Q network parameters."""

        # TODO: Batch normalisation of state dimensions.

        if len(self.memory) < self.P["batch_size"]: return
        if self.total_t % self.P["model_freq"] == 0:
            # Optimise each model on a different batch.
            for model in self.models:
                # Sample a batch and transpose it (see https://stackoverflow.com/a/19343/3343043).
                batch = self.memory.element(*zip(*self.memory.sample(self.P["batch_size"])))
                states = torch.cat(batch.state)
                actions = torch.cat(batch.action)
                next_states = torch.cat(batch.next_state)
                # Update model in the direction of the true change in state using MSE loss.
                states_and_actions = _sa_concat(states, actions)
                target = next_states - states_and_actions[:,:self.state_dim]
                prediction = model(states_and_actions)
                loss = F.mse_loss(prediction, target)
                model.optimise(loss)
                self.ep_losses_model.append(loss.item()) # Keeping separate prevents confusion of DDPG methods.
        # Sample another batch, this time for training pi and Q.
        batch = self.memory.element(*zip(*self.memory.sample(self.P["batch_size"])))
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        reward = torch.cat(batch.reward).reshape(-1,1)
        next_states = torch.cat(batch.next_state)
        # Use models to build (hopefully) better Q_targets by simulating forward dynamics.
        Q_targets = torch.zeros((self.P["batch_size"], self.P["horizon"]+1, self.P["num_models"], len(self.Q_target)))

        # TODO: Handle termination.

        with torch.no_grad(): 
            # Compute model-free targets.
            for j, target_net in enumerate(self.Q_target):
                next_actions = self.pi_target(next_states) # Select a' using the target pi network.
                # Same target for all models at this point.
                Q_targets[:,0,:,j] = (reward + self.P["gamma"] * target_net(_sa_concat(next_states, next_actions))).expand(self.P["batch_size"], self.P["num_models"])
            for i, model in enumerate(self.models):  
                # Run a forward simulation for each model. 
                sim_states, sim_actions, g = next_states, next_actions, 0      
                for h in range(1, self.P["horizon"]+1): 
                    # Use reward function to get reward for simulated state-action pair and add to cumulative return.
                    sim_rewards = [self.P["reward_function"](s, self._action_scale(a)) for s, a in zip(sim_states, sim_actions)]
                    g += (self.P["gamma"] ** h) * torch.Tensor(sim_rewards).reshape(-1, 1)
                    # Use model and target pi network to advance states and actions.
                    sim_states += model(_sa_concat(sim_states, sim_actions)) # Model predicts derivatives.
                    sim_actions = self.pi_target(sim_states)
                    # Store Q_targets for this horizon.
                    for j, target_net in enumerate(self.Q_target): 
                        Q_targets[:,h,i,j] = (g + ((self.P["gamma"] ** (h+1)) * target_net(_sa_concat(sim_states, sim_actions)))).squeeze()
        # Inverse variance weighting of horizons. 
        inverse_var = 1 / Q_targets.var(dim=(2, 3))
        normalised_weights = inverse_var / inverse_var.sum(dim=1, keepdims=True)
        self.ep_model_usage.append(1 - normalised_weights[:,0].mean().item())
        Q_targets = (Q_targets.mean(dim=(2, 3)) * normalised_weights).sum(dim=1)
        # Send Q_targets to DDPG update function and return losses.
        return DdpgAgent.update_on_batch(self, states, actions, Q_targets)

    def per_timestep(self, state, action, reward, next_state, done):
        """Operations to perform on each timestep during training."""
        if self.random_mode and self.total_t >= self.P["num_random_steps"]: 
            self.random_mode = False
            print("Random data collection complete.")
        DdpgAgent.per_timestep(self, state, action, reward, next_state, done, do_update=not(self.random_mode))

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        out = DdpgAgent.per_episode(self)
        if self.ep_losses_model: out["logs"]["model_loss"] = np.mean(self.ep_losses_model)
        else: out["logs"]["model_loss"] = 0.
        if self.ep_model_usage: out["logs"]["model_usage"] = np.mean(self.ep_model_usage)
        else: out["logs"]["model_usage"] = 0.
        del self.ep_losses_model[:]; del self.ep_model_usage[:]
        out["logs"]["random_mode"] = int(self.random_mode)
        return out

    def _action_scale(self, action):
        """Rescale action from [-1,1] to action space extents."""
        return self.act_k * action.numpy() + self.act_b