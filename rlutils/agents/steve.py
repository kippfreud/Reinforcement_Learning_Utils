from .ddpg import DdpgAgent # STEVE inherits from DDPG.
from ._default_hyperparameters import default_hyperparameters
from ..common.networks import SequentialNetwork
from ..common.utils import col_concat

import numpy as np
import torch
import torch.nn.functional as F


class SteveAgent(DdpgAgent):
    def __init__(self, env, hyperparameters):
        """
        Stochastic ensemble value expansion (STEVE). From:
            "Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion"
        NOTE: Currently requires reward function to be provided rather than learned.
        NOTE: The model is also set up to predict state *derivatives*, unlike in the original paper. 
        """
        assert "reward" in hyperparameters, f"{type(self).__name__} requires a reward function."
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
            self.nonfixed_dim = self.P["nonfixed_dim"] if "nonfixed_dim" in self.P else self.env.observation_space.shape[0] 
            self.fixed_pad = torch.nn.ZeroPad2d((0, self.env.observation_space.shape[0] - self.nonfixed_dim, 0, 0))
            action_dim = self.env.action_space.shape[0]        
        self.models = [SequentialNetwork(code=self.P["net_model"], input_shape=self.nonfixed_dim+action_dim, output_size=self.nonfixed_dim, lr=self.P["lr_model"]).to(self.device) for _ in range(self.P["num_models"])]
        # Parameters for action scaling.
        self.act_k = torch.tensor((self.env.action_space.high - self.env.action_space.low) / 2.)
        self.act_b = torch.tensor((self.env.action_space.high + self.env.action_space.low) / 2.)
        # Small float used to prevent div/0 errors.
        self.eps = np.finfo(np.float32).eps.item() 
        # Tracking variables.
        self.random_mode = True
        self.ep_losses_model = []
        self.ep_model_usage = []

    def act(self, state, explore=True, do_extra=False):
        """Either random or DDPG action selection."""
        with torch.no_grad():
            extra = {}
            if self.random_mode: action = torch.tensor([self.env.action_space.sample()])
            else: action, extra = DdpgAgent.act(self, state, explore, do_extra); action = torch.tensor([action])
            if do_extra: 
                sim_next_state = self.predict(state, action)
                extra["sim_next_state"] = sim_next_state[0].numpy()
                # NOTE: misleading as uses simulated next state rather than true one.
                # extra["reward_components"] = self.P["reward"](state, self._action_scale(action), sim_next_state)[0] 
            return action[0].numpy(), extra

    def predict(self, states, actions, mode="mean"):
        """Use all models to predict the next state for an array of state-action pairs. Return either mean or all."""
        sa = col_concat(states[:,:self.nonfixed_dim], actions)
        ds = torch.cat([self.fixed_pad(model(sa)).unsqueeze(2) for model in self.models], dim=2)
        if mode == "mean":  return states + ds.mean(axis=2)
        elif mode == "all": return states + ds

    def update_on_batch(self, model_only=False):
        """Use a random batch from the replay memory to update the model, pi and Q network parameters."""
        # TODO: Batch normalisation of state dimensions.
        if self.total_t % self.P["model_freq"] == 0:
            # Optimise each model on a different batch.
            for model in self.models:
                states, actions, _, _, next_states = self.memory.sample(self.P["batch_size"], keep_terminal_next=True)
                if states is None: return 
                # Update model in the direction of the true state derivatives using MSE loss.
                states_and_actions = col_concat(states[:,:self.nonfixed_dim], actions)
                loss = F.mse_loss(model(states_and_actions), next_states[:,:self.nonfixed_dim] - states[:,:self.nonfixed_dim])
                model.optimise(loss)
                self.ep_losses_model.append(loss.item()) # Keeping separate prevents confusing the DDPG methods.
        if model_only: return self.ep_losses_model[-1]
        # TODO: Handle termination via nonterminal_mask.
        # Sample another batch, this time for training pi and Q.
        states, actions, rewards, _, next_states = self.memory.sample(self.P["batch_size"], keep_terminal_next=True)
        raise NotImplementedError("Use intrinsic reward function instead of stored rewards in buffer")
        rewards = rewards.reshape(-1,1)         
        # Use models to build (hopefully) better Q_targets by simulating forward dynamics.
        Q_targets = torch.zeros((self.P["batch_size"], self.P["horizon"]+1, self.P["num_models"], len(self.Q_target)))
        with torch.no_grad(): 
            # Compute model-free targets.
            for j, target_net in enumerate(self.Q_target):
                next_actions = self.pi_target(next_states) # Select a' using the target pi network.
                # Same target for all models at this point.
                Q_targets[:,0,:,j] = (rewards + self.P["gamma"] * target_net(col_concat(next_states, next_actions))).expand(self.P["batch_size"], self.P["num_models"])
            for i, model in enumerate(self.models):  
                # Run a forward simulation for each model. 
                sim_states, sim_actions, sim_returns = next_states, next_actions, 0      
                for h in range(1, self.P["horizon"]+1): 
                    # Use model and target pi network to get next states and actions.
                    sim_next_states = sim_states + self.fixed_pad(model(col_concat(sim_states[:,:self.nonfixed_dim], sim_actions))) # Model predicts derivatives.
                    sim_next_actions = self.pi_target(sim_next_states)
                    # Use reward function to get reward for simulated state-action-next-state tuple and add to cumulative return.
                    sim_rewards = self.P["reward"](sim_states, self._action_scale(sim_actions), sim_next_states)
                    assert sim_rewards.shape == (self.P["batch_size"], 1)
                    sim_returns += (self.P["gamma"] ** h) * sim_rewards
                    # Store Q_targets for this horizon.
                    for j, target_net in enumerate(self.Q_target): 
                        Q_targets[:,h,i,j] = (sim_returns + ((self.P["gamma"] ** (h+1)) * target_net(col_concat(sim_next_states, sim_next_actions)))).squeeze()
                    sim_states, sim_actions = sim_next_states, sim_next_actions
        # Inverse variance weighting of horizons. 
        var = Q_targets.var(dim=(2, 3)) + self.eps # Prevent div/0 error.
        inverse_var = 1 / var
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
        DdpgAgent.per_timestep(self, state, action, reward, next_state, done, suppress_update=self.random_mode)

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        logs = DdpgAgent.per_episode(self)
        logs["model_loss"] = np.mean(self.ep_losses_model) if self.ep_losses_model else 0.
        logs["model_usage"] = np.mean(self.ep_model_usage) if self.ep_model_usage else 0.
        logs["random_mode"] = int(self.random_mode)
        del self.ep_losses_model[:]; del self.ep_model_usage[:]
        return logs

    def _action_scale(self, actions):
        """Rescale actions from [-1,1] to action space extents."""
        return (self.act_k * actions) + self.act_b