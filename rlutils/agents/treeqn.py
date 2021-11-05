from ._generic import Agent
from .dqn import DqnAgent # TreeQN inherits from DQN.
from ..common.networks import TreeNetwork
from ..common.memory import ReplayMemory
from rlutils.common.exploration import EpsilonGreedy

import torch
import torch.nn.functional as F

# NOTE: Name taken :( https://arxiv.org/abs/1710.11417.
class TreeqnAgent(DqnAgent):
    def __init__(self, env, hyperparameters):
        """
        DESCRIPTION
        """
        assert "reward" in hyperparameters, f"{type(self).__name__} requires a reward function."
        Agent.__init__(self, env, hyperparameters) # NOTE: Use generic initialisation not DQN!
        # Create tree-structured decompositional Psi network.
        net_code, input_shape, num_actions = self.P["net_node"], self.env.observation_space.shape[0], self.env.action_space.n
        self.Psi = TreeNetwork(code=net_code, input_shape=input_shape, num_actions=num_actions, lr=self.P["lr_Q"]).to(self.device)
        self.Psi_target = TreeNetwork(code=net_code, input_shape=input_shape, num_actions=num_actions, eval_only=True).to(self.device)
        self.Psi_target.load_state_dict(self.Psi.state_dict()) # Clone.
        # Create replay memory.
        self.memory = ReplayMemory(self.P["replay_capacity"])
        # Initialise epsilon-greedy exploration.
        self.exploration = EpsilonGreedy(self.P["epsilon_start"], self.P["epsilon_end"], self.P["epsilon_decay"])
        # Tracking variables.
        if self.P["target_update"][0] == "hard": self.updates_since_target_clone = 0
        else: assert self.P["target_update"][0] == "soft"
        self.ep_losses = []

    def act(self, state, explore=True, do_extra=False):
        """Epsilon-greedy action selection."""
        assert self.Psi.m == self.P["reward"].m
        with torch.no_grad():
            Q = (self.Psi(state) * self.P["reward"].weights).sum(axis=2).squeeze()
            action, extra = self.exploration(Q, explore, do_extra)
            if do_extra: extra["reward_components"] = (self.P["reward"](state, action)).squeeze().cpu().numpy()
            return action, extra

    def update_on_batch(self):
        """Use a random batch from the replay memory to update the Q network parameters."""
        assert self.Psi.m == self.P["reward"].m
        states, actions, _, nonterminal_mask, nonterminal_next_states = self.memory.sample(self.P["batch_size"]) # Ignore extrinsic reward.
        if states is None: return 
        # Use target network to compute Psi_target(s', a') for each nonterminal next state.
        next_Psi_values = torch.zeros((self.P["batch_size"], self.Psi.m), device=self.device)
        Psi_t_n = self.Psi_target(nonterminal_next_states).detach()        
        # Compute Psi(s, a) by running each s through self.Psi, then selecting the corresponding column.
        Psi_values = self.Psi(states)[torch.arange(self.P["batch_size"]), actions, :]
        # In double DQN, a' is the Q-maximising action for self.Psi. This decorrelation reduces overestimation bias.
        # In regular DQN, a' is the Q-maximising action for self.Psi_target.
        Q_for_a_n = (self.Psi(nonterminal_next_states) if self.P["double"] else Psi_t_n) * self.P["reward"].weights                    
        nonterminal_next_actions = Q_for_a_n.sum(axis=2).argmax(1).detach()
        next_Psi_values[nonterminal_mask] = Psi_t_n[torch.arange(Psi_t_n.shape[0]), nonterminal_next_actions, :]  
        # Compute target = phi + discounted Psi_target(s', a').
        Psi_targets = self.P["reward"].phi(states, actions) + (self.P["gamma"] * next_Psi_values)
        # Update value in the direction of TD error using Huber loss. 
        if False: 
            # Regular Bellman update. 
            loss = F.smooth_l1_loss(Psi_values * self.P["reward"].weights, Psi_targets * self.P["reward"].weights) 
        else:   
            # Prioritise learning all successor features equally.
            # loss = F.smooth_l1_loss(Psi_values, Psi_targets)
            loss = F.mse_loss(Psi_values, Psi_targets)
        self.Psi.optimise(loss)
        # TODO: Pull this update into common.networks?
        if self.P["target_update"][0] == "hard":
            # Perform periodic hard update on target.
            self.updates_since_target_clone += 1
            if self.updates_since_target_clone >= self.P["target_update"][1]:
                self.Psi_target.load_state_dict(self.Psi.state_dict())
                self.updates_since_target_clone = 0
        elif self.P["target_update"][0] == "soft": self.Psi_target.polyak(self.Psi, tau=self.P["target_update"][1])
        else: raise NotImplementedError()
        return loss.item()