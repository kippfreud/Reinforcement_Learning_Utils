# from ..common.networks import SequentialNetwork
# from ..common.memory import ReplayMemory

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F 


DEFAULT_HYPERPARAMETERS = {
    "replay_capacity": 50000,
    "batch_size": 128,
    "lr_pi": 1e-4,
    "lr_Q": 1e-3,
    "gamma": 0.99,
    "tau": 1e-2,
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
        num_actions = action_space.shape[0]
        self.pi = Actor(state_shape[0], 256, num_actions)
        self.optimiser_pi = optim.Adam(self.pi.parameters(), lr=self.P["lr_pi"])
        self.pi_target = Actor(state_shape[0], 256, num_actions)
        self.pi_target.load_state_dict(self.pi.state_dict()) # Clone.
        self.pi_target.eval() # Turn off training mode for target net.
        self.Q = Critic(state_shape[0] + num_actions, 256) # Action is an *input* to the Q network here.
        self.optimiser_Q = optim.Adam(self.Q.parameters(), lr=self.P["lr_Q"])
        self.Q_target = Critic(state_shape[0] + num_actions, 256)
        self.Q_target.load_state_dict(self.Q.state_dict()) 
        self.Q_target.eval()
        # Create replay memory.
        self.memory = ReplayMemory(self.P["replay_capacity"]) 
        # Create noise process for exploration.
        self.noise = OUNoise(action_space); self.noise.reset()
        # Tracking variables.   
        self.ep_t = 0  
        self.ep_losses = []  
    
    def act(self, state):
        """Deterministic action selection plus additive noise."""
        action = self.pi.forward(state)
        action = action.detach().numpy()[0,0]
        return self.noise.get_action(action, self.ep_t), {}
    
    def update_on_batch(self):
        """Use a random batch from the replay memory to update the pi and Q network parameters."""
        if len(self.memory) < self.P["batch_size"]: return
        # Sample a batch and transpose it (see https://stackoverflow.com/a/19343/3343043).
        batch = self.memory.element(*zip(*self.memory.sample(self.P["batch_size"])))
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        # Compute Q(s, a) by running each s, a through self.Q.
        Q_values = self.Q(states, actions).squeeze()
        # Identify nonterminal states (note that replay memory elements are initialised to None).
        nonterminal_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        nonterminal_next_states = torch.cat([s for s in batch.next_state if s is not None])
        # Use target Q network to compute Q_target(s', a') for each nonterminal next state.    
        # a' is chosen using the target pi network.
        nonterminal_next_actions = self.pi_target(nonterminal_next_states)
        next_Q_values = torch.zeros(self.P["batch_size"], device=self.device)
        next_Q_values[nonterminal_mask] = self.Q_target(nonterminal_next_states, nonterminal_next_actions.detach()).squeeze()
        # Compute target = reward + discounted Q_target(s', a').
        Q_targets = rewards + (self.P["gamma"] * next_Q_values)
        # Zero gradient buffers of all parameters.
        self.optimiser_pi.zero_grad(); self.optimiser_Q.zero_grad()
        # Update value in the direction of TD error using MSE loss. 
        value_loss = F.mse_loss(Q_values, Q_targets)
        value_loss.backward() 
        self.optimiser_Q.step()
        # Update policy in the direction of increasing value according to the Q network.
        policy_loss = -self.Q(states, self.pi(states)).mean()
        policy_loss.backward()
        self.optimiser_pi.step()
        # Perform soft updates on targets.
        for target_param, param in zip(self.pi_target.parameters(), self.pi.parameters()):
            target_param.data.copy_(param.data * self.P["tau"] + target_param.data * (1.0 - self.P["tau"]))
        for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
            target_param.data.copy_(param.data * self.P["tau"] + target_param.data * (1.0 - self.P["tau"]))
        return policy_loss.item(), value_loss.item()

    def per_timestep(self, state, action, reward, next_state):
        """Operations to perform on each timestep during training."""
        self.memory.add(state, torch.tensor([action], device=agent.device), torch.tensor([reward], device=self.device), next_state)
        losses = self.update_on_batch()
        if losses: self.ep_losses.append(losses)
        self.ep_t += 1

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        if self.ep_losses: mean_policy_loss, mean_value_loss = np.mean(self.ep_losses, axis=0)
        else: mean_policy_loss, mean_value_loss = 0., 0.
        del self.ep_losses[:]; self.noise.reset(); self.ep_t = 0
        return {"logs":{"policy_loss": mean_policy_loss, "value_loss": mean_value_loss}}


import torch.nn as nn
class Critic(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1).float()
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 3e-4):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x


# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

import gym
# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)


from collections import namedtuple
import random


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        # Structure of a memory element.
        self.element = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
        self.memory = []
        self.position = 0

    def __len__(self): return len(self.memory) # Length is length of memory list.

    def add(self, *args):
        """Save a transition."""
        # Extend memory if capacity not yet reached.
        if len(self.memory) < self.capacity: self.memory.append(None) 
        # Overwrite current entry at this position.
        self.memory[self.position] = self.element(*args)
        # Increment position, cycling back to the beginning if needed.
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Retrieve a random sample of transitions."""
        return random.sample(self.memory, batch_size)



import matplotlib.pyplot as plt


env = NormalizedEnv(gym.make("Pendulum-v0"))

agent = DdpgAgent(env.observation_space.shape, env.action_space)
rewards = []
avg_rewards = []

for episode in range(50):
    state = env.reset()
    episode_reward = 0

    state = torch.from_numpy(state).float().unsqueeze(0)
    
    for step in range(500):
        action, _ = agent.act(state)

        next_state, reward, done, _ = env.step(action) 
        if done: next_state = None
        else: next_state = torch.from_numpy(next_state).float().unsqueeze(0)

        agent.memory.add(state, torch.tensor([action], device=agent.device), torch.tensor([reward], device=agent.device), next_state)
        
        agent.update_on_batch()        
        
        state = next_state
        episode_reward += reward

        if done:
            print("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
            break

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))

plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()