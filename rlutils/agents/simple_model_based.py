"""
Simple model-based agent for discrete action spaces.
One component of the architecture from:
    "Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning"
"""

from ..common.networks import SequentialNetwork
from ..common.memory import ReplayMemory

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_HYPERPARAMETERS = {  
    "replay_capacity": 2000,
    "random_replay_capacity": 2000, 
    "batch_size": 256,
    "batch_ratio": 0.9, # Proportion of on-policy transitions.
    "steps_between_update": 10,
    "lr_model": 1e-3,
    "num_rollouts": 50,
    "rollout_horizon": 20,
    "gamma": 0.99,
}

class SimpleModelBasedAgent:
    def __init__(self, 
                 state_shape, 
                 action_space,
                 reward_function,
                 hyperparameters=DEFAULT_HYPERPARAMETERS
                 ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space = action_space
        self.reward_function = reward_function
        self.P = hyperparameters 
        # Create model network.
        if len(state_shape) > 1: raise NotImplementedError()
        else:
            net_code = [(state_shape[0]+1, 32), "R", (32, 64), "R", (64, state_shape[0])]
        self.model = SequentialNetwork(code=net_code, lr=self.P["lr_model"]).to(self.device)
        # Create replay memory in two components: one for on-policy transitions and one for random transitions.
        self.memory = ReplayMemory(self.P["replay_capacity"]) 
        self.random_memory = ReplayMemory(self.P["random_replay_capacity"]) 
        self.batch_split = (round(self.P["batch_size"] * self.P["batch_ratio"]), round(self.P["batch_size"] * (1-self.P["batch_ratio"])))
        # Tracking variables.
        self.random_mode = True
        self.total_t = 0 # Used for steps_between_update.
        self.ep_losses = []

    def act(self, state, explore=True):
        """Either random or model-based action selection."""
        if self.random_mode: action, extra = self.action_space.sample(), {}
        else: 
            returns, first_actions = self._model_rollout(state)
            best_rollout = np.argmax(returns)
            action, extra = first_actions[best_rollout], {"G": returns[best_rollout]}
        return action, extra

    def update_on_batch(self):
        """Use a random batch from the replay memory to update the model network parameters."""

        # TODO: NORMALISATION

        if self.random_mode: # During random mode, just sample from random memory.   
            if len(self.random_memory) < self.P["batch_size"]: return 
            batch = self.random_memory.sample(self.P["batch_size"])
        else: # After random mode, sample from both memories according to self.batch_split.
            if len(self.memory) < self.batch_split[0]: return 
            batch = list(self.memory.sample(self.batch_split[0])) + list(self.random_memory.sample(self.batch_split[1]))
        states_and_actions = torch.cat(tuple(torch.cat((x.state, torch.Tensor([[x.action]]).to(self.device)), dim=-1) for x in batch), dim=0)
        next_states = torch.cat(tuple(x.next_state for x in batch)).to(self.device)
        # Update model in the direction of the true change in state using MSE loss.
        target = next_states - states_and_actions[:,:-1]
        prediction = self.model(states_and_actions)
        loss = F.mse_loss(prediction, target)
        self.model.optimise(loss)
        return loss.item()

    def per_timestep(self, state, action, reward, next_state):
        """Operations to perform on each timestep during training."""
        state = state.to(self.device)
        #action = torch.tensor([action]).float().to(self.device)
        reward = torch.tensor([reward]).float().to(self.device)
        if self.random_mode and len(self.random_memory) >= self.P["random_replay_capacity"]: 
            self.random_mode = False
            print("Random data collection complete.")
        if next_state != None: 
            if self.random_mode: self.random_memory.add(state, action, reward, next_state)
            else: self.memory.add(state, action, reward, next_state)
        if self.total_t % self.P["steps_between_update"] == 0:
            loss = self.update_on_batch()
            if loss: self.ep_losses.append(loss)
        self.total_t += 1

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        if self.ep_losses: mean_loss = np.mean(self.ep_losses)
        else: mean_loss = 0.
        del self.ep_losses[:]
        return {"logs":{"model_loss": mean_loss, "random_mode": int(self.random_mode)}}

    def _model_rollout(self, state): 
        """Use model and reward function to generate and evaluate rollouts with random action selection.
        Then select the first action from the rollout with maximum return."""
        returns = []; first_actions = []
        for _ in range(self.P["num_rollouts"]):
            rollout_state, rollout_return = state[0].detach().clone().to(self.device), 0
            for t in range(self.P["rollout_horizon"]):
                rollout_action = self.action_space.sample() # Random action selection.
                if t == 0: first_actions.append(rollout_action)               
                rollout_state += self.model(torch.cat((rollout_state, torch.Tensor([rollout_action]).to(self.device))).to(self.device))
                rollout_return += (self.P["gamma"] ** t) * self.reward_function(rollout_state)                
            returns.append(rollout_return)
        return returns, first_actions


"""
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from RL_framework.common.networks import SequentialNetwork
from RL_framework.common.buffer import ReplayMemory, ProcessMinibatch
from RL_framework.common.utils import *
import wandb
import math
import numpy as np
import matplotlib.pyplot as plt


# Environment details
# ~~~~~~~~~~~~~~~~~~~
env = gym.make('CartPole-v0')
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

def reward_func(state):
    theta_threshold_radians = 12 * 2 * math.pi / 360
    x_threshold = 2.4
    x, _, theta, _ = state
    done = bool(x < -x_threshold
                or x > x_threshold
                or theta < -theta_threshold_radians
                or theta > theta_threshold_radians)
    return 1 if not done else 0

# General details
# ~~~~~~~~~~~~~~
wandb.init(project='framework_cartpole')
wandb.config.algorithm = 'MBMF'
num_episodes = 100

gamma = 0.99
params = {'sample_collection': 10,
          'buffer_size': 2000,
          'minibatch_size': 256,
          'random_buffer_size': 2000,
          'training_epoch': 50,
          'control_horizon': 20,
          'K_actions_sample': 50,
          'dataset_ratio': 0.9
}
wandb.config.gamma = gamma
wandb.config.update(params)


# Networks details
# ~~~~~~~~~~~~~~~~
network_layers = {'model_layers': [nn.Linear(obs_size + 1, 32),
                                   nn.ReLU(),
                                   nn.Linear(32, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, obs_size)]
                  }
learning_rates = dict(model_lr=1E-3)
loss_fnc = torch.nn.MSELoss()
wandb.config.update(network_layers)
wandb.config.update(learning_rates)

def K_rollouts(state, K, horizon):
    samples = np.random.randint(n_actions, size=(K, horizon))
    rewards = np.zeros(K)
    for i, sample in enumerate(samples):
        current_state = torch.Tensor(state)
        reward = 0
        for action in sample:
            state_action = torch.cat((current_state, torch.Tensor([action])))
            current_state += model(state_action)
            reward += reward_func(current_state)
        rewards[i] = reward
    best_K = np.argmax(rewards)
    return samples[best_K, 0]

# Initialisation
# ~~~~~~~~~~~~~~
model = SequentialNetwork(network_layers['model_layers'])
opt = optim.Adam(model.parameters(), lr=learning_rates['model_lr'])

dataset_random = ReplayMemory(params['random_buffer_size'])
dataset_rl = ReplayMemory(params['buffer_size'])

# Gather random data and train dynamics models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
while len(dataset_random) < params['random_buffer_size']:
    state = env.reset() + torch.normal(0, 0.001, size=(obs_size,)).numpy()
    terminal = False
    while terminal is False:
        action = np.random.randint(n_actions)
        next_state, reward, terminal, _ = env.step(action)
        dataset_random.add(state, action, reward, next_state, terminal, None, None)
        state = next_state

losses = []
for i in range(params['training_epoch']):
    minibatch = dataset_random.random_sample(params['minibatch_size'])
    t = ProcessMinibatch(minibatch)
    t.standardise()
    target = t.next_states - t.states + torch.normal(0, 0.001, size=t.states.shape)
    state_actions = torch.cat((t.states, t.actions), dim=1)
    current = model(state_actions + torch.normal(0, 0.001, size=state_actions.shape))
    loss = loss_fnc(target, current)
    losses.append(loss)
    opt.zero_grad()
    loss.backward()
    opt.step()
plt.plot(list(range(params['training_epoch'])), losses)
plt.show()

# Model based controller loop
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
episode_rewards = []
for episode in tqdm(range(num_episodes)):
    episode_reward = 0
    step = 0
    state = env.reset()
    terminal = False
    while terminal is False:
        action = K_rollouts(state, params['K_actions_sample'], params['control_horizon'])
        next_state, reward, terminal, _ = env.step(action)
        step += 1
        dataset_rl.add(state, action, reward, next_state, terminal, step, None)

        state = next_state
        episode_reward += reward

        if (step % params['sample_collection'] == 0 or terminal is True) and\
                len(dataset_rl) >= params['minibatch_size']:

            minibatch_random = dataset_random.random_sample(round(params['minibatch_size'] * (1-params['dataset_ratio'])))
            minibatch_rl = dataset_rl.random_sample(round(params['minibatch_size'] * params['dataset_ratio']))
            t_random = ProcessMinibatch(minibatch_random)
            t_rl = ProcessMinibatch(minibatch_rl)
            t_random.standardise()
            t_rl.standardise()
            target = torch.cat((t_random.next_states, t_rl.next_states)) - torch.cat((t_random.states, t_rl.states))
            state_actions = torch.cat((torch.cat((t_random.states, t_rl.states)),
                                       torch.cat((t_random.actions, t_rl.actions))), dim=1)
            current = model(state_actions)
            loss = loss_fnc(target, current)
            wandb.log({"model_loss": loss}, commit=False)
            opt.zero_grad()
            loss.backward()
            opt.step()

    wandb.log({"reward": episode_reward})
    episode_rewards.append(episode_reward)

"""