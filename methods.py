import argparse
import gym
import numpy as np
from itertools import count
import random
import math
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import namedtuple

wandb.init(project="RL")

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.affine1 = nn.Linear(4, 16)
        #self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(16, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        #x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return action_scores

class ReinforcementAgent(object):
    """
    This class does all the reinforcement learning
    """
    def __init__(self,
                 env,
                 policy,
                 optimizer,
                 args,
                 device,
                 name="reinforcement agent"):
        self.name = name
        self._env = env
        self._policy = policy
        self._optimizer = optimizer
        self._args = args
        self._device = device
        self._eps = np.finfo(np.float32).eps.item()

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self._policy(state)
        m = Categorical(probs)
        action = m.sample()
        self._policy.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def finish_episode(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self._policy.rewards[::-1]:
            R = r + self._args.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self._eps)
        for log_prob, R in zip(self._policy.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self._optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self._optimizer.step()
        del self._policy.rewards[:]
        del self._policy.saved_log_probs[:]

    def reinforce(self):
        running_reward = 10
        for i_episode in count(1):
            state, ep_reward = self._env.reset(), 0
            for t in range(1, 10000):  # Don't infinite loop while learning
                action = self.select_action(state)
                state, reward, done, _ = self._env.step(action)
                if self._args.render:
                    self._env.render()
                self._policy.rewards.append(reward)
                ep_reward += reward
                if done:
                    break

            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            self.finish_episode()
            if i_episode % self._args.log_interval == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i_episode, ep_reward, running_reward))
            if running_reward > self._env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                      "the last episode runs to {} time steps!".format(running_reward, t))
                break


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent(object):
    """
    This class does all the reinforcement learning with DQN algo
    """
    def __init__(self,
                 env,
                 policy,
                 target,
                 optimizer,
                 args,
                 device,
                 name="reinforcement agent"):
        self.name = name
        self._env = env
        self._policy = policy
        self._target = target
        self._optimizer = optimizer
        self._args = args
        self._device = device
        self._steps_done = 0
        self._memory = ReplayMemory(10000)
        wandb.watch(self._policy)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self._args.eps_start + (self._args.eps_start - self._args.eps_end) * \
                        math.exp(-1. * self._steps_done / self._args.eps_decay)
        self._steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self._policy(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self._env.action_space.n)]], device=self._device, dtype=torch.long)

    def optimize_model(self):
        if len(self._memory) < self._args.batch_size:
            return
        transitions = self._memory.sample(self._args.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self._device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self._policy(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self._args.batch_size, device=self._device)
        next_state_values[non_final_mask] = self._target(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self._args.gamma) + reward_batch
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        wandb.log({"Loss": loss})
        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        for param in self._policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self._optimizer.step()

    def reinforce(self):
        num_episodes = 5000
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            state = self._env.reset()
            state = torch.from_numpy(state).float().unsqueeze(0)

            for t in count():
                # Select and perform an action
                action = self.select_action(state)
                next_state, reward, done, _ = self._env.step(action.item())
                reward = torch.tensor([reward], device=self._device)
                next_state = torch.from_numpy(next_state).float().unsqueeze(0)

                # Store the transition in memory
                self._memory.push(state, action, next_state, reward)
                wandb.log({"memory_length": len(self._memory)})
                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                self.optimize_model()
                if done:
                    wandb.log({"Episode Length": t + 1})
                    #plot_durations()
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self._args.target_update == 0:
                self._target.load_state_dict(self._policy.state_dict())