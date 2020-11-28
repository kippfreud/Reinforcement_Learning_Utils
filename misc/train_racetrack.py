"""
Exercise from p111-112 of Sutton & Barto.

Velocity in {0,1,2,3,4,5}^2. Cannot both be zero when t > 0.
Action (acceleration) in {-1,0,+1}^2. Size of action space = 9. 
With probability p_noise (default = 0.1), acceleration has no effect (velocity remains unchanged).
Random initialisation at one of the start states with velocity = [0,0].
Termination when projected location (pos + vel + acc) crosses the finish line.
Reward = -1 per timestep.
If projected to hit a track boundary, moved back to a random start state with velocity = [0,0].
"""

from agents.off_policy_MC import *

import numpy as np
import matplotlib.pyplot as plt


class RaceTrack:
    def __init__(self, n):
        grid = np.zeros((n,n))
        s0 = np.random.randint(n*0.5)     # Start line left.
        s1 = np.random.randint(s0+3, n-2) # Start line right.
        grid[0,s0:s1] = 1
        f0 = np.random.randint(n*0.25, n*0.75) # Finish line top.
        f1 = np.random.randint(f0+3, n+1)      # Finish line bottom.
        grid[f0:f1,-1] = 1
        outside = np.sort(np.random.choice(np.arange(s0,s1+1), f1-1)) # Random steps for outside curve.
        inside = np.sort(np.random.choice(np.arange(s1,n-1), f0-1))   # Random steps for inside curve.
        for r, o in enumerate(outside): 
            if r+1 >= f0: grid[r+1,o:] = 1
            else: grid[r+1,o:inside[r]] = 1
        self.n, self.grid, self.start, self.finish = n, grid, (s0,s1), (f0,f1)

    def reset(self):
        self.state = np.array([0, np.random.randint(self.start[0], self.start[1]), 0, 0])
        return self.state.copy()
    
    def step(self, action): 
        # Update velocity and position.
        self.state[2:] = np.clip(self.state[2:] + action, [0,0], [5,5])
        self.state[:2] = np.clip(self.state[:2] + self.state[2:], [0,0], [self.n-1,self.n-1])
        done = False
        # If at finish line, terminate episode.
        if self.state[1] == self.n-1 and self.state[0] >= self.finish[0] and self.state[0] < self.finish[1]:
            done = True
        # If outside of track, go back to a random starting position.
        if self.grid[tuple(self.state[:2])] == 0: 
            self.reset()
        return self.state.copy(), -1, done, None

    def render(self):
        render = self.grid.copy()
        render[tuple(self.state[:2])] = 0.5
        return render





n = 30

env = RaceTrack(n)
action_space = np.array([[[1,1],[0,1],[-1,1]],[[1,0],[0,0],[-1,0]],[[1,-1],[0,-1],[-1,-1]]])
agent = OffPolicyMCAgent(state_shape=(n,n,6,6), action_space=action_space)

# Plotting window.
fig, ax = plt.subplots(figsize=(10,10))
# ax.set_xticks(np.arange(0, n, 1))
# ax.set_yticks(np.arange(0, n, 1))
# ax.set_xticklabels(np.arange(0, n, 1))
# ax.set_yticklabels(np.arange(0, n, 1))
# ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
# ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
# ax.grid(which='minor', color='gray', linewidth=0.5)

# plt.ion(); plt.show()

for ep in range(10000):
    state, done, reward_sum = env.reset(), False, 0
    Q_last = agent.Q.copy()
    while not done:
        action = agent.act_behaviour(state)
        next_state, reward, done, _ = env.step(action)
        reward_sum += reward
        agent.ep_transitions.append((state, action, reward))
        state = next_state

    agent.update_on_episode()
    print(ep, np.abs(agent.Q - Q_last).max())
    Q_last = agent.Q.copy()

for ep in range(10000):
    state, done = env.reset(), False
    while not done:
        action = agent.act_target(state)
        print(action)
        next_state, reward, done, _ = env.step(action)
        ax.imshow(env.render(), cmap='gray')
        plt.pause(0.01)
        plt.cla()

plt.ioff(); plt.show()