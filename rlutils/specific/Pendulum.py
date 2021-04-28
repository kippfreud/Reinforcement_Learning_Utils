import numpy as np

def reward_function(state, action):
    """Reward function for Pendulum-v0."""
    MAX_TORQUE = 2.
    def angle_normalize(x): return ((x + np.pi) % (2 * np.pi)) - np.pi
    th = np.arccos(np.clip(state[0],-1,1)) # State[0] is cos(theta)
    thdot = state[2]
    u = np.clip(action, -MAX_TORQUE, MAX_TORQUE)
    costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
    return -costs