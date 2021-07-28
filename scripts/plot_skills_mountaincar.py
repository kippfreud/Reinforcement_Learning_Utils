import gym, rlutils
from rlutils.common.env_wrappers import NormaliseActionWrapper
import numpy as np
import matplotlib.pyplot as plt

env = NormaliseActionWrapper(gym.make("MountainCarContinuous-v0").unwrapped)

agent = rlutils.load(
    "agents/olive-dawn-9_ep1000.agent", 
    NormaliseActionWrapper(gym.make("MountainCarContinuous-v0").unwrapped)
)

obs = rlutils.Observer(state_dims=2, action_dims=1, do_extra=True)
rlutils.deploy(agent, {"num_episodes": 100, "episode_time_limit": 200, "render_freq": 0, "observe_freq": 1, "do_extra": True}, observer=obs)
df = obs.dataframe()
p, v, s = df[["s_0","s_1","skill"]].values.T

colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.scatter(p, v, c=[colours[i % len(colours)] for i in s.astype(int)], s=1)

plt.show()