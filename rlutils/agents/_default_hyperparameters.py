"""
NOTE: Agent names must be lowercase.
"""

default_hyperparameters = {
  
  "actor_critic": {
    "net_pi": [(None, 64), "R", (64, 128), "R", (128, None), "S"], # Softmax policy.
    "net_V": [(None, 64), "R", (64, 128), "R", (128, None)],
    "lr_pi": 1e-4, # Learning rate for policy.
    "lr_V": 1e-3, # Learning rate for state value function.
    "gamma": 0.99 # Discount factor.
  },   

  "ddpg": {
    "net_pi": [(None, 256), "R", (256, 256), "R", (256, None), "T"],
    "net_Q": [(None, 256), "R", (256, 256), "R", (256, 1)],
    "replay_capacity": 50000, # Size of replay buffer (starts overwriting when full).
    "batch_size": 128, # Size of batches to sample from replay buffer during learning.
    "lr_pi": 1e-4, # Learning rate for policy.
    "lr_Q": 1e-3, # Learning rate for state-action value function.
    "gamma": 0.99, # Discount factor.
    "tau": 0.005, # Parameter for Polyak averaging of target network parameters.
    "noise_params": ("ou", 0., 0.15, 0.3, 0.3, 1000),
    "td3": False, # Whether or not to enable the TD3 enhancements. 
    # --- If TD3 enabled ---
    "td3_noise_std": 0.2,
    "td3_noise_clip": 0.5,
    "td3_policy_freq": 2
  },

  "dqn": {
    "net_Q": [(None, 256), "R", (256, 128), "R", (128, 64), "R", (64, None)], # From https://github.com/transedward/pytorch-dqn/blob/master/dqn_model.py.
    "replay_capacity": 10000, # Size of replay buffer (starts overwriting when full).
    "batch_size": 128, # Size of batches to sample from replay buffer during learning.
    "lr_Q": 1e-3, # Learning rate for state-action value function.
    "gamma": 0.99, # Discount factor.
    "epsilon_start": 0.9,
    "epsilon_end": 0.05,
    "epsilon_decay": 500000,
    "target_update": ("soft", 0.0005), # Either ("hard", freq) or ("soft", tau).
    "double": True, # Whether to enable double DQN variant to reduce overestimation bias.
    "reward_components": 1 # For reward decomposition.
  },

  "off_policy_mc": {
    "gamma": 0.99, # Discount factor.
    "epsilon": 0.5
  },

  "random": {
    "method": "uniform",
    "inertia": 0,
    "gamma": 0.99 # Discount factor.
  },

  "reinforce": {
    "net_pi": [(None, 64), "R", (64, 128), "R", (128, None), "S"], # Softmax policy.
    "lr_pi": 1e-4, # Learning rate for policy.
    "gamma": 0.99, # Discount factor.
    "baseline": "adv", # Baselining method: either "off", "Z" or "adv".
    # --- If baseline == "adv" ---
    "net_V": [(None, 64), "R", (64, 128), "R", (128, None)],
    "lr_V": 1e-3, # Learning rate for state value function.
  },

  "sac": {
    "net_pi": [(None, 256), "R", (256, 256), "R", (256, None)],
    "net_Q": [(None, 256), "R", (256, 256), "R", (256, None)],
    "replay_capacity": 10000, # Size of replay buffer (starts overwriting when full).
    "batch_size": 256, # Size of batches to sample from replay buffer during learning.
    "lr_pi": 1e-4, # Learning rate for policy.
    "lr_Q": 1e-3, # Learning rate for state-action value function.
    "gamma": 0.99, # Discount factor.
    "alpha": 0.2, # Weighting for entropy regularisation term.
    "tau": 0.005 # Parameter for Polyak averaging of target network parameters.
  },

  "simple_model_based": {  
    "net_model": [(None, 32), "R", (32, 64), "R", (64, None)],
    "num_random_steps": 2000, # Size of random replay buffer (disables random mode when full).
    "batch_size": 256,
    "model_freq": 10, # Number of steps between model updates.
    "lr_model": 1e-3, # Learning rate for dynamics model.
    "random_mode_only": False,
    # --- If not random_mode_only ---
    "replay_capacity": 2000, # Size of replay buffer (starts overwriting when full).
    "batch_ratio": 0.9, # Proportion of on-policy transitions.
    "num_rollouts": 50,
    "rollout_horizon": 20,
    "gamma": 0.99 # Discount factor.
  },
  
  "stable_baselines": { # NOTE: Other defaults specified in StableBaselines library.
    "model_class": "DQN",
    "verbose": True
  },

  "steve": {
    "net_model": [(None, 32), "R", (32, 64), "R", (64, None)],
    "num_random_steps": 1000, # Number of steps before disabling random mode and starting learning.
    "num_models": 2, # Number of parallel dynamics models to train.
    "model_freq": 1, # Number of steps between model updates.
    "lr_model": 1e-3, # Learning rate for dynamics model.
    "horizon": 5, # Maximum number of model steps to run to produce Q values.
    "ddpg_parameters": {"td3": True} # STEVE is built around DDPG, and needs multiple Q_target networks.
  }

}