"""
NOTE: Must be lowercase.
"""

default_hyperparameters = {
  
  "actor_critic": {
    "lr_pi": 1e-4, # Learning rate for policy.
    "lr_V": 1e-3, # Learning rate for state value function.
    "gamma": 0.99 # Discount factor.
  },   

  "ddpg": {
    "replay_capacity": 50000, # Size of replay buffer (starts overwriting when full).
    "batch_size": 128,
    "lr_pi": 1e-4, # Learning rate for policy.
    "lr_Q": 1e-3, # Learning rate for state-action value function.
    "gamma": 0.99, # Discount factor.
    "tau": 0.005,
    "noise_params": (0., 0.15, 0.3, 0.3, 1000),
    "td3": True, # Whether or not to enable the TD3 enhancements. 
    "td3_noise_std": 0.2,
    "td3_noise_clip": 0.5,
    "td3_policy_freq": 2
  },

  "dqn": {
    "replay_capacity": 10000, # Size of replay buffer (starts overwriting when full).
    "batch_size": 128,
    "lr_Q": 1e-3, # Learning rate for state-action value function.
    "gamma": 0.99, # Discount factor.
    "epsilon_start": 0.9,
    "epsilon_end": 0.05,
    "epsilon_decay": 500000,
    "updates_between_target_clone": 2000,
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
    "lr_pi": 1e-4, # Learning rate for policy.
    "lr_V": 1e-3, # Learning rate for state value function.
    "gamma": 0.99, # Discount factor.
    "baseline": "adv"
  },

  "sac": {
    "replay_capacity": 10000, # Size of replay buffer (starts overwriting when full).
    "batch_size": 256,
    "lr_pi": 1e-4, # Learning rate for policy.
    "lr_Q": 1e-3, # Learning rate for state-action value function.
    "gamma": 0.99, # Discount factor.
    "alpha": 0.2,
    "tau": 0.01
  },

  "simple_model_based": {  
    "replay_capacity": 2000, # Size of replay buffer (starts overwriting when full).
    "random_replay_capacity": 2000, 
    "batch_size": 256,
    "batch_ratio": 0.9, # Proportion of on-policy transitions.
    "steps_between_update": 10,
    "lr_model": 1e-3, # Learning rate for dynamics model.
    "num_rollouts": 50,
    "rollout_horizon": 20,
    "gamma": 0.99 # Discount factor.
  },
  
  "stable_baselines": { # NOTE: Other defaults specified in StableBaselines library.
    "model_class": "DQN",
    "verbose": True
  }
}