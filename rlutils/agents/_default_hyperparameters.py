"""
NOTE: Agent names must be lowercase.
"""

default_hyperparameters = {
  
  "actor_critic": {
    "lr_pi": 1e-4, # Learning rate for policy.
    "lr_V": 1e-3, # Learning rate for state value function.
    "gamma": 0.99 # Discount factor.
  },   

  "ddpg": {
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
    "replay_capacity": 10000, # Size of replay buffer (starts overwriting when full).
    "batch_size": 128, # Size of batches to sample from replay buffer during learning.
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
    "baseline": "adv" # Baselining method: either "off", "Z" or "adv".
  },

  "sac": {
    "replay_capacity": 10000, # Size of replay buffer (starts overwriting when full).
    "batch_size": 256, # Size of batches to sample from replay buffer during learning.
    "lr_pi": 1e-4, # Learning rate for policy.
    "lr_Q": 1e-3, # Learning rate for state-action value function.
    "gamma": 0.99, # Discount factor.
    "alpha": 0.2, # Weighting for entropy regularisation term.
    "tau": 0.005 # Parameter for Polyak averaging of target network parameters.
  },

  "simple_model_based": {  
    "random_replay_capacity": 2000, # Size of random replay buffer (disables random mode when full).
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
    "num_random_steps": 1000, # Number of steps before disabling random mode and starting learning.
    "num_models": 2, # Number of parallel dynamics models to train.
    "model_freq": 1, # Number of steps between model updates.
    "lr_model": 1e-3, # Learning rate for dynamics model.
    "horizon": 5, # Maximum number of model steps to run to produce Q values.
    "ddpg_parameters": {"td3": True} # STEVE is built around DDPG, and needs multiple Q_target networks.
  }

}