from rlutils.common.deployment import deploy
from rlutils.common.env_wrappers import NormaliseActionWrapper, CustomRewardWrapper
from rlutils.common.observer import Observer

import gym
from joblib import load

# RUN_NAME = "cosmic-yogurt-123"
RUN_NAME = "untitled_2020-12-10_15-16-29"

deploy_parameters = {
    "num_episodes": 1000,
    "max_timesteps_per_episode": 500,
    "wandb_monitor": False,
    "render_freq": 1,
    "save_video": False,
    "save_final_agent": False,
}

# Create observer class.
observer = Observer(
                    # state_dim_names=["pos", "vel", "ang", "vel_ang"],
                    state_dim_names=["pos_x","pos_y","vel_x","vel_y","ang","vel_ang","left_contact","right_contact"],
                    # action_dim_names=["action_idx"]
                    action_dim_names=["main_engine","lr_engine"]
                    )

# ===================================
# EXPERIMENT WITH CUSTOM REWARD FUNCTIONS
from custom_reward_experiment import R
# ===================================

env = CustomRewardWrapper(
      NormaliseActionWrapper(
      gym.make("LunarLanderContinuous-v2")), R=R)

# agent = load(f"saved_runs/{RUN_NAME}.joblib")
from rlutils.agents.stable_baselines import *
agent = StableBaselinesAgent(env, {"model_class": "sac", "verbose": True})
agent.load(f"saved_runs/{RUN_NAME}")

# Deploy agent in environment.
deploy(agent=agent,
    #    env=gym.make("CartPole-v1"),
       env=env,
       parameters=deploy_parameters,
       observer=observer
       )

# observer.dataframe().to_csv(f"{RUN_NAME}.csv")