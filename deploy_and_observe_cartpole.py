from rlutils.common.deployment import deploy
from rlutils.common.observer import Observer

import gym
from joblib import load

RUN_NAME = "cosmic-yogurt-123"

deploy_parameters = {
    "num_episodes": 2,
    "max_timesteps_per_episode": 500,
    "wandb_monitor": False,
    "render": False
}

# Create observer class.
observer = Observer(
                    state_dim_names=["pos", "vel", "ang", "vel_ang"],
                    action_dim_names=["action_idx"]
                    )

# Deploy agent in environment.
deploy(agent=load(f"{RUN_NAME}.joblib"),
       env=gym.make("CartPole-v1"),
       parameters=deploy_parameters,
       observer=observer
       )

print(observer.dataframe())