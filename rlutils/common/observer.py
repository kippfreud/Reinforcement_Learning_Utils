import numpy as np
import pandas as pd


PROTECTED_DIM_NAMES = {"ep", "time", "reward", "pi", "Q", "V"}


class Observer:
    """
    This is a class for collecting observational data of an agent during deployment.
    """
    def __init__(self, state_dim_names, action_dim_names):
        # Check for protected dim_names.
        illegal = PROTECTED_DIM_NAMES & set(state_dim_names + action_dim_names)
        if illegal: raise Exception(f"dim_names {illegal} illegal because they are protected.")
        # List the dimensions in the dataset to be constructed.
        self.dim_names = ["ep", "time"] + state_dim_names + action_dim_names + ["reward"]
        # Initialise empty dataset.
        self.data = []

    def observe(self, ep, t, state, action, reward, next_state, extra):
        # Basics: state, action and reward.
        observation = [ep, t] \
                    + list(state.numpy().flatten()) \
                    + list(action.numpy().flatten()) \
                    + [reward]
        # Extra information produced by the agent in addition to its action.
        for k,v in extra.items():
            if type(v) == np.ndarray: v = list(v.flatten())
            else: v = [v]
            observation += v
            if len(self.data) == 0:
                # Complete the list of dim_names.
                if k in ("pi", "Q"): self.dim_names += [f"{k}_{i}" for i in range(len(v))]
                else: self.dim_names += [k]
        self.data.append(observation)

    def dataframe(self):
        return pd.DataFrame(self.data, columns=self.dim_names)

    """
    TODO: Methods for augmenting the dataset.
        .add_continuous_actions(mapping)
        .add_discounted_sums(dims=["reward"], gamma=0.99)
        .add_custom_dims(function, dim_names=[""])
        .add_derivatives(dims=["xxx"])
    """