import numpy as np
import pandas as pd


"""
TODO: 
    "mean" mode in .add_future()
    .add_custom_dims(function, dim_names=[""])
    .add_continuous_actions(mapping) - just a special case of the above?
"""


PROTECTED_DIM_NAMES = {"step", "ep", "time", "reward", "pi", "Q", "V"}


class Observer:
    """
    This is a class for collecting observational data of an agent during deployment.
    """
    def __init__(self, state_dim_names, action_dim_names):
        # Check for protected dim_names.
        illegal = PROTECTED_DIM_NAMES & set(state_dim_names + action_dim_names)
        if illegal: raise ValueError(f"dim_names {illegal} already in use.")
        # List the dimensions in the dataset to be constructed.
        self.dim_names = ["ep", "time"] + state_dim_names + action_dim_names + ["reward"] \
                       + [f"n_{d}" for d in state_dim_names]
        self.num_actions = len(action_dim_names)
        # Initialise empty dataset.
        self.data, self.empty = [], True

    def observe(self, ep, t, state, action, reward, next_state, info, extra):
        """Make an observation of a single timestep."""
        if self.empty: extra_dim_names = []
        # Basics: state, action, reward, next_state.
        observation = [ep, t] \
                    + list(state.numpy().flatten()) \
                    + list([action] if self.num_actions == 1 else list(action)) \
                    + [reward] \
                    + list(next_state.flatten()) # Already in NumPy format.
        # Dictionaries containing extra information produced by agent and environment.
        for k,v in {**info, **extra}.items():
            if type(v) == np.ndarray: v = list(v.flatten()); multi = True
            else: v = [v]; multi = False
            observation += v
            if self.empty: 
                if multi: extra_dim_names += [f"{k}_{i}" for i in range(len(v))]
                else: extra_dim_names += [k]
        self.data.append(observation)
        # Add extra dim names.
        if self.empty:
            illegal = set(self.dim_names) & set(extra_dim_names)
            if illegal: raise ValueError(f"dim_names {illegal} already in use.")
            self.dim_names += extra_dim_names
            self.empty = False

    def dataframe(self):
        df = pd.DataFrame(self.data, columns=self.dim_names)
        df.index.name = "step"
        return df

    def add_future(self, dims, gamma, mode="sum", new_dims=None):
        """
        Add dimensions to the dataset corresponding to the discounted sum of existing ones.
        """
        self.data = np.array(self.data)
        data_time = self.data[:,self.dim_names.index("time")]
        data_dims = self.data[:,[self.dim_names.index(d) for d in dims]]
        data_new_dims = np.zeros_like(data_dims)
        terminal = True
        for i, (t, x) in enumerate(reversed(list(zip(data_time, data_dims)))):
            if terminal: data_new_dims[i] = x
            else: data_new_dims[i] = x + (gamma * data_new_dims[i-1])
            if t == 0: terminal = True
            else: terminal = False
        self.data = np.hstack((self.data, np.flip(data_new_dims, axis=0)))
        if not new_dims: new_dims = [f"future_{mode}_{d}" for d in dims]
        self.dim_names += new_dims

    def add_derivatives(self, dims, new_dims=None): 
        """
        Add dimensions to the dataset corresponding to the change in existing dimensions
        between successive timesteps.
        """
        self.data = np.array(self.data)
        data_time = self.data[:,self.dim_names.index("time")]
        data_dims = self.data[:,[self.dim_names.index(d) for d in dims]]
        data_new_dims = np.zeros_like(data_dims)
        for i in range(len(self.data)-1):
            # NOTE: Just repeat last derivatives for terminal.
            if data_time[i+1] == 0: data_new_dims[i] = data_new_dims[i-1]             
            else: data_new_dims[i] = data_dims[i+1] - data_dims[i]
        self.data = np.hstack((self.data, data_new_dims))
        if not new_dims: new_dims = [f"d_{d}" for d in dims]
        self.dim_names += new_dims