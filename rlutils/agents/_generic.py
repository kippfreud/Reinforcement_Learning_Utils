import torch


class Agent:
    def __init__(self, env, hyperparameters):
        """
        Base agent class. All other agents inherit from this.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.P = hyperparameters 
        assert len(self.env.action_space.shape) in {0,1}

    def __str__(self):
        P = "\n".join([f"| - {k} = {v}" for k, v in self.P.items()])
        return f"\n| {self.__class__.__name__} in {self.env} with hyperparameters:\n{P}\n"

    def per_timestep(self, state, action, reward, next_state, done): pass

    def per_episode(self): return {"logs": {}}

    def save(self, path, clear_memory=True): 
        # Remove env for saving; stops pickle from throwing an error.
        env = self.env; self.env = None 
        if clear_memory: 
            try: memory = self.memory; self.memory = None
            except: pass # If no memory.
        torch.save(self, f"{path}.agent")
        self.env = env
        if clear_memory: 
            try: self.memory = memory
            except: pass