import torch


class HrRewardsAndCosts:
    def __init__(self, subsets, weights): 
        """
        Assigns reward/cost components to m hyperrectangular subsets of the state-action space S x A.
        """
        self.subsets, self.weights = torch.tensor(subsets)[None,:,:,:].detach(), torch.tensor(weights).detach()
        assert len(weights) == self.m
        assert self.subsets.shape[3] == 2
        assert (self.subsets[:,:,:,1] >= self.subsets[:,:,:,0]).all()

    @property
    def m(self): return self.subsets.shape[1]
    @property
    def d(self): return self.subsets.shape[2]

    def __str__(self): return f"- Subsets:\n{self.subsets.numpy()}\n- Weights: {self.weights.numpy()}"
    
    def __call__(self, states, actions): return self.phi(states, actions)*self.weights

    def phi(self, states, _):
        """
        Map a batch of state-action pairs to vectors in {0,1}^m, indicating occupancy.
        NOTE: Currently only uses states!
        """
        return (((states[:,None,:] < self.subsets[:,:,:,0]) | (states[:,None,:] > self.subsets[:,:,:,1])).sum(axis=2) == 0).type(torch.float32)