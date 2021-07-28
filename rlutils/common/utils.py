import torch

def col_concat(x, y):
    """Concatenate x and y along the second (column) dimension."""
    return torch.cat([x, y], 1).float()

def one_hot(idx, len, device):
    assert type(idx) == int and 0 <= idx < len
    return torch.tensor([[1 if i == idx else 0 for i in range(len)]], device=device, dtype=torch.float)