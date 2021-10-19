import torch


def col_concat(x, y):
    """Concatenate x and y along the second (column) dimension."""
    return torch.cat([x, y], 1).float()

def one_hot(idx, len, device):
    """Create a tensor of length len which is one-hot at idx."""
    assert type(idx) == int and 0 <= idx < len
    return torch.tensor([[1 if i == idx else 0 for i in range(len)]], device=device, dtype=torch.float)

def edit_output_size(net, mode:str, A:int, x:int=None, frac:list=None, x0:int=None, x1:int=None):
    """
    Edit the output size of a network by splitting or merging.
    TODO: Complete an arbitrary number of splits and merges with a single call.
    """
    with torch.no_grad():
        Am, n = net.layers[-1].weight.shape; m = int(Am / A)
        if mode == "split": assert 0 <= x < m; assert len(frac) > 1; assert sum(frac) == 1; m_new = m+len(frac)-1
        elif mode == "merge": assert 0 <= x0 < m-1; assert x0 < x1 < m; m_new = m-(x1-x0) 
        weights = net.layers[-1].weight.data.reshape(A, m, n)
        bias = net.layers[-1].bias.data.reshape(A, m)
        net.layers[-1] = torch.nn.Linear(n, A*m_new)
        if mode == "split":
            weights_x, bias_x, x0, x1 = weights[:,x,:].reshape(A, 1, n), bias[:,x].reshape(A, 1), x, x
            insert_w = torch.cat(tuple(weights_x * f for f in frac), axis=1)
            insert_b = torch.cat(tuple(bias_x * f for f in frac), axis=1)
        elif mode == "merge":
            insert_w = weights[:,x0:x1+1,:].sum(axis=1).reshape(A, 1, n)
            insert_b = bias[:,x0:x1+1].sum(axis=1).reshape(A, 1)
        net.layers[-1].weight = torch.nn.Parameter(torch.cat((weights[:,:x0,:], insert_w, weights[:,x1+1:,:]), axis=1).reshape(A*m_new, n))
        net.layers[-1].bias = torch.nn.Parameter(torch.cat((bias[:,:x0], insert_b, bias[:,x1+1:]), axis=1).reshape(A*m_new))