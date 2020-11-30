import torch
import numpy as np

# # Providing an output tensor as an argument
# x = torch.tensor([1, 2])
# y = torch.tensor([3, 4])
# z = torch.empty_like(x)
# torch.add(x, y, out=z)

# # Converting to/from a NumPy array
# a = torch.ones(5)
# b = a.numpy()
# a.add_(1) # Modifying one modifies the other; memory shared
# a = torch.from_numpy(b)
# print(a) 

# # The requires_grad option tells PyTorch to track all operations on a tensor, 
# # allowing for automatic differentiation 
# x = torch.ones(2, 2, requires_grad=True)
# y = x + 2
# print(y.grad_fn) # y was created as a result of an operation, so it has a grad_fn.
# z = y * y * 3
# out = z.mean()
# print(z, out)

# # Backprop using the torch.autograd engine
# out.backward()
# print(x.grad)

# The Jacobian of gradients can be implicitly created only for scalar outputs,
# but for vector outputs we can compute its *product* with a vector provided as an argument
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
y.backward(torch.Tensor([0.1, 1.0, 0.0001]))
print(x.grad)

# Detaching creates a new Tensor with the same content but that does not require gradients
z = x.detach()