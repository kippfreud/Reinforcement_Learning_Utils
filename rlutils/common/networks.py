import torch
import torch.nn as nn
import torch.optim as optim


class SequentialNetwork(nn.Module):
    def __init__(self, layers=None, code=None, preset=None, input_shape=None, output_size=None,
                 eval_only=False, optimiser=optim.Adam, lr=1e-3, clip_grads=False):
        """
        Net codes:
        - "R"                 = ReLU
        - "T"                 = Tanh
        - "S"                 = Softmax
        - ("D", p)            = Dropout
        - ("B", num_features) = Batch norm
        """
        super(SequentialNetwork, self).__init__() 
        if layers is None: 
            assert input_shape is not None and output_size is not None, "Must specify input_shape and output_size."
            if code is not None: layers = code_parser(code, input_shape, output_size)
            else:
                assert preset is not None, "Must specify layers, code or preset."
                layers = sequential_presets(preset, input_shape, output_size)
        self.layers = nn.Sequential(*layers)
        if eval_only: self.eval()
        else: 
            self.optimiser = optimiser(self.parameters(), lr=lr)
            self.clip_grads = clip_grads

    def forward(self, x): return self.layers(x)

    def optimise(self, loss, do_backward=True, retain_graph=True): 
        assert self.training, "Network is in eval_only mode."
        if do_backward: 
            self.optimiser.zero_grad()
            loss.backward(retain_graph=retain_graph) 
        if self.clip_grads: # Optional gradient clipping.
            for param in self.parameters(): param.grad.data.clamp_(-1, 1) 
        self.optimiser.step()

    def polyak(self, other, tau):
        """
        Use Polyak averaging to blend parameters with those of another network.
        """
        for self_param, other_param in zip(self.parameters(), other.parameters()):
            self_param.data.copy_((other_param.data * tau) + (self_param.data * (1.0 - tau)))


def code_parser(code, input_shape, output_size):
    layers = []
    for l in code:
        if type(l) in {list, tuple}:   
            i, o = l[0], l[1]
            if i is None: i = input_shape # NOTE: Only works for vectors at the moment.
            if o is None: o = output_size 
            layers.append(nn.Linear(i, o))
        elif l == "R":          layers.append(nn.ReLU())
        elif l == "T":          layers.append(nn.Tanh())
        elif l == "S":          layers.append(nn.Softmax(dim=1))
        elif l[0] == "D":       layers.append(nn.Dropout(p=l[1]))
        elif l[0] == "B":       layers.append(nn.BatchNorm2d(l[1]))
    return layers


# ===================================================================
# TREE NETWORK


class TreeNetwork(nn.Module):
    def __init__(self, code, input_shape, num_actions,
                 eval_only=False, optimiser=optim.Adam, lr=1e-3):
        super(TreeNetwork, self).__init__() 
        assert type(code[-1][0]) == int and code[-1][1] is None, "Must have linear final layer."
        self.code, self.input_shape, self.num_actions, self.eval_only, self.optimiser, self.lr = code, input_shape, num_actions, eval_only, optimiser, lr
        self.root = self.node() # Leaf()
        self.horizon = SequentialNetwork(code=self.code, input_shape=self.input_shape, output_size=self.num_actions,
                                         eval_only=self.eval_only, optimiser=self.optimiser, lr=self.lr) 

    def __call__(self, x): 
        if len(x.shape) == 1: x = x[None,:] # Handle single inputs.
        h = self.horizon(x)[:,:,None]
        return h if type(self.root) == Leaf else self.root(x).tensor() * h 

    @property
    def m(self): 
        def _recurse(node): 
            return 1 if node.left is None else _recurse(node.left) + _recurse(node.right)
        return _recurse(self.root)

    def state_dict(self): return (self.horizon.state_dict(), self.root.gsd())
    def load_state_dict(self, d): self.horizon.load_state_dict(d[0]); self.root.lsd(d[1])

    def polyak(self, other, tau): 
        def _recurse(node, other_node): 
            if node.left is not None: 
                node.net.polyak(other_node.net, tau)
                _recurse(node.left, other_node.left); _recurse(node.right, other_node.right)
        _recurse(self.root, other.root)
        self.horizon.polyak(other.horizon, tau)

    def node(self):
        # Multiply output by branching factor.
        return Node(code=self.code, input_shape=self.input_shape, output_size=2*self.num_actions, 
                    eval_only=self.eval_only, optimiser=self.optimiser, lr=self.lr) 

    def optimise(self, loss, **kwargs):
        self.zero_grad()
        loss.backward()
        self.root.optimise(loss, **kwargs)
        self.horizon.optimise(loss, do_backward=False)

    def zero_grad(self):
        def _recurse(node): 
            if node.left is not None: node.net.zero_grad(); _recurse(node.left); _recurse(node.right)
        _recurse(self.root)
        self.horizon.zero_grad()


class Node:
    def __init__(self, **net_params): 
        self.net = SequentialNetwork(**net_params)
        self.left, self.right = Leaf(), Leaf()
    
    def __call__(self, x): 
        # NOTE: Apply reshape and softmax here to get probabilities.
        P = nn.functional.softmax(self.net(x).reshape(x.shape[0], -1, 2), dim=2)
        return Output((P, self.left(x), self.right(x)))

    # Get and load state dicts.
    def gsd(self): return (self.net.state_dict(), self.left.gsd(), self.right.gsd())
    def lsd(self, d): self.net.load_state_dict(d[0]); self.left.lsd(d[1]); self.right.lsd(d[2]) 

    def optimise(self, loss, **kwargs):
        self.net.optimise(loss, do_backward=False, **kwargs)
        self.left.optimise(loss, **kwargs)
        self.right.optimise(loss, **kwargs)


class Leaf:
    def __call__(_, __): return None
    def gsd(_): return None
    def lsd(_, __): pass
    def optimise(_, __): pass

    # This prevents setting left and right before turning into an internal node.
    @property
    def left(_): return None
    @property
    def right(_): return None 


class Output:
    def __init__(self, tup): 
        self._tuple = tup

    def __str__(self): return self._print()

    def _print(self, indent=0):
        tab = "    "
        P, subtree_l, subtree_r = self._tuple
        return str(P[:,:,0].detach().numpy().tolist()) + "\n" \
            + ((tab * (indent+1)) + subtree_l._print(indent+1) if subtree_l is not None else "") \
            + (tab * indent) + str(P[:,:,1].detach().numpy().tolist()) + "\n" \
            + ((tab * (indent+1)) + subtree_r._print(indent+1) if subtree_r is not None else "") \

    def tensor(self):
        P, subtree_l, subtree_r = self._tuple
        Ps_l, Ps_r = P[:,:,0:1], P[:,:,1:2] # "One-element slice" to keep dims. 
        if subtree_l is not None: Ps_l = Ps_l * subtree_l.tensor()
        if subtree_r is not None: Ps_r = Ps_r * subtree_r.tensor()
        tensor = torch.cat([Ps_l, Ps_r], dim=2)
        assert torch.isclose(tensor.sum(axis=2), torch.tensor(1.)).all()
        return tensor


# ===================================================================
# MULTI-HEADED NETWORK


# class MultiHeadedNetwork(nn.Module):
#     def __init__(self, common=None, heads=None, preset=None, input_shape=None, output_size=None):
#         super(MultiHeadedNetwork, self).__init__() 
#         if common is None: 
#             assert preset is not None, "Must specify either layers or preset."
#             assert input_shape is not None and output_size is not None, "Must specify input_shape and output_size."
#             common, heads = multi_headed_presets(preset, input_shape, output_size)
#         self.common = nn.Sequential(*common)
#         self.heads = nn.ModuleList([nn.Sequential(*head) for head in heads])

#     def forward(self, x): 
#         x = self.common(x)
#         return tuple(head(x) for head in self.heads)


# ===================================================================
# PRESETS


def sequential_presets(name, input_shape, output_size):

    if name == "CartPolePi_Pixels":
        # Just added Softmax to Q version.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[3])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[2])))
        linear_input_size = convw * convh * 32
        return [
            nn.Conv2d(input_shape[1], 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(linear_input_size, output_size),
            nn.Softmax(dim=1)
        ]

    # if name == "CartPolePi_Vector":
    #     return [
    #         nn.Linear(input_shape[0], 64), 
    #         nn.ReLU(),
    #         nn.Linear(64, 128), 
    #         # nn.Dropout(p=0.6),
    #         nn.ReLU(),
    #         nn.Linear(128, output_size),
    #         nn.Softmax(dim=1)
    #     ]

    if name == "CartPoleQ_Pixels":
        # From https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[3])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[2])))
        linear_input_size = convw * convh * 32
        return [
            nn.Conv2d(input_shape[1], 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(linear_input_size, output_size)
        ]

    # if name == "CartPoleQ_Vector":
    #     # From https://github.com/transedward/pytorch-dqn/blob/master/dqn_model.py.
    #     return [
    #         nn.Linear(input_shape[0], 256),
    #         nn.ReLU(),
    #         nn.Linear(256, 128),
    #         nn.ReLU(),
    #         nn.Linear(128, 64),
    #         nn.ReLU(),
    #         nn.Linear(64, output_size)
    #     ]

    if name == "CartPoleV_Pixels":
        # Just change Q version to have one output node.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[3])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[2])))
        linear_input_size = convw * convh * 32
        return [
            nn.Conv2d(input_shape[1], 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(linear_input_size, 1)
        ]

    # if name == "CartPoleV_Vector":
    #     return [
    #         nn.Linear(input_shape[0], 64), 
    #         nn.ReLU(),
    #         nn.Linear(64, 128), 
    #         # nn.Dropout(p=0.6),
    #         nn.ReLU(),
    #         nn.Linear(128, 1)
    #     ]

    # if name == "PendulumPi_Vector":
    #     # From https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b.
    #     return [
    #         nn.Linear(input_shape[0], 256),
    #         nn.ReLU(),
    #         nn.Linear(256, 256),
    #         nn.ReLU(),
    #         nn.Linear(256, output_size),
    #         nn.Tanh()
    #     ]

    # if name == "PendulumQ_Vector":
    #     return [
    #         nn.Linear(input_shape[0], 256),
    #         nn.ReLU(),
    #         nn.Linear(256, 256),
    #         nn.ReLU(),
    #         nn.Linear(256, output_size)
    #     ]

    # if name == "StableBaselinesPi_Vector":
    #     # From https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b.
    #     return [
    #         nn.Linear(input_shape[0], 64),
    #         nn.ReLU(),
    #         nn.Linear(64, 64),
    #         nn.ReLU(),
    #         nn.Linear(64, output_size),
    #         nn.Tanh()
    #     ]

    # if name == "StableBaselinesQ_Vector":
    #     return [
    #         nn.Linear(input_shape[0], 64),
    #         nn.ReLU(),
    #         nn.Linear(64, 64),
    #         nn.ReLU(),
    #         nn.Linear(64, output_size)
    #     ]

    raise NotImplementedError(f"Invalid preset name {name}.")


# def multi_headed_presets(name, input_shape, output_size):

#     if name == "CartPolePiAndV_Vector":
#         # From https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py. 
#         # and https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py.
#         # (The latter erroneously describes the model as actor-critic; it's REINFORCE with baseline!)
#         return ([ # Common.
#             nn.Linear(input_shape[0], 128), 
#             nn.Dropout(p=0.6),
#             nn.ReLU()
#         ], 
#         [
#             [ # Policy head.
#                 nn.Linear(128, output_size),
#                 nn.Softmax(dim=1)
#             ],
#             [ # Value head.
#                 nn.Linear(128, 1)
#             ]
#         ])

#     raise NotImplementedError(f"Invalid preset name {name}.")