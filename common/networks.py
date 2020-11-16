import torch.nn as nn


class SequentialNetwork(nn.Module):
    def __init__(self, layers=None, preset=None, state_shape=None, num_actions=None):
        super(SequentialNetwork, self).__init__() 
        if layers is None: 
            assert preset is not None, "Must specify either layers or preset."
            assert state_shape is not None and num_actions is not None, "Must specify state_shape and num_actions."
            layers = sequential_presets(preset, state_shape, num_actions)
        self.layers = nn.Sequential(*layers)

    def forward(self, x): return self.layers(x)


class MultiHeadedNetwork(nn.Module):
    def __init__(self, common=None, heads=None, preset=None, state_shape=None, num_actions=None):
        super(MultiHeadedNetwork, self).__init__() 
        if common is None: 
            assert preset is not None, "Must specify either layers or preset."
            assert state_shape is not None and num_actions is not None, "Must specify state_shape and num_actions."
            common, heads = multi_headed_presets(preset, state_shape, num_actions)
        self.common = nn.Sequential(*common)
        self.heads = nn.ModuleList([nn.Sequential(*head) for head in heads])

    def forward(self, x): 
        x = self.common(x)
        return tuple(head(x) for head in self.heads)


# ===================================================================
# PRESETS.


def sequential_presets(name, state_shape, num_actions):

    if name == "CartPoleDQNPixels":
        # From https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(state_shape[3])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(state_shape[2])))
        linear_input_size = convw * convh * 32
        return [
            nn.Conv2d(state_shape[1], 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(linear_input_size, num_actions)
        ]

    if name == "CartPoleDQNVector":
        # From https://github.com/transedward/pytorch-dqn/blob/master/dqn_model.py.
        return [
            nn.Linear(state_shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        ]

    if name == "CartPoleReinforcePixels":
        # Just added Softmax to DQN version.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(state_shape[3])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(state_shape[2])))
        linear_input_size = convw * convh * 32
        return [
            nn.Conv2d(state_shape[1], 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(linear_input_size, num_actions),
            nn.Softmax(dim=1)
        ]

    if name == "CartPoleReinforceVector":
        # From https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py. 
        return [
            nn.Linear(state_shape[0], 128), 
            nn.Dropout(p=0.6),
            nn.ReLU(),
            nn.Linear(128, num_actions),
            nn.Softmax(dim=1)
        ]

    raise NotImplementedError("Invalid preset name.")


def multi_headed_presets(name, state_shape, num_actions):

    if name == "CartPoleReinforceVectorWithBaseline":
        # From https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py. 
        # and https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py.
        # (The latter erroneously describes the model as actor-critic; it's REINFORCE with baseline!)
        return ([ # Common.
            nn.Linear(state_shape[0], 128), 
            nn.Dropout(p=0.6),
            nn.ReLU()
        ], 
        [
            [ # Policy head.
                nn.Linear(128, num_actions),
                nn.Softmax(dim=1)
            ],
            [ # Value head.
                nn.Linear(128, 1)
            ]
        ])

    raise NotImplementedError("Invalid preset name.")