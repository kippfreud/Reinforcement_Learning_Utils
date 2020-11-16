import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from methods import *

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--eps_start', type=float, default=0.9, metavar='N',
                    help='start of eps')
parser.add_argument('--eps_end', type=float, default=0.05, metavar='N',
                    help='end of eps')
parser.add_argument('--eps_decay', type=float, default=10000, metavar='N',
                    help='decay of eps')
parser.add_argument('--batch_size', type=float, default=128, metavar='N',
                    help='size of batch')
parser.add_argument('--target_update', type=float, default=10, metavar='N',
                    help='lag of target net')
args = parser.parse_args()

env = gym.make('CartPole-v1')
env.seed(args.seed)
torch.manual_seed(args.seed)
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

policy = QNet()
target = QNet()
target.load_state_dict(policy.state_dict())
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

def main():
    #agent = ReinforcementAgent(env, policy, optimizer, args, device)
    agent = DQNAgent(env, policy, target, optimizer, args, device)
    agent.reinforce()


if __name__ == '__main__':
    main()