from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, in_channels, input_size, num_actions):
        super(DQN, self).__init__()
        x = torch.autograd.Variable(torch.from_numpy(np.random.randn(1, in_channels, input_size[0], input_size[1]).astype(np.float32)))
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        x = self.conv1(x)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        x = self.conv2(x)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        x = self.conv3(x)

        nchannels = x.size()[1]
        rowdim = x.size()[2]
        coldim = x.size()[3]

        self.fc1 = nn.Linear(nchannels * rowdim * coldim, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        out = self.fc2(x)
        return out
