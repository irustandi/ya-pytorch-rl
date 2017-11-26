from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque
import torch


class StateProcessor(object):
    def __init__(self, num_obs_in_state, tfs, init_obs):
        self.num_obs_in_state = num_obs_in_state
        self.tfs = tfs
        self.state = deque()
        init_obs_tf = self.tfs(init_obs)
        for _ in range(self.num_obs_in_state):
            self.state.append(init_obs_tf)

    def push_obs(self, obs):
        # update state
        obs_tf = self.tfs(obs)
        self.state.popleft()
        self.state.append(obs_tf)

    def get_state(self):
        state = torch.cat(self.state)

        return state

