from unittest import TestCase

import gym
import random
import utils, state_processor


class TestStateProcessor(TestCase):
    def setUp(self):
        self.env = gym.envs.make("Breakout-v0")
        self.tfs = utils.get_transforms()
        self.num_obs_in_state = 4

    def test_init(self):
        num_obs_in_state = 4
        obs = self.env.reset()
        sp = state_processor.StateProcessor(num_obs_in_state, self.tfs, obs)
        state = sp.get_state()

        self.assertEqual(len(state), num_obs_in_state)

    def test_push(self):
        num_obs_in_state = 4
        obs = self.env.reset()
        sp = state_processor.StateProcessor(num_obs_in_state, self.tfs, obs)
        action = random.randrange(len(self.env.action_space))
        obs, _, _, _ = self.env.step(action)
        sp.push_obs(obs)
        state = sp.get_state()

        self.assertEqual(len(state), num_obs_in_state)


