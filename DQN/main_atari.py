from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import torch
import random
import argparse
from model import DQN
from utils import Transition, ReplayMemory, get_transforms
from state_processor import StateProcessor
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


def select_action(model, state, num_actions, eps, use_cuda):
    sample = random.random()
    if sample > eps:
        state = torch.unsqueeze(state, dim=0)
        state_var = Variable(state, volatile=True)
        if use_cuda:
            state_var = state_var.cuda()
        model_output = model(state_var)
        action = model_output.data.max(1)[1].view(1, 1).cpu()

        return action
    else:
        action = random.randrange(num_actions)
        action = torch.LongTensor([[action]])
        return action

def optimize_model(optimizer, memory, model, model_target, batch_size, gamma, use_cuda):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)))

    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = Variable(torch.stack([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)
    state_batch = Variable(torch.stack(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.stack(batch.reward))

    if use_cuda:
        non_final_mask = non_final_mask.cuda()
        non_final_next_states = non_final_next_states.cuda()
        state_batch = state_batch.cuda()
        action_batch = action_batch.cuda()
        reward_batch = reward_batch.cuda()

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(batch_size, 1).type(torch.Tensor))
    if use_cuda:
        next_state_values = next_state_values.cuda()

    next_state_values[non_final_mask] = model_target(non_final_next_states).max(1)[0]
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    loss = F.mse_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def main():
    parser = argparse.ArgumentParser(description='DQN Breakout Script')
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='whether to use CUDA (default: False)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='M',
                        help='batch size (default: 128)')
    parser.add_argument('--gamma', type=float, default=0.999, metavar='M',
                        help='gamma (default: 0.999)')
    parser.add_argument('--eps-start', type=float, default=0.9, metavar='M',
                        help='eps start (default: 0.9)')
    parser.add_argument('--eps-end', type=float, default=0.05, metavar='M',
                        help='eps end (default: 0.05)')
    parser.add_argument('--eps-decay', type=int, default=200, metavar='M',
                        help='eps decay (default: 200)')
    parser.add_argument('--num-obs-in-state', type=int, default=4, metavar='M',
                        help='num observations in state (default: 4)')
    parser.add_argument('--replay-memory-capacity', type=int, default=10000, metavar='M',
                        help='replay memory capacity (default: 10000)')
    parser.add_argument('--num-episodes', type=int, default=10, metavar='M',
                        help='num of episodes (default: 10)')
    parser.add_argument('--reset-period', type=int, default=5, metavar='M',
                        help='period to reset target network (default: 5)')
    parser.add_argument('--atari-env', type=str, default='Breakout-v0', metavar='M',
                        help='Atari environment to use (default: Breakout-v0)')
    args = parser.parse_args()

    env = gym.envs.make(args.atari_env)

    model = DQN(args.num_obs_in_state, (84, 84), env.action_space.shape[0])
    model_target = DQN(args.num_obs_in_state, (84, 84), env.action_space.shape[0])

    if args.use_cuda:
        model.cuda()
        model_target.cuda()

    optimizer = optim.RMSprop(model.parameters())
    memory = ReplayMemory(args.replay_memory_capacity)

    epsilons = np.linspace(args.eps_start, args.eps_end, args.eps_decay)
    step_idx = 1
    reset_idx = 1

    tfs = get_transforms()

    episode_reward = 0.
    episode_length = 0

    for i_episode in range(args.num_episodes):
        # Initialize the environment and state
        obs = env.reset()
        state_processor = StateProcessor(args.num_obs_in_state, tfs, obs)
        state = state_processor.get_state()

        while True:
            episode_length += 1
            if step_idx < args.eps_decay:
                eps = epsilons[step_idx]
            else:
                eps = args.eps_end

            action = select_action(model, state, env.action_space.shape[0], eps, args.use_cuda)
            # print('%d %d' % (episode_length, action[0,0]))
            next_obs, reward, done, info = env.step(action[0, 0])
            episode_reward += reward
            reward = torch.Tensor([reward])
            if args.use_cuda:
                reward = reward.cuda()

            if not done:
                state_processor.push_obs(next_obs)
                next_state = state_processor.get_state()
            else:
                next_state = None # None next_state marks done

            memory.push(state, action, next_state, reward)

            # optimize
            optimize_model(optimizer, memory, model, model_target, args.batch_size, args.gamma, args.use_cuda)

            step_idx += 1
            reset_idx += 1
            if reset_idx == args.reset_period:
                reset_idx = 1
                model_target.load_state_dict(model.state_dict())

            if done:
                break

        print(episode_reward)
        print(episode_length)
        episode_reward = 0.
        episode_length = 0


if __name__ == '__main__':
    main()
