"""Reinforment learning A3c-mujoco
CS489 project
Author: Lu Jiarui 
Date: 2020/06/11
"""

from __future__ import print_function

import argparse
import os

import torch
import torch.multiprocessing as mp
import gym

from train import train, test
from util import create_env
from model import ActorCritic
from shared_adam import SharedAdam 
import matplotlib.pyplot as plt


# Training setting
parser = argparse.ArgumentParser(description='A3C-Mujoco')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=4,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--max-episode', type=int, default=1000,
                    help='number of episodes in training (default: 1000)')
parser.add_argument('--max-episode-length', type=int, default=2000,
                    help='maximum length of an episode (default: 2000)')
parser.add_argument('--env-name', type=str, default='Ant-v2',
                    help='environment to train on or test (default: Ant-v2)')
parser.add_argument('--render', type=bool, default=True,
                    help='visualize render for rank 0')
parser.add_argument('--do-test', type=bool, default=False,
                    help='do test and load local checkpoint as model.ckpt')
parser.add_argument('--init-checkpoint', type=str, default=None, 
                    help='initial checkpoint of AC model, pytorch chenckpoint')

if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    print("[INFO] Using {} CPUs".format(mp.cpu_count()))

    args = parser.parse_args()

    if not args.do_test:
        print('[INFO] Enter training subroutine. (Default: train)')
        # Global networks
        torch.manual_seed(args.seed)
        env = create_env(args.env_name)
        shared_model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0])
        shared_model.share_memory()
        optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

        # multiprocess
        processes = []
        counter, shared_reward, reward_queue = \
            mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
        lock = mp.Lock()

        for rank in range(args.num_processes):
            p = mp.Process(target=train, args=(rank, args, shared_model, counter, shared_reward, reward_queue, lock, optimizer))
            p.start()
            processes.append(p)
        
        reward_result = []
        while True:
            r = reward_queue.get()
            if r is not None:
                reward_result.append(r)
            else:
                break   # End flag
        # join
        for p in processes:
            p.join()

        # Saving the results
        torch.save(shared_model, 'model-{}-{}.ckpt'.format(args.env_name, counter.value))
        plt.plot(reward_result)
        plt.ylabel('Moving average episode reward')
        plt.xlabel('Episode')
        plt.savefig('A3c-' + args.env_name + '-reward.png')
        plt.close()
    # Test
    if args.do_test:
        print('[INFO] Enter testing subroutine.)
        test(args.env_name)
        
