from __future__ import division
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import gym

from util import create_env, submit_shared_grads, logger
from model import ActorCritic


def train(rank, args, shared_model, counter, shared_reward, reward_queue, lock, optimizer=None):
    """Training subroutine, for each process to run
    Args:
        - rank: rank of each process, [0, #cpu - 1]
        - args: argument list
            - args.seed
            - args.env_name
            - args.lr
            - args.gamma
        - shared_model
        - counter: the episode(global)
        - lock: mp.lock
        - shared reward(global)
    """
    torch.manual_seed(args.seed + rank)
    # Create environment
    env = create_env(args.env_name)
    env.seed(args.seed + rank)
    # Create model
    model = ActorCritic(env.observation_space.shape[0],
                        env.action_space.shape[0],
                        env.action_space.high[0])
    optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)
    
    state = env.reset().astype(np.float32)
    state = Variable(torch.from_numpy(state).unsqueeze(0))
    done = True

    while counter.value < args.max_episode:
        pi = Variable(torch.FloatTensor([math.pi]))
        # ======Episode Begin=======
        # Sync with shared model
        model.load_state_dict(shared_model.state_dict())
        # Bootstrap after done
        values = []
        log_probs = []
        rewards = []
        entropies = []
        episode_reward = 0
        episode_length = 0
        for step in range(args.max_episode_length):
            if rank == 0 and args.render: 
                env.render()
            episode_length += 1

            mu, sigma, value = model(state) # vector with action_space_n
            sigma = F.softplus(sigma)   # smooth
            eps = torch.randn(mu.size())
            action = (mu + sigma.sqrt() * Variable(eps)).data
            prob = (-1 * (action - mu).pow(2) / (2 * sigma)).exp()
            log_prob = prob.log()
            
            entropy = - 0.5 * ((sigma + 2 * pi.expand_as(sigma)).log()) - 0.5
            entropies.append(entropy)
            
            state, reward, done, _ = env.step(action.numpy())
            state = state.astype(np.float32)
            # Terminal flag
            done = done or episode_length >= args.max_episode_length
            
            # Clip reward
            reward = max(min(reward, 1), -1)
            episode_reward += reward

            if done:
                episode_length = 0
                # Reset state
                state = env.reset().astype(np.float32)
            # Wrapper
            state = Variable(torch.from_numpy(state).unsqueeze(0))
            
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            # End the episode
            if done:
                break
        # ====== After an episode, do logger and fit =======
        R = torch.zeros(1,1)    # Return, as 0 or end_value
        if not done:
            _m, _s, value = model(state)
            R = value.data

        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        adv_estim = torch.zeros(1,1)
        # From rear to head
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i] # Return update
            advantage = R - values[i]
            # Accumulate value loss
            value_loss +=  0.5 * advantage.pow(2)

            td_err = rewards[i] + args.gamma * \
                    values[i + 1] - values[i]
            # Here we use Generalized Advantage Estimataion (GAE)
            # tau = 1.0
            adv_estim = adv_estim * args.gamma * 1.0 + td_err.detach()
            policy_loss = policy_loss - (log_probs[i] * Variable(adv_estim).expand_as(log_probs[i])).sum() \
					- (0.01 * entropies[i]).sum()
        loss = policy_loss + 0.5 * value_loss
        model.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        logger(rank,
               counter,
               shared_reward,
               episode_reward,
               reward_queue)
        # Interaction with shared model
        submit_shared_grads(model, shared_model)
        optimizer.step()

    reward_queue.put(None) # End flag
    env.close()

def test(env_name, path_to_model):
    model = torch.load(path_to_model)
    env = create_env(env_name)
    for episode in range(8):
        obs = env.reset()
        obs = torch.from_numpy(obs.astype(np.float32))
        for step in range(10000):
            env.render()
            mu,sigma,value = model(obs)
            eps = torch.randn(mu.size())
            action = (mu + sigma.sqrt() * Variable(eps)).data
            obs, reward, done, _ = env.step(action)
            obs = torch.from_numpy(obs.astype(np.float32))
            if done:
                break
    env.close()

