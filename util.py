"""Util
"""
import numpy as np
import mujoco_py
import gym


# Utils
# --------------------------

def weight_init(m):
    """Initialize the network
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


def submit_shared_grads(model, shared_model):
    """ Pass the grads of Parameters in model ==> shared_model
    """
    for local_param, shared_param in zip(model.parameters(),
                                        shared_model.parameters()):
        if shared_param.grad is not None:   # Do nothing, for security
            return
        shared_param._grad = local_param.grad

def create_env(env_name):
    return gym.make(env_name)

def logger(rank,
           shared_episode,
           shared_reward,
           episode_reward, 
           reward_queue):
    """Logger Function, print out the iteration info
    Args:
        shared_episode: (shared) shared var, record the number of past episode
        episode_reward: (local) record how many rewards gained in this local agent episode(float)
        reward_queue: for plotting, record the shared_ep_reward.value into list
    """
    # Update the total episode(shared)
    with shared_episode.get_lock():
        shared_episode.value += 1
    # Update the episode reward (shared)
    with shared_reward.get_lock():
        if shared_reward.value == 0.:
            shared_reward.value = episode_reward
        else:
            shared_reward.value =  episode_reward * 0.3 + shared_reward.value * 0.7
    reward_queue.put(shared_reward.value)
    print("rank:{} | Episode:{} | Episode_reward: {:.0f}".format(
            rank, shared_episode.value, episode_reward))
