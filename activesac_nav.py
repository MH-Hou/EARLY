import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

# from IPython.display import clear_output
# import matplotlib.pyplot as plt
# from matplotlib import animation
# from IPython.display import display
from datetime import datetime
import os
from sklearn.neighbors import KernelDensity
from time import sleep
import argparse


import warnings

warnings.filterwarnings("ignore")

from nav_env import NavEnv
from nav_oracle_rrt import RRT_Oracle
from reply_buffer import PrioritizedReplayBuffer
from scripts.joystick import Joystick



TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
# device   = torch.device("cpu")
print(device)
# device = torch.device("mps")


''' Utility Functions '''
class ReplayBufferOld:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity # max buffer length for the learner buffer
        self.learner_buffer = []
        self.teacher_buffer = []
        self.learner_buffer_idx = 0

    def push(self, state, action, reward, next_state, done, role):
        if role == 'teacher':
            self.teacher_buffer.append((state, action, reward, next_state, done))
        else:
            if len(self.learner_buffer) < self.capacity:
                self.learner_buffer.append(None)
                # self.learner_buffer[self.learner_buffer_idx] = (state, action, reward, next_state, done)
                # self.learner_buffer_idx = (self.learner_buffer_idx + 1) % self.capacity
            # else:
            #     self.learner_buffer_idx = np.random.choice(self.capacity)
            #     self.learner_buffer[self.learner_buffer_idx] = (state, action, reward, next_state, done)

            self.learner_buffer[self.learner_buffer_idx] = (state, action, reward, next_state, done)
            self.learner_buffer_idx = (self.learner_buffer_idx + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.learner_buffer) == 0:
            teacher_batch = random.sample(self.teacher_buffer, batch_size)
            batch = [*teacher_batch]
        elif len(self.teacher_buffer) == 0:
            learner_batch = random.sample(self.learner_buffer, batch_size)
            batch = [*learner_batch]
        else:
            if len(self.learner_buffer) < int(batch_size/2):
                learner_batch = random.choices(self.learner_buffer, k=int(batch_size / 2))
            else:
                learner_batch = random.sample(self.learner_buffer, int(batch_size/2))

            if len(self.teacher_buffer) < int(batch_size/2):
                teacher_batch = random.choices(self.teacher_buffer, k=int(batch_size / 2))
            else:
                teacher_batch = random.sample(self.teacher_buffer, int(batch_size/2))

            batch = [*learner_batch, *teacher_batch]

        # batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def empty(self):
        self.learner_buffer = []
        self.teacher_buffer = []
        self.learner_buffer_idx = 0

    def __len__(self):
        buffer_length = len(self.teacher_buffer) + len(self.learner_buffer)

        return buffer_length

class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


''' Functions for constructing neural network '''
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3, seed=3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, seed=3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2, seed=3):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z.to(device))
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample().to(device)
        action = torch.tanh(mean + std * z)

        action = action.cpu()  # .detach().cpu().numpy()
        return action[0]

    def get_action_std(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        std = std.cpu()  # .detach().cpu().numpy()
        return std[0]


def update(replay_buffer, soft_q_net1, soft_q_net2, value_net, policy_net, target_value_net,
           soft_q_criterion1, soft_q_criterion2, soft_q_optimizer1, soft_q_optimizer2,
           value_criterion, value_optimizer, policy_optimizer,
           batch_size, gamma=0.99, soft_tau=1e-2, prior_eps=1e-6):
    # state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    experience = replay_buffer.sample(batch_size, beta=1.0)
    (state, action, reward, next_state, done, roles, weights, batch_idxes) = experience

    epsilon_d = np.array([1.0 if role =='teacher' else 0.0 for role in roles])

    state = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action = torch.FloatTensor(action).to(device)
    reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)
    weights = torch.FloatTensor(weights).unsqueeze(1).to(device)

    predicted_q_value1 = soft_q_net1(state, action)
    predicted_q_value2 = soft_q_net2(state, action)
    predicted_value = value_net(state)
    new_action, log_prob, epsilon, mean, log_std = policy_net.evaluate(state)

    # Training Q Function
    target_value = target_value_net(next_state)
    target_q_value = reward + (1 - done) * gamma * target_value
    q_value_loss1 = soft_q_criterion1(predicted_q_value1 * weights, target_q_value.detach() * weights)
    q_value_loss2 = soft_q_criterion2(predicted_q_value2 * weights, target_q_value.detach() * weights)

    soft_q_optimizer1.zero_grad()
    q_value_loss1.backward()
    soft_q_optimizer1.step()
    soft_q_optimizer2.zero_grad()
    q_value_loss2.backward()
    soft_q_optimizer2.step()

    # Training Value Function
    predicted_new_q_value = torch.min(soft_q_net1(state, new_action), soft_q_net2(state, new_action))
    target_value_func = predicted_new_q_value - log_prob
    value_loss = value_criterion(predicted_value * weights, target_value_func.detach() * weights)

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    # Training Policy Function
    policy_loss = ((log_prob - predicted_new_q_value) * weights).mean()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )

    # update priority values of sampled transitions
    v_next_state = target_value_net(next_state).detach()
    q_state_action = torch.min(soft_q_net1(state, action), soft_q_net2(state, action)).detach()
    td_error = reward + gamma * (1.0 - done) * v_next_state - q_state_action
    new_priorities = abs(td_error.cpu().numpy().reshape(-1,)) + prior_eps + epsilon_d
    replay_buffer.update_priorities(batch_idxes, new_priorities, roles)


def update_vanilla(replay_buffer, soft_q_net1, soft_q_net2, value_net, policy_net, target_value_net,
           soft_q_criterion1, soft_q_criterion2, soft_q_optimizer1, soft_q_optimizer2,
           value_criterion, value_optimizer, policy_optimizer,
           batch_size, gamma=0.99, soft_tau=1e-2):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action = torch.FloatTensor(action).to(device)
    reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    predicted_q_value1 = soft_q_net1(state, action)
    predicted_q_value2 = soft_q_net2(state, action)
    predicted_value = value_net(state)
    new_action, log_prob, epsilon, mean, log_std = policy_net.evaluate(state)

    # Training Q Function
    target_value = target_value_net(next_state)
    target_q_value = reward + (1 - done) * gamma * target_value
    q_value_loss1 = soft_q_criterion1(predicted_q_value1, target_q_value.detach())
    q_value_loss2 = soft_q_criterion2(predicted_q_value2, target_q_value.detach())

    soft_q_optimizer1.zero_grad()
    q_value_loss1.backward(retain_graph=True)
    soft_q_optimizer1.step()
    soft_q_optimizer2.zero_grad()
    q_value_loss2.backward(retain_graph=True)
    soft_q_optimizer2.step()

    # Training Value Function
    predicted_new_q_value = torch.min(soft_q_net1(state, new_action), soft_q_net2(state, new_action))
    target_value_func = predicted_new_q_value - log_prob
    value_loss = value_criterion(predicted_value, target_value_func.detach())

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    # Training Policy Function
    policy_loss = ((log_prob - predicted_new_q_value)).mean()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )


def evaluate_policy(test_env, policy_net, evaluation_res, env_step, test_episode_num=10, total_seed_num=10, random_goal_task=False, random_y=False, success_evaluation_res=None):
    test_episode_reward = 0.0
    total_success_num = 0.0
    # total_seed_num = 10
    res = []
    success_res = []
    res.append(env_step)
    success_res.append(env_step)
    for seed in range(total_seed_num):
        # np.random.seed(seed)
        sampled_xs = np.random.default_rng(seed).uniform(low=0.01, high=19.9, size=test_episode_num)
        if random_goal_task:
            sampled_goal_xs = np.random.default_rng(seed).uniform(low=0.01, high=19.9, size=test_episode_num)
            if random_y:
                sampled_ys = np.random.default_rng(seed).uniform(low=1.0, high=8.0, size=test_episode_num)
                sampled_goal_ys = np.random.default_rng(seed).uniform(low=12.0, high=19.0, size=test_episode_num)
        for i in range(test_episode_num):
            if random_goal_task:
                if random_y:
                    state_ = test_env.reset(random=False, initial_x=sampled_xs[i], initial_y=sampled_ys[i], random_goal=False,
                                            goal_x=sampled_goal_xs[i], goal_y=sampled_goal_ys[i], random_y=False)
                else:
                    state_ = test_env.reset(random=False, initial_x=sampled_xs[i], initial_y=4.0, random_goal=False, goal_x=sampled_goal_xs[i], goal_y=16.0, random_y=False)
            else:
                state_ = test_env.reset(random=False, initial_x=sampled_xs[i], initial_y=4.0)
            done_ = False
            episode_reward = 0.0
            whether_success = 0.0
            while not done_:
                action_ = policy_net.get_action(state_).detach()
                next_state_, reward_, done_, _ = test_env.step(action_.numpy())
                episode_reward += reward_
                test_episode_reward += reward_
                state_ = next_state_

                if reward_ == 1000.0:
                    whether_success = 1.0
                    total_success_num += 1.0

            res.append(episode_reward)
            success_res.append(whether_success)

    evaluation_res.append(res)
    average_episode_reward = test_episode_reward / (test_episode_num * total_seed_num)

    success_evaluation_res.append(success_res)
    average_success_rate = total_success_num / (test_episode_num * total_seed_num)

    return average_episode_reward, evaluation_res, average_success_rate, success_evaluation_res


def estimate_traj_uncertainty(states_traj, policy_net):
    cumulated_uncertainty = 0.0
    total_states_num = len(states_traj)
    for state in states_traj:
        std_vector = policy_net.get_action_std(state).detach()
        state_uncertainty = np.linalg.norm(std_vector)
        cumulated_uncertainty += state_uncertainty

    traj_uncertainty = cumulated_uncertainty / total_states_num

    return traj_uncertainty


def estimate_traj_uncertainty_td(traj, q_net_1, q_net_2, target_v_net):
    cumulated_td_error = 0.0
    episode_length = len(traj)
    # td_error_list = []
    for state, action, reward, next_state, done in traj:
        abs_td_error = calculate_TD_error(state, action, reward, next_state, done, q_net_1, q_net_2, target_v_net)
        cumulated_td_error += abs_td_error
        # td_error_list.append(abs_td_error)

    traj_uncertainty = cumulated_td_error / episode_length
    # traj_uncertainty = np.std(np.array(td_error_list))

    return traj_uncertainty


def estimate_traj_uncertainty_reward(traj):
    cumulated_rewards = 0.0
    episode_length = len(traj)
    for _, _, reward, _, _ in traj:
        cumulated_rewards += reward

    traj_uncertainty = cumulated_rewards / episode_length

    return traj_uncertainty


def calculate_TD_error(state, action, reward, next_state, done, q_net_1, q_net_2, target_v_net, gamma=0.99):
    state = torch.FloatTensor(state.reshape(-1, state.shape[0])).to(device)
    next_state = torch.FloatTensor(next_state.reshape(-1, next_state.shape[0])).to(device)
    action = torch.FloatTensor(action.reshape(-1, action.shape[0])).to(device)

    v_next_state = target_v_net(next_state).detach().item()
    q_state_action = torch.min(q_net_1(state, action), q_net_2(state, action)).detach().item()
    td_error = reward + gamma * (1.0 - done) * v_next_state - q_state_action

    # print("td error: {}".format(td_error))

    return abs(td_error)


def query_strategy(uncertainty_list, current_uncertainty, max_history_len=20, ratio=0.2):
    if len(uncertainty_list) < max_history_len:
        return False
    else:
        # turn it into descending order without replace original list
        ordered_list = sorted(uncertainty_list, reverse=True)
        thres_idx = int(max_history_len * ratio) - 1
        if thres_idx < 0:
            thres_idx = 0
        thres_uncertainty = ordered_list[thres_idx]

        if current_uncertainty >= thres_uncertainty:
            return True
        else:
            return False


def sample_initial_state(visited_init_states):
    visited_init_xs = np.array([init_state[2] for init_state in visited_init_states])
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(visited_init_xs.reshape((-1, 1)))
    candidate_xs = np.random.uniform(low=0.1, high=19.9, size=100)
    inverse_probs = 1.0 / np.exp(kde.score_samples(candidate_xs.reshape(-1, 1)))
    sampled_x = random.choices(candidate_xs, weights=inverse_probs, k=1)[0]
    # print(sampled_x)

    state = np.array([10.0, 16.0, sampled_x, 4.0])

    return state


def generate_query(env, policy_net, q_net_1, q_net_2, target_v_net, sample_size=100, uncertainty_method='td_error',
                   add_to_buffer=False, use_prioritized_replay_buffer=False, replay_buffer=None):
    init_states_list = []
    uncertainty_list = []
    for i in range(sample_size):
        state = env.reset()
        init_states_list.append(state)
        done = False
        traj = []

        while not done:
            action = policy_net.get_action(state).detach()
            next_state, reward, done, _ = env.step(action.numpy())
            traj.append((state, action, reward, next_state, float(done)))
            state = next_state

            if add_to_buffer:
                if use_prioritized_replay_buffer:
                    replay_buffer.add(state, action, reward, next_state, float(done), role='learner')
                else:
                    replay_buffer.push(state, action, reward, next_state, done, role='learner')

        if uncertainty_method == 'td_error':
            uncertainty = estimate_traj_uncertainty_td(traj=traj,
                                                       q_net_1=q_net_1,
                                                       q_net_2=q_net_2,
                                                       target_v_net=target_v_net)
        else:
            uncertainty = estimate_traj_uncertainty_reward(traj=traj)

        uncertainty_list.append(uncertainty)

    max_uncertain_idx = np.array(uncertainty_list).argmax()
    init_state_to_query = init_states_list[max_uncertain_idx]

    return init_state_to_query


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', help='name of task', default='nav_1')
    parser.add_argument('--sub_id', help='id of human subject', default=1, type=int)
    parser.add_argument('--method', help='name of the method')
    parser.add_argument('--load_joystick_demo', help='whether to load demo for afterwards retraining', default=0, type=int)

    return parser.parse_args()


def evaluate_oracle_performance():
    print('start')
    test_env = NavEnv(render=False)
    oracle = RRT_Oracle()
    tasks = ['nav_1', 'nav_2', 'nav_3']

    total_seed_num = 10
    test_episode_num = 10

    for task in tasks:
        if task == 'nav_1':
            random_goal_task = False
            random_y = False
        elif task == 'nav_2':
            random_goal_task = True
            random_y = False
        else:
            random_goal_task = True
            random_y = True

        test_episode_reward = 0.0
        total_success_num = 0.0

        for seed in range(total_seed_num):
            # np.random.seed(seed)
            sampled_xs = np.random.default_rng(seed).uniform(low=0.01, high=19.9, size=test_episode_num)
            if random_goal_task:
                sampled_goal_xs = np.random.default_rng(seed).uniform(low=0.01, high=19.9, size=test_episode_num)
                if random_y:
                    sampled_ys = np.random.default_rng(seed).uniform(low=1.0, high=8.0, size=test_episode_num)
                    sampled_goal_ys = np.random.default_rng(seed).uniform(low=12.0, high=19.0, size=test_episode_num)
            for i in range(test_episode_num):
                if random_goal_task:
                    if random_y:
                        state = test_env.reset(random=False, initial_x=sampled_xs[i], initial_y=sampled_ys[i],
                                                random_goal=False,
                                                goal_x=sampled_goal_xs[i], goal_y=sampled_goal_ys[i], random_y=False)
                    else:
                        state = test_env.reset(random=False, initial_x=sampled_xs[i], initial_y=4.0, random_goal=False,
                                                goal_x=sampled_goal_xs[i], goal_y=16.0, random_y=False)
                else:
                    state = test_env.reset(random=False, initial_x=sampled_xs[i], initial_y=4.0)

                pos_goal = (state[0], state[1])
                pos_init = (state[2], state[3])
                path = oracle.path_planning(pos_init=pos_init,
                                                pos_goal=pos_goal)

                # 2d np array of (total_step_num, action/state_dimension)
                oracle_states, oracle_actions = oracle.recover_demo(path=path,
                                                                    delta_t=1.0,
                                                                    pos_goal=pos_goal)
                # print('[Task {}, seed {}, test episdoe {}]: finished getting the oracle path')

                done = False
                step = 0
                while not done:
                    action = oracle_actions[step]
                    next_state, reward, done, _ = test_env.step(action)
                    test_episode_reward += reward
                    state = next_state
                    step += 1

                    if reward == 1000.0:
                        total_success_num += 1.0

        average_episode_reward = test_episode_reward / (test_episode_num * total_seed_num)
        print('[{}]: Average episode rewards: {}'.format(task, average_episode_reward))

        average_success_rate = total_success_num / (test_episode_num * total_seed_num)
        print('[{}]: Average success rate: {}'.format(task, average_success_rate))


''' Run the training '''
def main():
    args = argparser()

    seed = 6
    # seed = 12
    random.seed(seed)
    np.random.seed(seed)

    wall_thickness = 0.0
    arrival_thres = 1.0

    ''' prepare the environment '''
    whether_render = False
    nav_env = NavEnv(render=whether_render, wall_thickness=wall_thickness, arrival_thres=arrival_thres)
    env = NormalizedActions(nav_env)
    test_env = NavEnv(render=False, wall_thickness=wall_thickness, arrival_thres=arrival_thres)
    test_env = NormalizedActions(test_env)

    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    hidden_dim = 256

    ''' construct neural networks '''
    torch.manual_seed(seed)
    value_net = ValueNetwork(state_dim, hidden_dim, seed=seed).to(device)
    target_value_net = ValueNetwork(state_dim, hidden_dim, seed=seed).to(device)

    soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim, seed=seed).to(device)
    soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim, seed=seed).to(device)
    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, seed=seed).to(device)

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(param.data)

    value_criterion = nn.MSELoss()
    soft_q_criterion1 = nn.MSELoss()
    soft_q_criterion2 = nn.MSELoss()

    value_lr = 3e-4
    soft_q_lr = 3e-4
    policy_lr = 3e-4

    value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)
    soft_q_optimizer1 = optim.Adam(soft_q_net1.parameters(), lr=soft_q_lr)
    soft_q_optimizer2 = optim.Adam(soft_q_net2.parameters(), lr=soft_q_lr)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)

    ''' Training hyperparameters'''
    # task_name = 'nav_1'
    task_name = args.task_name

    if task_name == 'nav_1':
        random_goal = False
        random_y = False
        max_env_steps = int(10 * 1e4)
        ratio = 0.35
    elif task_name == 'nav_2':
        random_goal = True
        random_y = False
        max_env_steps = int(10 * 1e4)
        ratio = 0.3
    else:
        random_goal = True
        random_y = True
        max_env_steps = int(15 * 1e4)
        # ratio = 0.55
        # ratio = 0.5
        # ratio = 0.4
        # ratio = 0.45
        # ratio = 0.35
        ratio = 0.55

    max_demo_num = 60
    env_step = 0
    batch_size = 128
    teacher_demo_num = 0
    query_pool_size = 20
    max_ratio = 0.6
    min_ratio = 0.0

    ''' Choose the method to use '''
    method = args.method
    # method = 'active_sac'
    # method = 'active_sac_human'
    # method = 'active_sac_bern_timing' # same "what to query" strategy as active_sac, but different "when to query"
    # method = 'active_sac_stream_query' # same "when to query" strategy as active_sac, but different "what to query"
    # method = 'sac_lfd'
    # method = 'sac'
    # method = 'sac_lfd_human'

    ''' Choose how to measure uncertainty '''
    uncertainty_method = 'td_error'
    # uncertainty_method = 'reward'
    # uncertainty_method = 'std'

    ''' Choose how to generate candidate queries '''
    uniform_init_sampling = True
    max_uncertain_rollout = False

    ''' Choose how to select the query to make '''
    stream_based_query = True

    # start joystick node if requiring actively collecting human episodic demonstrations
    if method == 'active_sac_human':
        if args.load_joystick_demo == 0:
            env_info = {'random_goal': random_goal, 'random_y': random_y}
            joystick = Joystick(env_info=env_info)
            demo_state_start_ids = []
            demo_action_start_ids = []
            demo_state_trajs = []
            demo_action_trajs = []
            joystick.env.render_env.draw_text('Please wait for instruction ...')
        else:
            joystick_demo_traj_data_path = 'joystick_trajectory_data/' + task_name + '/' + method + '/' + 'sub_' + str(
                args.sub_id) + '/max_demo_' + str(max_demo_num) + '/ratio_' + str(
                ratio) + '/' + uncertainty_method + '/'

            demo_state_start_ids = np.genfromtxt(joystick_demo_traj_data_path + 'demo_state_start_ids.csv', delimiter=' ')
            demo_action_start_ids = np.genfromtxt(joystick_demo_traj_data_path + 'demo_action_start_ids.csv', delimiter=' ')
            demo_state_trajs = np.genfromtxt(joystick_demo_traj_data_path + 'joystick_demo_state_trajs.csv', delimiter=' ')
            demo_action_trajs = np.genfromtxt(joystick_demo_traj_data_path + 'joystick_demo_action_trajs.csv', delimiter=' ')

    plot_interval = 1000
    rrt_oracle = RRT_Oracle(wall_thickness=wall_thickness)
    evaluate_res_per_step = []
    success_evaluate_res_per_step = []
    evaluate_res_per_demo = []
    success_evaluate_res_per_demo = []
    init_state_query_list = []
    init_state_history = []
    uncertainty_list = []
    traj_list = []
    states_traj_list = []
    demo_traj_data = [] # a list of 2d trajectories, each trajectory include a bunch of (x,y) points
    test_episode_num = 100

    ''' Set the replay buffer '''
    replay_buffer_size = 1000000
    use_prioritized_replay_buffer = False
    if not use_prioritized_replay_buffer:
        replay_buffer = ReplayBuffer(replay_buffer_size)
    else:
        prioritized_replay_alpha = 0.3
        prioritized_replay_buffer = PrioritizedReplayBuffer(size=replay_buffer_size,
                                                            alpha=prioritized_replay_alpha)

    if method == 'sac_lfd_human' or method == 'active_sac_human':
        writer = SummaryWriter('logs/new/' + task_name + '/' + method + '/' + 'sub_' + str(args.sub_id) + '/max_demo_' + str(max_demo_num) + '/ratio_' + str(
            ratio) + '/' + uncertainty_method + '/' + TIMESTAMP)
    else:
        writer = SummaryWriter('logs/new/' + task_name + '/' + method + '/max_demo_' + str(max_demo_num) + '/ratio_' + str(ratio) + '/' + uncertainty_method + '/' + TIMESTAMP)

    ''' start training '''
    if method == 'sac_lfd':
        print("[SAC LfD]: Start to first collect teacher demo before learning ... ")
        for demo_num in range(max_demo_num):
            state = env.reset(random=True, random_goal=random_goal, random_y=random_y)
            pos_goal = (state[0], state[1])
            pos_init = (state[2], state[3])
            path = rrt_oracle.path_planning(pos_init=pos_init,
                                            pos_goal=pos_goal)

            # 2d np array of (total_step_num, action/state_dimension)
            oracle_states, oracle_actions = rrt_oracle.recover_demo(path=path,
                                                                    delta_t=1.0,
                                                                    pos_goal=pos_goal)

            demo_traj = np.array([[np.inf, np.inf]])
            demo_traj = np.concatenate((demo_traj, oracle_states[:, 2:]))  # 2d np array
            demo_traj = demo_traj.tolist()  # 2d list of (x, y)
            demo_traj_data.extend(demo_traj)  # 2d list of (x, y)

            state = env.reset(random=False, initial_x=pos_init[0], initial_y=pos_init[1], random_goal=False, goal_x=pos_goal[0], goal_y=pos_goal[1], random_y=False)
            done = False
            step = 0
            while not done:
                action = oracle_actions[step]
                next_state, reward, done, _ = env.step(action)
                if use_prioritized_replay_buffer:
                    prioritized_replay_buffer.add(state, action, reward, next_state, float(done), role='teacher')
                else:
                    replay_buffer.push(state, action, reward, next_state, done, role='teacher')

                state = next_state
                step += 1

            print("[SAC LfD]: demo {} is collected".format(demo_num + 1))

        print("[SAC LfD]: All demo are collected. Going to train the model")
        print("***************************")
    elif method =='sac_lfd_human':
        print("[SAC LfD Human]: Start to first load teacher demo before learning ... ")
        subject_id = args.sub_id
        path = 'human_data/' + task_name + '_' + 'sub_' + str(subject_id) + '_' + 'init_and_goal.csv'
        human_data = np.genfromtxt(path, delimiter=' ') # 2d np array as (demo_num, 4), each row is (init_x, init_y, goal_x, goal_y)

        for demo_num in range(max_demo_num):
            pos_init = (human_data[demo_num][0], human_data[demo_num][1])
            pos_goal = (human_data[demo_num][2], human_data[demo_num][3])
            path = rrt_oracle.path_planning(pos_init=pos_init,
                                            pos_goal=pos_goal)
            # 2d np array of (total_step_num, action/state_dimension)
            oracle_states, oracle_actions = rrt_oracle.recover_demo(path=path,
                                                                    delta_t=1.0,
                                                                    pos_goal=pos_goal)

            demo_traj = np.array([[np.inf, np.inf]])
            demo_traj = np.concatenate((demo_traj, oracle_states[:, 2:]))  # 2d np array
            demo_traj = demo_traj.tolist()  # 2d list of (x, y)
            demo_traj_data.extend(demo_traj)  # 2d list of (x, y)

            state = env.reset(random=False, initial_x=pos_init[0], initial_y=pos_init[1], random_goal=False,
                              goal_x=pos_goal[0], goal_y=pos_goal[1], random_y=False)
            done = False
            step = 0
            while not done:
                action = oracle_actions[step]
                next_state, reward, done, _ = env.step(action)
                if use_prioritized_replay_buffer:
                    prioritized_replay_buffer.add(state, action, reward, next_state, float(done), role='teacher')
                else:
                    replay_buffer.push(state, action, reward, next_state, done, role='teacher')

                state = next_state
                step += 1

            print("[SAC LfD Human]: demo {} is collected, reward is {}".format(demo_num + 1, reward))

        print("[SAC LfD Human]: All demo are collected. Going to train the model")
        print("***************************")

    while env_step <= max_env_steps:
        if len(init_state_history) == 0:
            state = env.reset(random=True, random_goal=random_goal, random_y=random_y)
        else:
            if uniform_init_sampling:
                state = env.reset(random=True, random_goal=random_goal, random_y=random_y)
            elif max_uncertain_rollout:
                init_state = generate_query(env=env, policy_net=policy_net, q_net_1=soft_q_net1, q_net_2=soft_q_net2, target_v_net=target_value_net, sample_size=100, uncertainty_method=uncertainty_method,
                                            add_to_buffer=False, use_prioritized_replay_buffer=use_prioritized_replay_buffer, replay_buffer=replay_buffer)
                state = env.reset(random=False, initial_x=init_state[2], initial_y=init_state[3])
            else:
                # init_state = sample_initial_state(init_state_history)
                if len(init_state_query_list) < query_pool_size:
                    state = env.reset()
                else:
                    init_state = sample_initial_state(init_state_query_list)
                    state = env.reset(random=False, initial_x=init_state[2], initial_y=init_state[3])

        init_state = state.copy()
        done = False
        states_traj = []
        traj = []
        step_count = 0
        while not done:
            if whether_render:
                env.render()
                sleep(0.001)

            # if env_step > 1000:
            action = policy_net.get_action(state).detach()
            next_state, reward, done, _ = env.step(action.numpy())
            # else:
            #     action = env.action_space.sample()
            #     next_state, reward, done, _ = env.step(action)

            states_traj.append(state)
            traj.append((state, action, reward, next_state, float(done)))

            if use_prioritized_replay_buffer:
                prioritized_replay_buffer.add(state, action, reward, next_state, float(done), role='learner')
                if len(prioritized_replay_buffer) > batch_size:
                    update(prioritized_replay_buffer, soft_q_net1, soft_q_net2, value_net, policy_net, target_value_net,
                           soft_q_criterion1, soft_q_criterion2, soft_q_optimizer1, soft_q_optimizer2,
                           value_criterion, value_optimizer, policy_optimizer, batch_size)
            else:
                replay_buffer.push(state, action, reward, next_state, done, role='learner')
                if len(replay_buffer) > batch_size:
                    update_vanilla(replay_buffer, soft_q_net1, soft_q_net2, value_net, policy_net, target_value_net,
                           soft_q_criterion1, soft_q_criterion2, soft_q_optimizer1, soft_q_optimizer2,
                           value_criterion, value_optimizer, policy_optimizer, batch_size)

            # evaluate policy when not using active sac while collecting human joystick demos
            # if not (method == "active_sac_human" and args.load_joystick_demo == 0):

            if env_step % plot_interval == 0:
                if not (method == "active_sac_human"):
                    average_episode_reward, evaluate_res_per_step, average_success_rate, success_evaluate_res_per_step = evaluate_policy(test_env=test_env, policy_net=policy_net,
                                                                           evaluation_res=evaluate_res_per_step,
                                                                           env_step=env_step, test_episode_num=test_episode_num, total_seed_num=10, random_goal_task=random_goal, random_y=random_y, success_evaluation_res=success_evaluate_res_per_step)
                    writer.add_scalar(tag='episode_reward/train/per_env_step', scalar_value=average_episode_reward,
                                      global_step=env_step)
                    writer.add_scalar(tag='success_rate/train/per_env_step', scalar_value=average_success_rate,
                                      global_step=env_step)
                    print("[{} environment steps finished]: Average episode reward is {}".format(env_step,
                                                                                                 average_episode_reward))
                    model_path = 'models/' + task_name + '/' + method + '/max_demo_' + str(
                        max_demo_num) + '/ratio_' + str(ratio) + '/' + uncertainty_method + '/per_step/'
                else:
                    print("[{} environment steps finished]".format(env_step))
                    model_path = 'models/' + task_name + '/' + method + '/' + 'sub_' + str(args.sub_id) + '/max_demo_' + str(
                        max_demo_num) + '/ratio_' + str(ratio) + '/' + uncertainty_method + '/per_step/'

                # save checkpoint model
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                torch.save(policy_net.state_dict(), model_path + str(env_step) + '.pth')
                # print("Model saved")

            state = next_state
            env_step += 1
            step_count += 1

        ''' Estimate uncertainty for latest learner roll-out '''
        if method == 'active_sac' or method == 'active_sac_bern_timing' or method == 'active_sac_stream_query' or method == 'active_sac_human':
            if uncertainty_method == 'td_error':
                uncertainty = estimate_traj_uncertainty_td(traj=traj, q_net_1=soft_q_net1, q_net_2=soft_q_net2, target_v_net=target_value_net)
                traj_list.append(traj)
            elif uncertainty_method == 'reward':
                uncertainty = estimate_traj_uncertainty_reward(traj=traj)
                traj_list.append(traj)
            else:
                uncertainty = estimate_traj_uncertainty(states_traj, policy_net)
                states_traj_list.append(states_traj)

            if method == 'active_sac' or method == 'active_sac_stream_query' or method == 'active_sac_human':
                # if len(replay_buffer) <= batch_size:
                if (use_prioritized_replay_buffer and len(prioritized_replay_buffer) <= batch_size) or \
                        (not use_prioritized_replay_buffer and len(replay_buffer) <= batch_size):
                    whether_query = False
                else:
                    # ratio_t = max_ratio - teacher_demo_num * (max_ratio - min_ratio)/max_demo_num
                    whether_query = query_strategy(uncertainty_list, uncertainty, max_history_len=query_pool_size, ratio=ratio)
            else:
                seed = np.random.uniform(0, 1)
                whether_query = (seed <= ratio)

            uncertainty_list.append(uncertainty)
            init_state_query_list.append(init_state)
            init_state_history.append(init_state)
        else:
            whether_query = False

        ''' Query teacher for demo '''
        if teacher_demo_num < max_demo_num and whether_query:
            if method == 'active_sac' or method == 'active_sac_bern_timing' or method == 'active_sac_human':
                if stream_based_query:
                    max_uncertain_idx = np.array(uncertainty_list).argmax()
                    # max_uncertain_idx = -1
                    state = init_state_query_list[max_uncertain_idx]

                    """uncertain_idx = np.random.choice(len(uncertainty_list), p=np.array(uncertainty_list)/sum(uncertainty_list))
                    state = init_state_query_list[uncertain_idx]

                    if task_name == 'nav_1':
                        state[2] += np.random.normal(loc=0.0, scale=1.0)
                    elif task_name == 'nav_2':
                        state[0] += np.random.normal(loc=0.0, scale=1.0)
                        state[2] += np.random.normal(loc=0.0, scale=1.0)
                    else:
                        state[0] += np.random.normal(loc=0.0, scale=1.0)
                        state[1] += np.random.normal(loc=0.0, scale=1.0)
                        state[2] += np.random.normal(loc=0.0, scale=1.0)
                        state[3] += np.random.normal(loc=0.0, scale=1.0)

                    state = np.clip(state, a_min=[0.01, 12.0, 0.01, 1.0], a_max=[19.9, 19.0, 19.9, 8.0])"""

                    # state[2] += np.random.normal(loc=0.0, scale=1.0)
                    # if state[2] <= 0.0:
                    #     state[2] = 0.001
                    # elif state[2] >= 20.0:
                    #     state[2] = 19.999
                else:
                    state = generate_query(env=env, policy_net=policy_net, q_net_1=soft_q_net1, q_net_2=soft_q_net2,
                                           target_v_net=target_value_net, sample_size=100, uncertainty_method=uncertainty_method,
                                           add_to_buffer=False, use_prioritized_replay_buffer=use_prioritized_replay_buffer, replay_buffer=replay_buffer)
            else:
                state = env.reset(random=True, random_goal=random_goal, random_y=random_y)

            pos_goal = (state[0], state[1])
            pos_init = (state[2], state[3])

            if method == 'active_sac_human':
                if args.load_joystick_demo == 0:
                    joystick.env.render_env.draw_text('[Demo {}: Please provide a demo]'.format(teacher_demo_num + 1))
                    oracle_states, oracle_actions = joystick.provide_joystick_demo(starting_pos=pos_init, goal_pos=pos_goal) # oracle_states will be one step more than oracle_actions (terminal state)

                    demo_state_start_id = len(demo_state_trajs)
                    demo_state_start_ids.append(demo_state_start_id)
                    demo_action_start_id = len(demo_action_trajs)
                    demo_action_start_ids.append(demo_action_start_id)

                    demo_state_traj = np.array([[np.inf, np.inf, np.inf, np.inf]])
                    demo_state_traj = np.concatenate((demo_state_traj, oracle_states))  # 2d np array
                    demo_state_traj = demo_state_traj.tolist()  # 2d list of (goal_x, goal_y, x, y)
                    demo_state_trajs.extend(demo_state_traj)  # 2d list of (goal_x, goal_y, x, y)

                    demo_action_traj = np.array([[np.inf, np.inf]])
                    demo_action_traj = np.concatenate((demo_action_traj, oracle_actions))  # 2d np array
                    demo_action_traj = demo_action_traj.tolist()  # 2d list of (vx, vy)
                    demo_action_trajs.extend(demo_action_traj)  # 2d list of (vx, vy)

                    joystick.env.render_env.clean()
                    if teacher_demo_num < (max_demo_num - 1):
                        joystick.env.render_env.draw_text('Please wait for instruction ...')
                    else:
                        joystick.env.render_env.draw_text('Great! All demos are provided!')
                else:
                    state_start_id = int(demo_state_start_ids[teacher_demo_num])
                    action_start_id = int(demo_action_start_ids[teacher_demo_num])

                    if teacher_demo_num < max_demo_num - 1:
                        state_start_id_next = int(demo_state_start_ids[teacher_demo_num + 1])
                        action_start_id_next = int(demo_action_start_ids[teacher_demo_num + 1])
                        oracle_states = demo_state_trajs[(state_start_id + 1):state_start_id_next, :]
                        oracle_actions = demo_action_trajs[(action_start_id + 1):action_start_id_next, :]
                    else:
                        oracle_states = demo_state_trajs[(state_start_id + 1):, :]
                        oracle_actions = demo_action_trajs[(action_start_id + 1):, :]

                    print('[Active Sac Human]: Queried initial state: {}, loaded demo initial state: {}'.format(state, oracle_states[0, :]))
            else:
                path = rrt_oracle.path_planning(pos_init=pos_init,
                                                pos_goal=pos_goal)
                # 2d np array of (total_step_num, action/state_dimension)
                oracle_states, oracle_actions = rrt_oracle.recover_demo(path=path,
                                                                        delta_t=1.0,
                                                                        pos_goal=pos_goal)

            demo_traj = np.array([[np.inf, np.inf]])
            demo_traj = np.concatenate((demo_traj, oracle_states[:, 2:])) # 2d np array
            demo_traj = demo_traj.tolist() # 2d list of (x, y)
            demo_traj_data.extend(demo_traj) # 2d list of (x, y)

            state = env.reset(random=False, initial_x=pos_init[0], initial_y=pos_init[1], random_goal=False, goal_x=pos_goal[0], goal_y=pos_goal[1], random_y=False)
            done = False
            step = 0
            while not done:
                if whether_render:
                    env.render()
                    sleep(0.001)
                action = oracle_actions[step]
                next_state, reward, done, _ = env.step(action)

                if use_prioritized_replay_buffer:
                    prioritized_replay_buffer.add(state, action, reward, next_state, float(done), role='teacher')
                    if len(prioritized_replay_buffer) > batch_size:
                        update(prioritized_replay_buffer, soft_q_net1, soft_q_net2, value_net, policy_net,
                               target_value_net,
                               soft_q_criterion1, soft_q_criterion2, soft_q_optimizer1, soft_q_optimizer2,
                               value_criterion, value_optimizer, policy_optimizer, batch_size)
                else:
                    replay_buffer.push(state, action, reward, next_state, done, role='teacher')
                    if len(replay_buffer) > batch_size:
                        update_vanilla(replay_buffer, soft_q_net1, soft_q_net2, value_net, policy_net, target_value_net,
                               soft_q_criterion1, soft_q_criterion2, soft_q_optimizer1, soft_q_optimizer2,
                               value_criterion, value_optimizer, policy_optimizer, batch_size)

                # evaluate policy when not using active sac while collecting human joystick demos
                # if not (method == "active_sac_human" and args.load_joystick_demo == 0):

                if env_step % plot_interval == 0:
                    if not (method == "active_sac_human"):
                        average_episode_reward, evaluate_res_per_step, average_success_rate, success_evaluate_res_per_step = evaluate_policy(test_env=test_env, policy_net=policy_net,
                                                                               evaluation_res=evaluate_res_per_step, env_step=env_step,
                                                                               test_episode_num=test_episode_num, total_seed_num=10, random_goal_task=random_goal, random_y=random_y, success_evaluation_res=success_evaluate_res_per_step)
                        writer.add_scalar(tag='episode_reward/train/per_env_step', scalar_value=average_episode_reward,
                                          global_step=env_step)
                        writer.add_scalar(tag='success_rate/train/per_env_step', scalar_value=average_success_rate,
                                          global_step=env_step)
                        print("[{} environment steps finished]: Average episode reward is {}".format(env_step,
                                                                                                     average_episode_reward))
                        model_path = 'models/' + task_name + '/' + method + '/max_demo_' + str(
                            max_demo_num) + '/ratio_' + str(ratio) + '/' + uncertainty_method + '/per_step/'
                    else:
                        print("[{} environment steps finished]".format(env_step))
                        model_path = 'models/' + task_name + '/' + method + '/' + 'sub_' + str(
                            args.sub_id) + '/max_demo_' + str(max_demo_num) + '/ratio_' + str(ratio) + '/' + uncertainty_method + '/per_step/'

                    # save the checkpoint model
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    torch.save(policy_net.state_dict(), model_path + str(env_step) + '.pth')
                    # print("Model saved")

                state = next_state
                env_step += 1
                step += 1

            teacher_demo_num += 1
            # evaluate policy when not using active sac while collecting human joystick demos
            # if not (method == "active_sac_human" and args.load_joystick_demo == 0):
            if not (method == "active_sac_human"):
                average_episode_reward, evaluate_res_per_demo, average_success_rate, success_evaluate_res_per_demo = evaluate_policy(test_env=test_env, policy_net=policy_net,
                                                                                evaluation_res=evaluate_res_per_demo,
                                                                                env_step=teacher_demo_num,
                                                                                test_episode_num=test_episode_num, total_seed_num=10, random_goal_task=random_goal, random_y=random_y, success_evaluation_res=success_evaluate_res_per_demo)
                writer.add_scalar(tag='episode_reward/train/per_demo', scalar_value=average_episode_reward,
                                  global_step=teacher_demo_num)
                writer.add_scalar(tag='success_rate/train/per_demo', scalar_value=average_success_rate,
                                  global_step=teacher_demo_num)
                print("[{} demos provided]: Average episode reward is {}".format(teacher_demo_num, average_episode_reward))
            else:
                print("[{} demos provided]".format(teacher_demo_num))

            writer.add_scalar(tag='total_demo_num/per_env_step', scalar_value=teacher_demo_num,
                              global_step=env_step)

            # if not (method == "active_sac_human" and args.load_joystick_demo == 0):
            # if not (method == "active_sac_human"):
            if teacher_demo_num % 5 == 0:
                if method == 'active_sac_human':
                    model_path = 'models/' + task_name + '/' + method + '/' + 'sub_' + str(
                        args.sub_id) + '/max_demo_' + str(max_demo_num) + '/ratio_' + str(
                        ratio) + '/' + uncertainty_method + '/per_demo/'
                else:
                    model_path = 'models/' + task_name + '/' + method + '/max_demo_' + str(max_demo_num) + '/ratio_' + str(
                        ratio) + '/' + uncertainty_method + '/per_demo/'

                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                torch.save(policy_net.state_dict(), model_path + str(teacher_demo_num) + '.pth')
                # print("Model saved")

            # ''' update uncertainty of queried traj'''
            # if method == 'active_sac':
            #     if uncertainty_method == 'td_error':
            #         uncertainty = estimate_traj_uncertainty_td(traj=traj_list[max_uncertain_idx], q_net_1=soft_q_net1, q_net_2=soft_q_net2,
            #                                                    target_v_net=target_value_net)
            #     else:
            #         uncertainty = estimate_traj_uncertainty(states_traj_list[max_uncertain_idx], policy_net)
            #
            #     uncertainty_list[max_uncertain_idx] = uncertainty

        if len(uncertainty_list) > query_pool_size:
            uncertainty_list.pop(0)
            init_state_query_list.pop(0)

            if uncertainty_method == 'td_error':
                traj_list.pop(0)
            elif uncertainty_method == 'reward':
                traj_list.pop(0)
            else:
                states_traj_list.pop(0)

        if (method == 'active_sac_human') and (args.load_joystick_demo == 0) and (teacher_demo_num == max_demo_num):
            joystick_demo_traj_data_path = 'joystick_trajectory_data/' + task_name + '/' + method + '/' + 'sub_' + str(
                args.sub_id) + '/max_demo_' + str(max_demo_num) + '/ratio_' + str(ratio) + '/' + uncertainty_method + '/'
            if not os.path.exists(joystick_demo_traj_data_path):
                os.makedirs(joystick_demo_traj_data_path)

            np.savetxt(joystick_demo_traj_data_path + 'joystick_demo_state_trajs.csv', demo_state_trajs, delimiter=' ')
            np.savetxt(joystick_demo_traj_data_path + 'demo_state_start_ids.csv', demo_state_start_ids, delimiter=' ')

            np.savetxt(joystick_demo_traj_data_path + 'joystick_demo_action_trajs.csv', demo_action_trajs, delimiter=' ')
            np.savetxt(joystick_demo_traj_data_path + 'demo_action_start_ids.csv', demo_action_start_ids, delimiter=' ')

            joystick.stop()
            print("[Active SAC Human]: All joystick demos are collected and saved. Going to quit...")

            break

    ''' Save the results '''
    if method == 'sac_lfd_human' or method == 'active_sac_human':
        res_per_step_path = 'evaluation_res/new/' + task_name + '/' + method + '/' + 'sub_' + str(args.sub_id) + '/max_demo_' + str(
            max_demo_num) + '/ratio_' + str(
            ratio) + '/' + uncertainty_method + '/'
        if not os.path.exists(res_per_step_path):
            os.makedirs(res_per_step_path)

        res_per_demo_path = 'evaluation_res/new/' + task_name + '/' + method + '/' + 'sub_' + str(args.sub_id) + '/max_demo_' + str(
            max_demo_num) + '/ratio_' + str(
            ratio) + '/' + uncertainty_method + '/'
        if not os.path.exists(res_per_demo_path):
            os.makedirs(res_per_demo_path)

        demo_traj_data_path = 'demo_trajectory_data/' + task_name + '/' + method + '/' + 'sub_' + str(
            args.sub_id) + '/max_demo_' + str(
            max_demo_num) + '/ratio_' + str(
            ratio) + '/' + uncertainty_method + '/'
        if not os.path.exists(demo_traj_data_path):
            os.makedirs(demo_traj_data_path)
    else:
        res_per_step_path = 'evaluation_res/new/' + task_name + '/' + method + '/max_demo_' + str(max_demo_num) + '/ratio_' + str(
                        ratio) + '/' + uncertainty_method + '/'
        if not os.path.exists(res_per_step_path):
            os.makedirs(res_per_step_path)

        res_per_demo_path = 'evaluation_res/new/' + task_name + '/' + method + '/max_demo_' + str(max_demo_num) + '/ratio_' + str(
            ratio) + '/' + uncertainty_method + '/'
        if not os.path.exists(res_per_demo_path):
            os.makedirs(res_per_demo_path)

        demo_traj_data_path = 'demo_trajectory_data/' + task_name + '/' + method + '/max_demo_' + str(max_demo_num) + '/ratio_' + str(
                        ratio) + '/' + uncertainty_method + '/'
        if not os.path.exists(demo_traj_data_path):
            os.makedirs(demo_traj_data_path)

    if not (method == "active_sac_human"):
        np.savetxt(res_per_step_path + 'res_per_step_new.csv', evaluate_res_per_step, delimiter=' ')
        np.savetxt(res_per_demo_path + 'res_per_demo_new.csv', evaluate_res_per_demo, delimiter=' ')
        np.savetxt(res_per_step_path + 'success_res_per_step_new.csv', success_evaluate_res_per_step, delimiter=' ')
        np.savetxt(res_per_demo_path + 'success_res_per_demo_new.csv', success_evaluate_res_per_demo, delimiter=' ')
    else:
        if args.load_joystick_demo == 0:
            pass
        else:
            trained_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, seed=seed).to(device)
            print("*******************************************")
            print("*******************************************")
            print("[Active SAC Human]: Going to evaluate trained models ...")
            training_step_to_load = 0

            while training_step_to_load <= max_env_steps:
                model_path = 'models/' + task_name + '/' + method + '/' + 'sub_' + str(
                            args.sub_id) + '/max_demo_' + str(max_demo_num) + '/ratio_' + str(ratio) + '/' + uncertainty_method + '/per_step/' + str(training_step_to_load) + '.pth'
                trained_policy_net.load_state_dict(torch.load(model_path))

                average_episode_reward, evaluate_res_per_step, average_success_rate, success_evaluate_res_per_step = evaluate_policy(
                    test_env=test_env, policy_net=trained_policy_net,
                    evaluation_res=evaluate_res_per_step, env_step=training_step_to_load,
                    test_episode_num=test_episode_num, total_seed_num=10, random_goal_task=random_goal, random_y=random_y,
                    success_evaluation_res=success_evaluate_res_per_step)
                writer.add_scalar(tag='episode_reward/train/per_env_step', scalar_value=average_episode_reward,
                                  global_step=training_step_to_load)
                writer.add_scalar(tag='success_rate/train/per_env_step', scalar_value=average_success_rate,
                                  global_step=training_step_to_load)
                print("[{} environment steps finished]: Average episode reward is {}".format(training_step_to_load,
                                                                                             average_episode_reward))

                training_step_to_load += plot_interval

            np.savetxt(res_per_step_path + 'res_per_step_new.csv', evaluate_res_per_step, delimiter=' ')
            np.savetxt(res_per_step_path + 'success_res_per_step_new.csv', success_evaluate_res_per_step, delimiter=' ')

            print("[Active SAC Human]: Evaluations finished")


    if not (method == "active_sac_human" and args.load_joystick_demo == 1):
        np.savetxt(demo_traj_data_path + 'demo_trajectory_data_new.csv', demo_traj_data, delimiter=' ')

    writer.close()
    print("************************")
    print("All training finished")


if __name__ == '__main__':
    main()
    # evaluate_oracle_performance()
