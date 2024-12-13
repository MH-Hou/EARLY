import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
import os
from sklearn.neighbors import KernelDensity
from time import sleep
import argparse
import math


import warnings

warnings.filterwarnings("ignore")

from nav_env import NavEnv
from nav_oracle_rrt import RRT_Oracle
from reply_buffer import PrioritizedReplayBuffer
from activesac_nav import argparser, NormalizedActions, ValueNetwork, SoftQNetwork, PolicyNetwork, ReplayBuffer, update, update_vanilla, evaluate_policy
from scripts.joystick import Joystick


TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
print(device)


def estimate_uncertainty(state, policy_net):
    std_vector = policy_net.get_action_std(state).detach() # in the shape of (, action_dim)
    state_uncertainty = np.linalg.norm(std_vector)

    return state_uncertainty


def query_strategy(uncertainty_history, uncertainty, max_history_len, ratio):
    if len(uncertainty_history) < max_history_len:
        return False
    else:
        # turn it into descending order without replace original list
        ordered_list = sorted(uncertainty_history, reverse=True)
        thres_idx = int(max_history_len * ratio) - 1
        if thres_idx < 0:
            thres_idx = 0
        thres_uncertainty = ordered_list[thres_idx]

        if uncertainty >= thres_uncertainty:
            return True
        else:
            return False


def main():
    args = argparser()

    task_name = args.task_name
    if task_name == 'nav_1':
        random_goal = False
        random_y = False
        max_env_steps = int(10 * 1e4)
        ratio = 0.1
        wall_thickness = 0.1
    elif task_name == 'nav_2':
        random_goal = True
        random_y = False
        max_env_steps = int(10 * 1e4)
        ratio = 0.1
        wall_thickness = 0.1
    else:
        random_goal = True
        random_y = True
        max_env_steps = int(15 * 1e4)
        ratio = 0.1
        wall_thickness = 0.1

    seed = 6
    random.seed(seed)
    np.random.seed(seed)

    ''' prepare the environment '''
    whether_render = False
    nav_env = NavEnv(render=whether_render, wall_thickness=wall_thickness)
    env = NormalizedActions(nav_env)
    test_env = NavEnv(render=False, wall_thickness=wall_thickness)

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
    method = args.method
    # method = "isolated_active"
    # method = 'isolated_active_human'
    max_demo_num = 60
    env_step = 0
    batch_size = 128

    # start joystick node if requiring actively collecting human episodic demonstrations
    if method == 'isolated_active_human':
        if args.load_joystick_demo == 0:
            env_info = {'random_goal': random_goal, 'random_y': random_y}
            joystick = Joystick(env_info=env_info, mode='isolated_active_human')
            demo_state_start_ids = []
            demo_action_start_ids = []
            demo_state_trajs = []
            demo_action_trajs = []
            joystick.env.render_env.draw_text('Please wait for instruction ...')
        else:
            joystick_demo_traj_data_path = 'joystick_trajectory_data/' + task_name + '/' + method + '/' + 'sub_' + str(
                args.sub_id) + '/max_demo_' + str(max_demo_num) + '/ratio_' + str(ratio) + '/'

            demo_state_start_ids = np.genfromtxt(joystick_demo_traj_data_path + 'demo_state_start_ids.csv', delimiter=' ')
            demo_action_start_ids = np.genfromtxt(joystick_demo_traj_data_path + 'demo_action_start_ids.csv', delimiter=' ')
            demo_state_trajs = np.genfromtxt(joystick_demo_traj_data_path + 'joystick_demo_state_trajs.csv', delimiter=' ')
            demo_action_trajs = np.genfromtxt(joystick_demo_traj_data_path + 'joystick_demo_action_trajs.csv', delimiter=' ')

            demo_iteration = 0

    plot_interval = 1000
    rrt_oracle = RRT_Oracle(wall_thickness=wall_thickness)
    evaluate_res_per_step = []
    success_evaluate_res_per_step = []
    uncertainty_history = []
    max_history_len = 5000
    max_steps_per_demo = 10
    max_demo_amount = 1500
    demo_traj_data = [] # a list of 2d trajectories, each trajectory include a bunch of (x,y) points

    ''' Set the replay buffer '''
    replay_buffer_size = 1000000
    use_prioritized_replay_buffer = True
    if not use_prioritized_replay_buffer:
        replay_buffer = ReplayBuffer(replay_buffer_size)
    else:
        prioritized_replay_alpha = 0.3
        prioritized_replay_buffer = PrioritizedReplayBuffer(size=replay_buffer_size,
                                                            alpha=prioritized_replay_alpha)

    if method == 'isolated_active_human':
        writer = SummaryWriter(
            'logs/new/' + task_name + '/' + method + '/' + 'sub_' + str(args.sub_id) + '/max_demo_' + str(
                max_demo_num) + '/ratio_' + str(ratio) + '/' + TIMESTAMP)
    else:
        writer = SummaryWriter('logs/new/' + task_name + '/' + method + '/max_demo_' + str(max_demo_num) + '/ratio_' + str(
            ratio) + '/' + TIMESTAMP)

    ''' start training '''
    demo_amount = 0
    while env_step <= max_env_steps:
        done = False
        state = env.reset(random=True, random_goal=random_goal, random_y=random_y)
        while not done:
            uncertainty = estimate_uncertainty(state, policy_net)
            whether_query = query_strategy(uncertainty_history, uncertainty, max_history_len, ratio)

            if whether_query and demo_amount < max_demo_amount:
                pos_goal = (state[0], state[1])
                pos_init = (state[2], state[3])
                print("[Teacher Demo at env step {}]: starting pos: {}, goal pos: {}".format(env_step, pos_init, pos_goal))

                if method == 'isolated_active':
                    path = rrt_oracle.path_planning(pos_init=pos_init,
                                                    pos_goal=pos_goal)
                    # 2d np array of (total_step_num, action/state_dimension)
                    oracle_states, oracle_actions = rrt_oracle.recover_demo(path=path,
                                                                            delta_t=1.0,
                                                                            pos_goal=pos_goal)
                else:
                    if args.load_joystick_demo == 0:
                        sleep(2.0)

                        joystick.env.render_env.draw_text('[{}% demos already provided: Please provide a demo]'.format(100.0 * round(demo_amount / max_demo_amount, 3)))
                        oracle_states, oracle_actions = joystick.provide_joystick_demo(starting_pos=pos_init,
                                                                                       goal_pos=pos_goal,
                                                                                       reset_step=env.current_step)  # oracle_states will be one step more than oracle_actions (terminal state)

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
                        if demo_amount + oracle_actions.shape[0] < max_demo_amount:
                            joystick.env.render_env.draw_text('Please wait for instruction ...')
                        else:
                            joystick.env.render_env.draw_text('Great! All demos are provided!')
                    else:
                        state_start_id = int(demo_state_start_ids[demo_iteration])
                        action_start_id = int(demo_action_start_ids[demo_iteration])

                        # if demo_amount < max_demo_amount:
                        #     state_start_id_next = int(demo_state_start_ids[demo_iteration + 1])
                        #     action_start_id_next = int(demo_action_start_ids[demo_iteration + 1])
                        #     oracle_states = demo_state_trajs[(state_start_id + 1):state_start_id_next, :]
                        #     oracle_actions = demo_action_trajs[(action_start_id + 1):action_start_id_next, :]
                        # else:
                        #     oracle_states = demo_state_trajs[(state_start_id + 1):, :]
                        #     oracle_actions = demo_action_trajs[(action_start_id + 1):, :]

                        if demo_iteration < len(demo_state_start_ids) - 1:
                            state_start_id_next = int(demo_state_start_ids[demo_iteration + 1])
                            action_start_id_next = int(demo_action_start_ids[demo_iteration + 1])
                            oracle_states = demo_state_trajs[(state_start_id + 1):state_start_id_next, :]
                            oracle_actions = demo_action_trajs[(action_start_id + 1):action_start_id_next, :]
                        else:
                            oracle_states = demo_state_trajs[(state_start_id + 1):, :]
                            oracle_actions = demo_action_trajs[(action_start_id + 1):, :]

                        demo_iteration += 1

                        print(
                            '[Isolated Active Human]: Queried initial state: {}, loaded demo initial state: {}'.format(state,
                                                                                                                  oracle_states[
                                                                                                                  0,
                                                                                                                  :]))

                if oracle_actions.shape[0] > max_steps_per_demo:
                    oracle_actions = oracle_actions[:max_steps_per_demo, :]

                oracle_states_to_store = oracle_states[:oracle_actions.shape[0], :]
                demo_traj = np.array([[np.inf, np.inf]])
                demo_traj = np.concatenate((demo_traj, oracle_states_to_store[:, 2:]))  # 2d np array
                demo_traj = demo_traj.tolist()  # 2d list of (x, y)
                demo_traj_data.extend(demo_traj)  # 2d list of (x, y)

                demo_step = 0
                while demo_step < oracle_actions.shape[0] and not done:
                    action = oracle_actions[demo_step]
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
                            update_vanilla(replay_buffer, soft_q_net1, soft_q_net2, value_net, policy_net,
                                           target_value_net,
                                           soft_q_criterion1, soft_q_criterion2, soft_q_optimizer1, soft_q_optimizer2,
                                           value_criterion, value_optimizer, policy_optimizer, batch_size)

                    # evaluate policy
                    if env_step % plot_interval == 0:
                        if not (method == "isolated_active_human"):
                            average_episode_reward, evaluate_res_per_step, average_success_rate, success_evaluate_res_per_step = evaluate_policy(
                                test_env=test_env, policy_net=policy_net,
                                evaluation_res=evaluate_res_per_step,
                                env_step=env_step, test_episode_num=100, total_seed_num=10,
                                random_goal_task=random_goal, random_y=random_y,
                                success_evaluation_res=success_evaluate_res_per_step)
                            writer.add_scalar(tag='episode_reward/train/per_env_step',
                                              scalar_value=average_episode_reward,
                                              global_step=env_step)
                            writer.add_scalar(tag='success_rate/train/per_env_step',
                                              scalar_value=average_success_rate,
                                              global_step=env_step)
                            print("[{} environment steps finished]: Average episode reward is {}".format(env_step,
                                                                                                         average_episode_reward))
                            model_path = 'models/' + task_name + '/' + method +  '/max_demo_' + str(
                                max_demo_num) + '/ratio_' + str(ratio) + '/'
                        else:
                            print("[{} environment steps finished]".format(env_step))
                            model_path = 'models/' + task_name + '/' + method + '/' + 'sub_' + str(args.sub_id) + '/max_demo_' + str(
                                max_demo_num) + '/ratio_' + str(ratio) + '/'

                        if not os.path.exists(model_path):
                            os.makedirs(model_path)
                        torch.save(policy_net.state_dict(), model_path + str(env_step) + '.pth')
                        # print("Model saved")

                    state = next_state
                    env_step += 1
                    demo_step += 1
                    demo_amount += 1

                    print("[Teacher Demo at env step {}]: {} total steps of demo were provided".format(env_step, demo_amount))
            else:
                if (method == 'isolated_active_human') and (args.load_joystick_demo == 0) and (demo_amount >= max_demo_amount):
                    break

                action = policy_net.get_action(state).detach()
                next_state, reward, done, _ = env.step(action.numpy())

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

                # evaluate policy
                if env_step % plot_interval == 0:
                    if not (method == "isolated_active_human"):
                        average_episode_reward, evaluate_res_per_step, average_success_rate, success_evaluate_res_per_step = evaluate_policy(
                            test_env=test_env, policy_net=policy_net,
                            evaluation_res=evaluate_res_per_step,
                            env_step=env_step, test_episode_num=100, total_seed_num=10,
                            random_goal_task=random_goal, random_y=random_y,
                            success_evaluation_res=success_evaluate_res_per_step)
                        writer.add_scalar(tag='episode_reward/train/per_env_step',
                                          scalar_value=average_episode_reward,
                                          global_step=env_step)
                        writer.add_scalar(tag='success_rate/train/per_env_step',
                                          scalar_value=average_success_rate,
                                          global_step=env_step)
                        print("[{} environment steps finished]: Average episode reward is {}".format(env_step,
                                                                                                     average_episode_reward))
                        model_path = 'models/' + task_name + '/' + method + '/max_demo_' + str(
                            max_demo_num) + '/ratio_' + str(ratio) + '/'
                    else:
                        print("[{} environment steps finished]".format(env_step))
                        model_path = 'models/' + task_name + '/' + method + '/' + 'sub_' + str(
                            args.sub_id) + '/max_demo_' + str(
                            max_demo_num) + '/ratio_' + str(ratio) + '/'

                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    torch.save(policy_net.state_dict(), model_path + str(env_step) + '.pth')
                    # print("Model saved")

                uncertainty_history.append(uncertainty)
                if len(uncertainty_history) > max_history_len:
                    uncertainty_history.pop(0)

                state = next_state
                env_step += 1

        if (method == 'isolated_active_human') and (args.load_joystick_demo == 0) and (demo_amount >= max_demo_amount):
            joystick_demo_traj_data_path = 'joystick_trajectory_data/' + task_name + '/' + method + '/' + 'sub_' + str(
                args.sub_id) + '/max_demo_' + str(max_demo_num) + '/ratio_' + str(ratio) + '/'
            if not os.path.exists(joystick_demo_traj_data_path):
                os.makedirs(joystick_demo_traj_data_path)

            np.savetxt(joystick_demo_traj_data_path + 'joystick_demo_state_trajs.csv', demo_state_trajs, delimiter=' ')
            np.savetxt(joystick_demo_traj_data_path + 'demo_state_start_ids.csv', demo_state_start_ids, delimiter=' ')

            np.savetxt(joystick_demo_traj_data_path + 'joystick_demo_action_trajs.csv', demo_action_trajs, delimiter=' ')
            np.savetxt(joystick_demo_traj_data_path + 'demo_action_start_ids.csv', demo_action_start_ids, delimiter=' ')

            joystick.stop()
            print("[Isolated Active Human]: All joystick demos are collected and saved. Going to quit...")

            break

    ''' Save the results '''
    if method == 'isolated_active_human':
        res_per_step_path = 'evaluation_res/new/' + task_name + '/' + method + '/' + 'sub_' + str(args.sub_id) + '/max_demo_' + str(
            max_demo_num) + '/ratio_' + str(
            ratio) + '/' + '/'
        if not os.path.exists(res_per_step_path):
            os.makedirs(res_per_step_path)

        # for visualization
        demo_traj_data_path = 'demo_trajectory_data/' + task_name + '/' + method + '/' + 'sub_' + str(args.sub_id) + '/max_demo_' + str(
            max_demo_num) + '/ratio_' + str(ratio) + '/'
        if not os.path.exists(demo_traj_data_path):
            os.makedirs(demo_traj_data_path)
    else:
        res_per_step_path = 'evaluation_res/new/' + task_name + '/' + method + '/max_demo_' + str(
            max_demo_num) + '/ratio_' + str(
            ratio) + '/' + '/'
        if not os.path.exists(res_per_step_path):
            os.makedirs(res_per_step_path)

        demo_traj_data_path = 'demo_trajectory_data/' + task_name + '/' + method + '/max_demo_' + str(
            max_demo_num) + '/ratio_' + str(ratio) + '/'
        if not os.path.exists(demo_traj_data_path):
            os.makedirs(demo_traj_data_path)

    if not (method == "isolated_active_human"):
        np.savetxt(res_per_step_path + 'res_per_step_new.csv', evaluate_res_per_step, delimiter=' ')
        np.savetxt(res_per_step_path + 'success_res_per_step_new.csv', success_evaluate_res_per_step, delimiter=' ')
        np.savetxt(demo_traj_data_path + 'demo_trajectory_data_new.csv', demo_traj_data, delimiter=' ')
    else:
        if args.load_joystick_demo == 0:
            pass
        else:
            trained_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, seed=seed).to(device)
            print("*******************************************")
            print("*******************************************")
            print("[Isolated Active Human]: Going to evaluate trained models ...")
            training_step_to_load = 0

            while training_step_to_load <= max_env_steps:
                model_path = 'models/' + task_name + '/' + method + '/' + 'sub_' + str(args.sub_id) + '/max_demo_' + str(max_demo_num) + '/ratio_' + str(
                        ratio) + '/' + str(training_step_to_load) + '.pth'
                trained_policy_net.load_state_dict(torch.load(model_path))

                average_episode_reward, evaluate_res_per_step, average_success_rate, success_evaluate_res_per_step = evaluate_policy(
                    test_env=test_env, policy_net=trained_policy_net,
                    evaluation_res=evaluate_res_per_step, env_step=training_step_to_load,
                    test_episode_num=100, total_seed_num=10, random_goal_task=random_goal, random_y=random_y,
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

            print("[Isolated Active Human]: Evaluations finished")


    if not (method == "isolated_active_human" and args.load_joystick_demo == 1):
        np.savetxt(demo_traj_data_path + 'demo_trajectory_data_new.csv', demo_traj_data, delimiter=' ')

    writer.close()
    print("************************")
    print("All training finished")


if __name__ == '__main__':
    main()