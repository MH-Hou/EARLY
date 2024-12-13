import random
import numpy as np
import torch
import argparse
from matplotlib import pyplot as plt

from nav_env import NavEnv
from activesac_nav import NormalizedActions, ValueNetwork, SoftQNetwork, PolicyNetwork, evaluate_policy


use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', help='name of task')
    parser.add_argument('--step_to_check', help='training step to check', type=int)

    return parser.parse_args()


def plot_rollout_trajs(success_trajs, fail_trajs, arrival_thres=1.0):
    all_trajs = [success_trajs, fail_trajs]
    all_cleaned_trajs = []
    for rollout_trajs in all_trajs:
        traj_id = 0
        steps = rollout_trajs.shape[0]
        all_trajs_cleaned = []
        traj = []  # a list of (x, y) points
        for step in range(steps):
            if rollout_trajs[step][0] == np.inf:
                # not to save when the traj is empty
                if traj_id > 0:
                    traj = np.array(traj) # 2d np array
                    all_trajs_cleaned.append(traj)
                    # print("Finished loading trajectory {}".format(traj_id))

                traj = []
                # print("Start to load trajectory {}".format(traj_id + 1))
                traj_id += 1
                continue

            new_point = [rollout_trajs[step][0], rollout_trajs[step][1]]
            traj.append(new_point)

            # save the last traj
            if step == (steps - 1):
                traj = np.array(traj)  # 2d np array
                all_trajs_cleaned.append(traj)
                # print("Finished loading trajectory {}".format(traj_id))

        all_cleaned_trajs.append(all_trajs_cleaned)

    print("Finished cleaning all trajectories")

    fig, ax = plt.subplots(figsize=(6, 6))

    # plot the scene
    ax.plot([5.2, 12.8], [10, 10], c='black', linewidth=2)
    ax.plot([15.2, 16.8], [10, 10], c='black', linewidth=2)
    circle1 = plt.Circle((10.0, 16.0), arrival_thres, color='blue', fill=False)

    plt.xlim(0, 20)
    plt.ylim(0, 20)
    ax.set_aspect('equal', adjustable='box')
    ax.add_patch(circle1)

    success_cleaned_trajs = all_cleaned_trajs[0]
    fail_cleaned_trajs = all_cleaned_trajs[1]

    for i in range(len(success_cleaned_trajs)):
        traj = success_cleaned_trajs[i]

        ax.plot(traj[:, 0], traj[:, 1], c='g')

    for i in range(len(fail_cleaned_trajs)):
        traj = fail_cleaned_trajs[i]

        ax.plot(traj[:, 0], traj[:, 1], c='r')

    plt.show()

def generate_policy_test_rollouts(test_env, policy_net, seed, random_goal_task, random_y):
    test_episode_num = 20
    test_episode_reward = 0.0
    rollout_trajs = []
    success_rollout_trajs = []
    fail_rollout_trajs = []

    sampled_xs = np.random.default_rng(seed).uniform(low=0.01, high=19.9, size=test_episode_num)
    if random_goal_task:
        sampled_goal_xs = np.random.default_rng(seed).uniform(low=0.01, high=19.9, size=test_episode_num)
        if random_y:
            sampled_ys = np.random.default_rng(seed).uniform(low=1.0, high=8.0, size=test_episode_num)
            sampled_goal_ys = np.random.default_rng(seed).uniform(low=12.0, high=19.0, size=test_episode_num)

    for i in range(test_episode_num):
        if random_goal_task:
            if random_y:
                state_ = test_env.reset(random=False, initial_x=sampled_xs[i], initial_y=sampled_ys[i],
                                        random_goal=False,
                                        goal_x=sampled_goal_xs[i], goal_y=sampled_goal_ys[i], random_y=False)
            else:
                state_ = test_env.reset(random=False, initial_x=sampled_xs[i], initial_y=4.0, random_goal=False,
                                        goal_x=sampled_goal_xs[i], goal_y=16.0, random_y=False)
        else:
            state_ = test_env.reset(random=False, initial_x=sampled_xs[i], initial_y=4.0)

        done_ = False
        traj = [[np.inf, np.inf]]
        episode_reward = 0.0
        step = 0
        while not done_:
            action_ = policy_net.get_action(state_).detach()
            next_state_, reward_, done_, _ = test_env.step(action_.numpy())
            episode_reward += reward_
            test_episode_reward += reward_
            traj.append([state_[2], state_[3]])
            state_ = next_state_
            step += 1

        # store the last one
        traj.append([state_[2], state_[3]])
        rollout_trajs.extend(traj)

        if reward_ == 1000:
            print("[Test episode {} with length {}]: success!".format(i + 1, step))
            success_rollout_trajs.extend(traj)
        else:
            fail_rollout_trajs.extend(traj)

            if reward_ == -1000:
                print("[Test episode {} with length {}]: failure because of collision".format(i + 1, step))
            else:
                print("[Test episode {} with length {}]: failure because of time-out".format(i + 1, step))


        print("[Test episode {}]: episode reward is {}".format(i + 1, episode_reward))

    print("Average test episode reward: {}".format(test_episode_reward/test_episode_num))

    return np.array(rollout_trajs), np.array(success_rollout_trajs), np.array(fail_rollout_trajs)


def main():
    args = argparser()
    wall_thickness = 0.1
    arrival_thres = 1.0

    """ prepare the environment """
    task_name = args.task_name
    if task_name == 'nav_1':
        random_goal = False
        random_y = False
        ratio = 0.3
    elif task_name == 'nav_2':
        random_goal = True
        random_y = False
        ratio = 0.3
    else:
        random_goal = True
        random_y = True
        ratio = 0.4

    nav_env = NavEnv(render=False, wall_thickness=wall_thickness, arrival_thres=arrival_thres)
    env = NormalizedActions(nav_env)
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    hidden_dim = 256
    seed = 6
    random.seed(seed)
    np.random.seed(seed)

    ''' create nn models '''
    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, seed=seed).to(device)

    """ Load trained model """
    method = 'active_sac'
    max_demo_num = 60
    uncertainty_method = 'td_error'
    training_step_to_load = args.step_to_check
    model_path = 'models/' + task_name + '/' + method + '/max_demo_' + str(max_demo_num) + '/ratio_' + str(ratio) + '/' \
                  + uncertainty_method + '/per_step/' + str(training_step_to_load) + '.pth'
    policy_net.load_state_dict(torch.load(model_path))

    rollout_trajs, success_trajs, fail_trajs = generate_policy_test_rollouts(test_env=nav_env,
                                                  policy_net=policy_net,
                                                  seed=seed,
                                                  random_goal_task=random_goal,
                                                  random_y=random_y)

    plot_rollout_trajs(success_trajs=success_trajs, fail_trajs=fail_trajs, arrival_thres=arrival_thres)


if __name__ == '__main__':
    main()

