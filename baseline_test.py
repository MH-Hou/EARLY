import gym
import random
import numpy as np
import torch
import torch.nn as nn

from nav_env import NavEnv
from nav_oracle_rrt import RRT_Oracle
from activesac_nav import NormalizedActions, ValueNetwork, SoftQNetwork, PolicyNetwork, evaluate_policy

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


def visualized_test(policy_net, seed, test_episode_num=10):
    test_episode_reward = 0.0
    test_env = NavEnv(render=True)
    sampled_xs = np.random.default_rng(seed).uniform(low=0.01, high=19.9, size=test_episode_num)

    for i in range(test_episode_num):
        state_ = test_env.reset(random=False, initial_x=sampled_xs[i], initial_y=4.0)
        done_ = False
        episode_reward = 0.0
        while not done_:
            action_ = policy_net.get_action(state_).detach()
            next_state_, reward_, done_, _ = test_env.step(action_.numpy())
            episode_reward += reward_
            test_episode_reward += reward_
            state_ = next_state_

        print("[Test episode {}]: episode reward is {}".format(i + 1, episode_reward))

    average_episode_reward = test_episode_reward / test_episode_num
    print("All tests finished. Average episode reward is {}".format(average_episode_reward))



def main():
    """ prepare the environment """
    nav_env = NavEnv(render=False)
    env = NormalizedActions(nav_env)
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    hidden_dim = 256
    seed = 3
    random.seed(seed)
    np.random.seed(seed)

    ''' create nn models '''
    # value_net = ValueNetwork(state_dim, hidden_dim, seed=seed).to(device)
    # target_value_net = ValueNetwork(state_dim, hidden_dim, seed=seed).to(device)
    # soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim, seed=seed).to(device)
    # soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim, seed=seed).to(device)
    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, seed=seed).to(device)

    method = 'active_sac'
    # method = 'active_sac_bern'
    # method = 'sac_lfd'
    max_demo_num = 20
    ratio = 0.1
    uncertainty_method = 'td_error'
    env_step = int(20 * 1e3)
    model_path = 'models/' + method + '/max_demo_' + str(max_demo_num) + '/ratio_' + str(
        ratio) + '/' + uncertainty_method + '/per_step/' + str(env_step) + '.pth'
    policy_net.load_state_dict(torch.load(model_path))
    policy_net.eval()

    average_episode_reward, _ = evaluate_policy(test_env=env,
                                                policy_net=policy_net,
                                                evaluation_res=[],
                                                env_step=env_step,
                                                test_episode_num=100,
                                                total_seed_num=10)
    print(average_episode_reward)

    visualized_test(policy_net=policy_net,
                    seed=78,
                    test_episode_num=10)


if __name__ == '__main__':
    main()




