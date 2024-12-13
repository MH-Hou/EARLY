import numpy as np
from torch.utils.data.dataset import Dataset, random_split
from stable_baselines3 import PPO, A2C
from datetime import datetime

from nav_oracle_bc_ppo import ExpertDataSet, pretrain_agent
from nav_env import NavEnv

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

def calculate_traj_feature(traj_states):
    waypoints = traj_states[:, 2:4] # 2d np array in the form of (total_steps_num, pos_dimension)
    xs = waypoints[:, 0]
    ys = waypoints[:, 1]

    y_distances = np.abs(ys - 10.0) # distance to the obstacle line in y direction
    min_id = np.argmin(y_distances)

    phi = xs[min_id]

    return phi

def mode_uniform_sample(trajs_states_all_modes, trajs_actions_all_modes):
    total_modes_num = len(trajs_states_all_modes)
    modes_trajs_num_list = []
    for trajs_states_mode in trajs_states_all_modes:
        modes_trajs_num_list.append(len(trajs_states_mode)) # get the total trajs num under a given mode

    modes_trajs_num_min = min(modes_trajs_num_list)
    res_trajs_states = [] # a list of 2d np array trajs with various lengths
    res_trajs_actions = []

    for mode_id in range(total_modes_num):
        trajs_states_current_mode = trajs_states_all_modes[mode_id] # a list of 2d np array trajs
        trajs_actions_current_mode = trajs_actions_all_modes[mode_id] # a list of 2d np array trajs
        modes_trajs_num = len(trajs_states_current_mode)
        if modes_trajs_num == modes_trajs_num_min:
            for i in range(modes_trajs_num):
                res_trajs_states.append(trajs_states_current_mode[i])
                res_trajs_actions.append(trajs_actions_current_mode[i])
        else:
            sampled_ids = np.random.choice(modes_trajs_num, modes_trajs_num_min) # 1d np array of sampled indices
            for i in sampled_ids:
                res_trajs_states.append(trajs_states_current_mode[i])
                res_trajs_actions.append(trajs_actions_current_mode[i])

    demo_states = None  # 2d np array as (total_steps_num, state_dimension)
    demo_actions = None
    for i in range(len(res_trajs_states)):
        trajs_states = res_trajs_states[i] # 2d np array as (total_steps_num, state_dimension)
        trajs_actions = res_trajs_actions[i]

        if demo_states is None:
            demo_states = trajs_states
            demo_actions = trajs_actions
        else:
            demo_states = np.concatenate((demo_states, trajs_states))
            demo_actions = np.concatenate((demo_actions, trajs_actions))

    sampled_trajs_num = len(res_trajs_states)

    return demo_states, demo_actions, sampled_trajs_num

def initial_state_uniform_sample(trajs_states_all_modes, trajs_actions_all_modes, sample_trajs_num):
    total_modes_num = len(trajs_states_all_modes)
    all_trajs_states = [] # a list of 2d np array trajs with various lengths
    all_trajs_actions = []
    res_trajs_states= [] # a list of 2d np array trajs with various lengths
    res_trajs_actions = []

    for mode_id in range(total_modes_num):
        trajs_states_current_mode = trajs_states_all_modes[mode_id]  # a list of 2d np array trajs with various lengths
        trajs_actions_current_mode = trajs_actions_all_modes[mode_id]

        for i in range(len(trajs_states_current_mode)):
            all_trajs_states.append(trajs_states_current_mode[i])
            all_trajs_actions.append(trajs_actions_current_mode[i])

    total_trajs_num = len(all_trajs_states)
    sampled_ids = np.random.choice(total_trajs_num, sample_trajs_num)
    for id in sampled_ids:
        res_trajs_states.append(all_trajs_states[id])
        res_trajs_actions.append(all_trajs_actions[id])

    demo_states = None # 2d np array as (total_steps_num, state_dimension)
    demo_actions = None
    for i in range(len(res_trajs_states)):
        trajs_states = res_trajs_states[i]
        trajs_actions = res_trajs_actions[i]

        # print('shape of trajs_states: {}'.format(np.shape(trajs_states)))
        # print('shape of trajs_actions: {}'.format(np.shape(trajs_actions)))
        # if demo_states is not None:
        #     print('shape of demo_states: {}'.format(np.shape(demo_states)))
        #     print('shape of demo_actions: {}'.format(np.shape(demo_actions)))

        if demo_states is None:
            demo_states = trajs_states
            demo_actions = trajs_actions
        else:
            demo_states = np.concatenate((demo_states, trajs_states))
            demo_actions = np.concatenate((demo_actions, trajs_actions))

    return demo_states, demo_actions

def bc_train(expert_observations, expert_actions, sample_type):
    env = NavEnv(render=False)
    a2c_student = PPO('MlpPolicy', env, seed=1, gamma=1.0, verbose=1, policy_kwargs=dict(net_arch=[64, 64]))

    expert_dataset = ExpertDataSet(expert_observations, expert_actions)
    train_size = int(0.8 * len(expert_dataset))
    test_size = len(expert_dataset) - train_size
    train_expert_dataset, test_expert_dataset = random_split(
        expert_dataset, [train_size, test_size]
    )
    print("test_expert_dataset: ", len(test_expert_dataset))
    print("train_expert_dataset: ", len(train_expert_dataset))

    print("Going to pretrain PPO with BC ... ")
    student = pretrain_agent(student=a2c_student,
                             env=env,
                             train_expert_dataset=expert_dataset,
                             test_expert_dataset=expert_dataset,
                             epochs=500,
                             scheduler_gamma=0.1,
                             learning_rate=1.0,
                             log_interval=100,
                             no_cuda=True,
                             seed=1,
                             batch_size=30,
                             test_batch_size=30
                             )

    student.save("rrt_oracle/clean_bc" + TIMESTAMP)
    # student.save("rrt_oracle/bc_wo_ppo_" + sample_type + TIMESTAMP)

    env.close()
    print("BC without PPO training finished!")



def main():
    # 2d np array in the form of (total_steps_num, state/action_dimension)
    oracle_states = np.genfromtxt("rrt_oracle/demos/oracle_states.csv", delimiter=" ")
    oracle_actions = np.genfromtxt("rrt_oracle/demos/oracle_actions.csv", delimiter=" ")

    total_steps_num = np.shape(oracle_states)[0]
    trajs_states_mode_1 = [] # a list of 2d np array, (total_episode_num, episode_steps_num, state_dimension)
    trajs_states_mode_2 = []
    trajs_states_mode_3 = []
    trajs_actions_mode_1 = []
    trajs_actions_mode_2 = []
    trajs_actions_mode_3 = []
    new_episode_states = None
    new_episode_actions = None
    for i in range(total_steps_num):
        if i == 0:
            new_episode_states = []
            new_episode_actions = []
        elif oracle_states[i][0] == np.inf or i == (total_steps_num - 1):
            new_episode_states = np.array(
                new_episode_states)  # 2d np array in the form of (episode_steps_num, state_dimension)
            new_episode_actions = np.array(new_episode_actions)
            phi = calculate_traj_feature(new_episode_states)

            if phi <= 6.0:
                trajs_states_mode_1.append(new_episode_states)
                trajs_actions_mode_1.append(new_episode_actions)
            elif phi <= 16.0:
                trajs_states_mode_2.append(new_episode_states)
                trajs_actions_mode_2.append(new_episode_actions)
            else:
                trajs_states_mode_3.append(new_episode_states)
                trajs_actions_mode_3.append(new_episode_actions)

            new_episode_states = []
            new_episode_actions = []
        else:
            new_episode_states.append(oracle_states[i])
            new_episode_actions.append(oracle_actions[i])

    print("Mode 1 trajs num: {}".format(len(trajs_states_mode_1)))
    print("Mode 2 trajs num: {}".format(len(trajs_states_mode_2)))
    print("Mode 3 trajs num: {}".format(len(trajs_states_mode_3)))

    trajs_states_all_modes = [trajs_states_mode_1, trajs_states_mode_2, trajs_states_mode_3]
    trajs_actions_all_modes = [trajs_actions_mode_1, trajs_actions_mode_2, trajs_actions_mode_3]

    demo_states_mode_uniform, demo_actions_mode_uniform, sampled_trajs_num = mode_uniform_sample(trajs_states_all_modes=trajs_states_all_modes,
                                                                                                 trajs_actions_all_modes=trajs_actions_all_modes)

    demo_states_init_uniform, demo_actions_init_uniform = initial_state_uniform_sample(trajs_states_all_modes=trajs_states_all_modes,
                                                                                       trajs_actions_all_modes=trajs_actions_all_modes,
                                                                                       sample_trajs_num=sampled_trajs_num)

    # np.savetxt("rrt_oracle/demos/demo_states_mode_uniform.csv", demo_states_mode_uniform, delimiter=" ")
    # np.savetxt("rrt_oracle/demos/demo_actions_mode_uniform.csv", demo_actions_mode_uniform, delimiter=" ")
    # np.savetxt("rrt_oracle/demos/demo_states_init_uniform.csv", demo_states_init_uniform, delimiter=" ")
    # np.savetxt("rrt_oracle/demos/demo_actions_init_uniform.csv", demo_actions_init_uniform, delimiter=" ")


def run_trainings():
    demo_states_mode_uniform = np.genfromtxt("rrt_oracle/demos/demo_states_mode_uniform.csv", delimiter=" ")
    demo_actions_mode_uniform = np.genfromtxt("rrt_oracle/demos/demo_actions_mode_uniform.csv", delimiter=" ")
    demo_states_init_uniform = np.genfromtxt("rrt_oracle/demos/demo_states_init_uniform.csv", delimiter=" ")
    demo_actions_init_uniform = np.genfromtxt("rrt_oracle/demos/demo_actions_init_uniform.csv", delimiter=" ")


    print("mode uniform data size: {}".format(np.shape(demo_states_mode_uniform)))
    print("init uniform data size: {}".format(np.shape(demo_states_init_uniform)))


    bc_train(expert_observations=demo_states_mode_uniform,
             expert_actions=demo_actions_mode_uniform,
             sample_type="mode_uniform")

    bc_train(expert_observations=demo_states_init_uniform,
             expert_actions=demo_actions_init_uniform,
             sample_type="init_uniform")


def clean_bc_train():
    demo_states =np.genfromtxt("rrt_oracle/demos/pure_oracle_states.csv", delimiter=" ")
    demo_actions = np.genfromtxt("rrt_oracle/demos/pure_oracle_actions.csv", delimiter=" ")

    bc_train(demo_states, demo_actions, sample_type=None)

if __name__ == '__main__':
    # main()
    # run_trainings()
    clean_bc_train()

