import numpy as np
from time import sleep
import os


from scripts.joystick import Joystick


import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub_id', help='id of human subject', default=1, type=int)

    return parser.parse_args()


def main():
    args = argparser()
    task_name = 'nav_1'
    random_goal = False
    random_y = False
    method = 'ddpg_lfd_human'
    max_demo_num = 60
    demo_traj_data = []  # a list of 2d trajectories, each trajectory include a bunch of (x,y) points
    demo_state_start_ids = []
    demo_action_start_ids = []
    demo_state_trajs = []
    demo_action_trajs = []
    joystick_demo_traj_data_path = 'joystick_trajectory_data/' + task_name + '/' + method + '/' + 'sub_' + str(
        args.sub_id) + '/max_demo_' + str(max_demo_num) + '/'
    if not os.path.exists(joystick_demo_traj_data_path):
        os.makedirs(joystick_demo_traj_data_path)
    demo_traj_data_path = 'demo_trajectory_data/' + task_name + '/' + method + '/' + 'sub_' + str(
        args.sub_id) + '/max_demo_' + str(max_demo_num) + '/'
    if not os.path.exists(demo_traj_data_path):
        os.makedirs(demo_traj_data_path)

    print("[DDPG LfD Human]: Start to collect teacher demo learning ... ")

    env_info = {'random_goal': random_goal, 'random_y': random_y}
    joystick = Joystick(env_info=env_info, mode='passive_human')
    joystick.env.render_env.draw_text('Please wait for instruction ...')
    for demo_num in range(max_demo_num):
        joystick.env.render_env.draw_text('[Demo {}: Please provide a demo]'.format(demo_num + 1))

        starting_pos, goal_pos = joystick.select_init_state()

        oracle_states, oracle_actions = joystick.provide_joystick_demo(starting_pos=starting_pos, goal_pos=goal_pos)
        joystick.env.render_env.clean()
        if demo_num < (max_demo_num - 1):
            joystick.env.render_env.draw_text('Please wait for instruction ...')
        else:
            joystick.env.render_env.draw_text('Great! All demos are provided!')

        # save demo data
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

        # demo data for visualization
        demo_traj = np.array([[np.inf, np.inf]])
        demo_traj = np.concatenate((demo_traj, oracle_states[:, 2:]))  # 2d np array
        demo_traj = demo_traj.tolist()  # 2d list of (x, y)
        demo_traj_data.extend(demo_traj)  # 2d list of (x, y)

        sleep(2.0)

    # save results
    np.savetxt(joystick_demo_traj_data_path + 'joystick_demo_state_trajs.csv', demo_state_trajs, delimiter=' ')
    np.savetxt(joystick_demo_traj_data_path + 'demo_state_start_ids.csv', demo_state_start_ids, delimiter=' ')
    np.savetxt(joystick_demo_traj_data_path + 'joystick_demo_action_trajs.csv', demo_action_trajs, delimiter=' ')
    np.savetxt(joystick_demo_traj_data_path + 'demo_action_start_ids.csv', demo_action_start_ids, delimiter=' ')
    np.savetxt(demo_traj_data_path + 'demo_trajectory_data_new.csv', demo_traj_data, delimiter=' ')

    joystick.stop()
    print("[DDPG LfD Human]: All joystick demos are collected and saved. Going to quit...")


if __name__ == '__main__':
    main()