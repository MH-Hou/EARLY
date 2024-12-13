import numpy as np
import rospy
import roslib
import subprocess
import time
from sensor_msgs.msg import Joy
import math
from time import sleep

import sys
sys.path.append("/home/ullrich/catkin_ws/src/E-ARLD/")
from nav_env import NavEnv
from nav_env_render import dummy_nav_render


import warnings

warnings.filterwarnings("ignore")


class Joystick:
    def __init__(self, env_info, mode='episodic_active_human'):
        self.mode = mode

        rospy.init_node('joystick_demo_node', anonymous=True)
        self.joystick_demo_subscriber = rospy.Subscriber('/joy', Joy, self.joystick_demo_callback)
        self.rate = rospy.Rate(20)

        self.vx = 0.0
        self.vy = 0.0
        self.whether_demo_start = False

        if self.mode == 'passive_human':
            self.whether_selected_init = False
            self.confirm_button_res = 0.0
            self.init_pos_x = 10.0
            self.init_pos_y = 4.0
            self.goal_pos_x = 10.0
            self.goal_pos_y = 16.0

            self.delta_init_x = 0.0

            self.init_state_selection_subscriber = rospy.Subscriber('/joy', Joy, self.init_state_selection_callback)

        self.env = NavEnv(render=True)
        self.random_goal = env_info['random_goal']
        self.random_y = env_info['random_y']

    def joystick_demo_callback(self, data):
        vx = -1.0 * data.axes[0]
        vy = data.axes[1]

        if not self.whether_demo_start:
            if self.mode == 'passive_human':
                if self.whether_selected_init and (abs(vx) > 0.0 or abs(vy) > 0.0):
                    self.whether_demo_start = True
            else:
                if (abs(vx) > 0.0 or abs(vy) > 0.0):
                    self.whether_demo_start = True

        # if not self.whether_demo_start and (abs(vx) > 0.0 or abs(vy) > 0.0):
        #     self.whether_demo_start = True

        # when vx and vy are both within the unit circle
        if abs(vx) < 1.0 and abs(vy) < 1.0:
            self.vx = vx
            self.vy = vy
        # when vx + vy is the unit vector along the x or y axis
        elif (abs(vx) == 1.0 and abs(vy) == 0.0) or (abs(vx) == 0.0 and abs(vy) == 1.0):
            self.vx = vx
            self.vy = vy
        # when vx + vy is outside the unit circle
        else:
            if abs(vx) < 1.0:
                self.vx = vx
                self.vy = vy / abs(vy) * math.sqrt(1.0 - vx * vx)
            else:
                self.vy = vy
                self.vx = vx / abs(vx) * math.sqrt(1.0 - vy * vy)

        # print("[Joystick input]: original vx is {}, vy is {}".format(vx, vy))
        # print("[Joystick input]: processed vx is {}, vy is {}".format(self.vx, self.vy))

    def init_state_selection_callback(self, data):
        if not self.whether_selected_init:
            self.delta_init_x = -1.0 * 0.1 * data.axes[6]
            self.confirm_button_res = data.buttons[0]

    def select_init_state(self):
        self.whether_selected_init = False
        whether_selected_init_pos = False
        whether_selected_goal_pos = False
        self.confirm_button_res = 0.0
        self.init_pos_x = 10.0
        self.init_pos_y = 4.0
        self.goal_pos_x = 10.0
        self.goal_pos_y = 16.0
        self.delta_init_x = 0.0

        while not self.whether_selected_init:
            self.init_pos_x += self.delta_init_x

            starting_pos_to_draw = np.array([self.init_pos_x, self.init_pos_y])
            goal_pos_to_draw = np.array([self.goal_pos_x, self.goal_pos_y])
            self.env.render_env.draw_current_position(starting_pos_to_draw)
            self.env.render_env.draw_current_goal_position(goal_pos_to_draw)

            # sleep(0.1)

            if self.confirm_button_res == 1.0:
                self.whether_selected_init = True

        starting_pos = np.array([self.init_pos_x, self.init_pos_y])
        goal_pos = np.array([self.goal_pos_x, self.goal_pos_y])

        return starting_pos, goal_pos

    def provide_joystick_demo(self, starting_pos, goal_pos, reset_step=None):
        state_traj = []
        action_traj = []
        max_steps_per_demo = 10
        self.whether_demo_start = False

        if self.mode == 'isolated_active_human':
            state = self.env.reset(random=False, initial_x=starting_pos[0], initial_y=starting_pos[1],
                                   random_goal=False, goal_x=goal_pos[0], goal_y=goal_pos[1], random_y=False, reset_step=reset_step)
        else:
            state = self.env.reset(random=False, initial_x=starting_pos[0], initial_y=starting_pos[1],
                                   random_goal=False, goal_x=goal_pos[0], goal_y=goal_pos[1], random_y=False)

        done = False
        starting_pos_to_draw = np.array(starting_pos)
        goal_pos_to_draw = np.array(goal_pos)
        self.env.render_env.draw_current_position(starting_pos_to_draw)
        self.env.render_env.draw_current_goal_position(goal_pos_to_draw)

        # print("[Joystick]: Going to roll out ...")
        while not done:
            if not self.whether_demo_start:
                # print("[Joystick]: Haven't received any input from joystick")
                continue

            action = np.array([self.vx, self.vy])
            next_state, reward, done, _ = self.env.step(action)

            # print('[Joystick]: current state: {}'.format(state))
            # print('[Joystick]: current action: {}'.format(action))

            state_traj.append(state)
            action_traj.append(action)

            # print("[Joystick]: state and action are appended")

            # print("state traj: {}".format(state_traj))
            # print("action traj: {}".format(action_traj))

            self.env.render()
            state = next_state

            sleep(0.1)

            if self.mode == 'isolated_active_human':
                if len(action_traj) >= max_steps_per_demo:
                    print("[Joystick]: Isolated active mode, reached max steps amount per demo.")
                    break

        # append the terminal state
        state_traj.append(state)
        self.whether_demo_start = False

        if self.mode == 'passive_human':
            self.whether_selected_init = False

        state_traj = np.array(state_traj)
        action_traj = np.array(action_traj)

        print("[Joystick]: One demo was provided.")

        return state_traj, action_traj

    def stop(self):
        self.env.close()


if __name__ == '__main__':
    random_goal = False
    random_y = False
    env_info = {'random_goal': random_goal, 'random_y': random_y}

    joystick = Joystick(env_info=env_info, mode='passive_human')
    joystick.env.render_env.draw_text('Please wait for instruction ...')

    for i in range(3):
        joystick.env.render_env.draw_text('[Demo {}: Please provide a demo]'.format(i + 1))

        # starting_pos = [10.0, 4.0]
        # goal_pos = [10.0, 16.0]

        starting_pos, goal_pos = joystick.select_init_state()

        joystick.provide_joystick_demo(starting_pos=starting_pos, goal_pos=goal_pos)
        joystick.env.render_env.clean()
        joystick.env.render_env.draw_text('Please wait for instruction ...')

        sleep(2.0)

    joystick.stop()
    print("All finished")
