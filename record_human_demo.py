from pynput import keyboard
import numpy as np
import redis
import threading
import argparse

from nav_env_render import dummy_nav_render

class Recorder:
    def __init__(self, max_demo_num=5, default_starting_pos=[10.0, 4.0], default_goal_pos=[10.0, 16.0], task_name='nav_1', subject_id=0):
        self.demo_num = 0
        self.max_demo_num = max_demo_num
        self.default_starting_position = np.array(default_starting_pos)
        self.default_goal_position = np.array(default_goal_pos)
        self.starting_position = np.array(default_starting_pos)
        self.goal_position = np.array(default_goal_pos)

        self.r = redis.Redis(host='localhost', port=6379, db=0)

        self.task_name = task_name
        self.subject_id = subject_id
        self.human_data = [] # [init_x, init_y, goal_x, goal_y]
        if self.task_name == 'nav_1':
            self.init_pos_min = [0.1, 4.0] # (min_x, min_y)
            self.init_pos_max = [19.9, 4.0] # (min_x, min_y)
            self.goal_pos_min = [0.1, 16.0]
            self.goal_pos_max = [19.9, 16.0]
            self.whether_chose_init = False
            self.whether_chose_goal = True
        elif self.task_name == 'nav_2':
            self.init_pos_min = [0.1, 4.0]
            self.init_pos_max = [19.9, 4.0]
            self.goal_pos_min = [0.1, 16.0]
            self.goal_pos_max = [19.9, 16.0]
            self.whether_chose_init = False
            self.whether_chose_goal = False
        else:
            self.init_pos_min = [0.1, 1.0]
            self.init_pos_max = [19.9, 8.0]
            self.goal_pos_min = [0.1, 12.0]
            self.goal_pos_max = [19.9, 19.0]
            self.whether_chose_init = False
            self.whether_chose_goal = False

    def on_press(self, key):
        try:
            if key.char == 'a':
                if not self.whether_chose_init:
                    self.starting_position[0] -= 0.5
                    self.starting_position = np.clip(self.starting_position, self.init_pos_min, self.init_pos_max)
                    print('[Choosing starting position]: Moved left: new starting position is {}'.format(self.starting_position))
                elif not self.whether_chose_goal:
                    self.goal_position[0] -= 0.5
                    self.goal_position = np.clip(self.goal_position, self.goal_pos_min, self.goal_pos_max)
                    print('[Choosing goal position]: Moved left: new goal position is {}'.format(self.goal_position))

                string_data = ''
                for data in self.starting_position:
                    string_data += str(data) + ' '
                self.r.publish('init_pos', string_data)

                string_data = ''
                for data in self.goal_position:
                    string_data += str(data) + ' '
                self.r.publish('goal_pos', string_data)

            elif key.char == 'd':
                if not self.whether_chose_init:
                    self.starting_position[0] += 0.5
                    self.starting_position = np.clip(self.starting_position, self.init_pos_min, self.init_pos_max)
                    print('[Choosing starting position]: Moved right: new starting position is {}'.format(self.starting_position))
                elif not self.whether_chose_goal:
                    self.goal_position[0] += 0.5
                    self.goal_position = np.clip(self.goal_position, self.goal_pos_min, self.goal_pos_max)
                    print('[Choosing goal position]: Moved right: new goal position is {}'.format(self.goal_position))

                string_data = ''
                for data in self.starting_position:
                    string_data += str(data) + ' '
                self.r.publish('init_pos', string_data)

                string_data = ''
                for data in self.goal_position:
                    string_data += str(data) + ' '
                self.r.publish('goal_pos', string_data)

            elif key.char == 'w' and self.task_name == 'nav_3':
                if not self.whether_chose_init:
                    self.starting_position[1] += 0.5
                    self.starting_position = np.clip(self.starting_position, self.init_pos_min, self.init_pos_max)
                    print('[Choosing starting position]: Moved up: new starting position is {}'.format(
                        self.starting_position))
                elif not self.whether_chose_goal:
                    self.goal_position[1] += 0.5
                    self.goal_position = np.clip(self.goal_position, self.goal_pos_min, self.goal_pos_max)
                    print('[Choosing goal position]: Moved up: new goal position is {}'.format(self.goal_position))

                string_data = ''
                for data in self.starting_position:
                    string_data += str(data) + ' '
                self.r.publish('init_pos', string_data)

                string_data = ''
                for data in self.goal_position:
                    string_data += str(data) + ' '
                self.r.publish('goal_pos', string_data)

            elif key.char == 's' and self.task_name == 'nav_3':
                if not self.whether_chose_init:
                    self.starting_position[1] -= 0.5
                    self.starting_position = np.clip(self.starting_position, self.init_pos_min, self.init_pos_max)
                    print('[Choosing starting position]: Moved up: new starting position is {}'.format(
                        self.starting_position))
                elif not self.whether_chose_goal:
                    self.goal_position[1] -= 0.5
                    self.goal_position = np.clip(self.goal_position, self.goal_pos_min, self.goal_pos_max)
                    print('[Choosing goal position]: Moved up: new goal position is {}'.format(self.goal_position))

                string_data = ''
                for data in self.starting_position:
                    string_data += str(data) + ' '
                self.r.publish('init_pos', string_data)

                string_data = ''
                for data in self.goal_position:
                    string_data += str(data) + ' '
                self.r.publish('goal_pos', string_data)


        except AttributeError:
            print('special key {0} pressed'.format(key))

    def on_release(self, key):
        if key == keyboard.Key.enter:
            if not self.whether_chose_init:
                self.whether_chose_init = True
            elif not self.whether_chose_goal:
                self.whether_chose_goal = True

            # after both initial position and goal position are chosen
            if self.whether_chose_init and self.whether_chose_goal:
                self.whether_chose_init = False
                if not self.task_name == 'nav_1':
                    self.whether_chose_goal = False

                self.demo_num += 1
                self.human_data.append([self.starting_position[0], self.starting_position[1],
                                        self.goal_position[0], self.goal_position[1]])

                self.r.publish('demo_num', str(self.demo_num))

                self.starting_position = self.default_starting_position.copy()
                print("reset starting position to {}".format(self.starting_position))
                self.goal_position = self.default_goal_position.copy()
                print("reset goal position to {}".format(self.goal_position))

                print("[Demo {}]: Finished".format(self.demo_num))

            string_data = ''
            for data in self.starting_position:
                string_data += str(data) + ' '
            self.r.publish('init_pos', string_data)

            string_data = ''
            for data in self.goal_position:
                string_data += str(data) + ' '
            self.r.publish('goal_pos', string_data)

            if self.demo_num == self.max_demo_num:
                self.r.publish('stop', str(0))

            # Stop listener
            return False

    def save_data(self):
        path = 'human_data/'
        human_data = np.array(self.human_data)
        np.savetxt(path + self.task_name + '_' + 'sub_' + str(self.subject_id) + '_' + 'init_and_goal.csv', human_data, delimiter=' ')


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub_id', help='id of human subject', default=0, type=int)
    parser.add_argument('--task_id', help='id of nav task', default=1, type=int)

    return parser.parse_args()

def main():
    args = argparser()

    max_demo_num = 60
    sub_id = args.sub_id
    task_name = 'nav_' + str(args.task_id)

    recorder = Recorder(max_demo_num=max_demo_num, task_name=task_name, subject_id=sub_id)

    while recorder.demo_num < max_demo_num:
        with keyboard.Listener(
                on_press=recorder.on_press,
                on_release=recorder.on_release) as listener:
            listener.join()

    recorder.save_data()
    print("All finished")


if __name__ == '__main__':
    main()