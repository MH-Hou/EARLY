import numpy as np
import gym
from gym import spaces
from stable_baselines3.common.env_checker import check_env

from nav_env_render import dummy_nav_render


class NavEnv(gym.Env):
    def __init__(self, render=False, target_pos=(10.0, 16.0), obstacle_area=([[5, 10], [13, 10]], [[15, 10], [17, 10]]), wall_thickness=0.0, arrival_thres=1.0, task_name=None):
        super(NavEnv, self).__init__()

        self.render_mode = 'human'

        self.task_name = task_name
        if task_name is not None:
            if task_name == 'nav_1':
                self.random_goal = False
                self.random_y = False
            elif task_name == 'nav_2':
                self.random_goal = True
                self.random_y = False
            else:
                self.random_goal = True
                self.random_y = True

        self.target_x = target_pos[0]
        self.target_y = target_pos[1]
        self.x = None
        self.y = None

        self.wall_thickness = wall_thickness

        # define state as s_t = (p_targ_x, p_targ_y, p_t_x, p_t_y)
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0]), high=np.array([20.0, 20.0, 20.0, 20.0]),
                                            shape=(4,), dtype=np.float32)

        # define action as action a_t = (v_t+_x, v_t+_y)
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]),
                                            shape=(2,), dtype=np.float32)

        self.obstacle_area = obstacle_area

        self.delta_t = 1.0
        self.current_step = 0
        self.max_steps = 200
        self.arrival_thres = arrival_thres

        self.render_env = None
        if render:
          self.render_env = dummy_nav_render()

    def reset(self, random=True, initial_x=10.0, initial_y=4.0, random_goal=False, goal_x=10.0, goal_y=16.0, random_y=False, reset_step=None, seed=None, options=None):
        if self.task_name is None:
            # reset initial pos
            if random:
                self.x = np.random.uniform(self.observation_space.low[2] + 0.01, self.observation_space.high[2] - 0.01, 1)[0]
                if random_y:
                    self.y = np.random.uniform(1.0, 8.0, 1)[0]
                else:
                    self.y = initial_y
            else:
                self.x = initial_x
                self.y = initial_y

            # reset goal pos
            if random_goal:
                self.target_x = np.random.uniform(self.observation_space.low[2] + 0.01, self.observation_space.high[2] - 0.01, 1)[0]
                if random_y:
                    self.target_y = np.random.uniform(12.0, 19.0, 1)[0]
                else:
                    self.target_y = goal_y
            else:
                self.target_x = goal_x
                self.target_y = goal_y
        else:
            if self.task_name == 'nav_1':
                self.x = np.random.uniform(self.observation_space.low[2] + 0.01, self.observation_space.high[2] - 0.01, 1)[0]
                self.y = initial_y
                self.target_x = goal_x
                self.target_y = goal_y
            elif self.task_name == 'nav_2':
                self.x = np.random.uniform(self.observation_space.low[2] + 0.01, self.observation_space.high[2] - 0.01, 1)[0]
                self.y = initial_y
                self.target_x = np.random.uniform(self.observation_space.low[2] + 0.01, self.observation_space.high[2] - 0.01, 1)[0]
                self.target_y = goal_y
            else:
                self.x = np.random.uniform(self.observation_space.low[2] + 0.01, self.observation_space.high[2] - 0.01, 1)[0]
                self.y = np.random.uniform(1.0, 8.0, 1)[0]
                self.target_x = np.random.uniform(self.observation_space.low[2] + 0.01, self.observation_space.high[2] - 0.01, 1)[0]
                self.target_y = np.random.uniform(12.0, 19.0, 1)[0]

        # self.y = initial_y

        if reset_step is None:
            self.current_step = 0
        else:
            self.current_step = reset_step

        state = np.array([self.target_x, self.target_y, self.x, self.y]).astype(np.float32)

        return state

    ''' 
        state s_t = (p_targ_x, p_targ_y, p_t_x, p_t_y)
        action a_t = (v_t+_x, v_t+_y)
    '''
    def step(self, action):
        self.current_step += 1

        v_x = action[0] # v_t+_x
        v_y = action[1] # v_t+_y

        x_next = self.x + v_x * self.delta_t
        y_next = self.y + v_y * self.delta_t

        if x_next <= 0.0:
            x_next = 0.0
        elif x_next >= 20.0:
            x_next = 20.0

        if y_next <= 0.0:
            y_next = 0.0
        elif y_next >= 20.0:
            y_next = 20.0

        p_t = np.array([x_next, y_next])
        p_targ_t = np.array([self.target_x, self.target_y])

        if np.linalg.norm(p_targ_t - p_t) <= self.arrival_thres:
            reward = 1000.0
            done = True
            self.x = x_next  # p_t_x
            self.y = y_next  # p_t_y
        elif self.obstacle_collision_check_new(curr_x=self.x, curr_y=self.y,
                                           next_x=x_next, next_y=y_next):
            reward = -1000.0
            done = True
        else:
            # reward = -0.01 * np.linalg.norm(p_targ_t - p_t)
            reward = -1.0
            if self.current_step >= self.max_steps:
                done = True
            else:
                done = False
            self.x = x_next  # p_t_x
            self.y = y_next  # p_t_y

        state = np.array([self.target_x, self.target_y, self.x, self.y]).astype(np.float32)
        info = {}

        return state, reward, done, info

    def render(self, mode='console'):
        if self.render_env is None:
            pass
        else:
            self.render_env.draw_current_position(pos=np.array([self.x, self.y]))

    def close(self):
        if self.render_env is None:
            pass
        else:
            self.render_env.stop_render()

    ''' Utility functions '''
    def obstacle_collision_check(self, curr_x, curr_y, next_x, next_y):
        whether_collision = False
        whether_vertical_line = False
        whether_horizontal_line = False

        if not (next_x - curr_x == 0.0):
            k = (next_y - curr_y) / (next_x - curr_x)
            b = next_y - k * next_x
            if k == 0.0:
                whether_horizontal_line = True
        else:
            whether_vertical_line = True

        for obstacle in self.obstacle_area:
            x_min = obstacle[0][0] # 1d list as [x, y]
            x_max = obstacle[1][0] # 1d list as [x, y]
            y_obst = obstacle[0][1]

            if whether_horizontal_line:
                if b == y_obst:
                    if (x_min >= min(curr_x, next_x) and x_min <= max(curr_x, next_x)) or \
                       (x_max >= min(curr_x, next_x) and x_max <= max(curr_x, next_x)):
                        whether_collision = True
            elif whether_vertical_line:
                if curr_x >= x_min and curr_x <= x_max:
                    if y_obst >= min(curr_y, next_y) and y_obst <= max(curr_y, next_y):
                        whether_collision = True
            else:
                x = (y_obst - b) / k
                if x >= min(curr_x, next_x) and x <= max(curr_x, next_x):
                    if x >= x_min and x <= x_max:
                        whether_collision = True

        if self.x <= 0.0 or self.x >= 20.0 or \
           self.y <= 0.0 or self.y >= 20.0:
            whether_collision = True

        if self.wall_thickness > 0:
            for obstacle in self.obstacle_area:
                x_min = obstacle[0][0]  # 1d list as [x, y]
                x_max = obstacle[1][0]  # 1d list as [x, y]
                y_min = obstacle[0][1]
                y_max = y_min + self.wall_thickness

                if (curr_x >= x_min and curr_x <= x_max and curr_y >= y_min and curr_y <= y_max) or \
                    (next_x >= x_min and next_x<= x_max and next_y >= y_min and next_y <= y_max):
                    whether_collision = True


        return whether_collision

    def obstacle_collision_check_new(self, curr_x, curr_y, next_x, next_y):
        whether_collision = False
        whether_vertical_line = False
        whether_horizontal_line = False

        if not (next_x - curr_x == 0.0):
            k = (next_y - curr_y) / (next_x - curr_x)
            b = next_y - k * next_x
            if k == 0.0:
                whether_horizontal_line = True
        else:
            whether_vertical_line = True

        for obstacle in self.obstacle_area:
            x_min = obstacle[0][0] - self.wall_thickness # 1d list as [x, y]
            x_max = obstacle[1][0] + self.wall_thickness# 1d list as [x, y]
            y_obst = obstacle[0][1]
            y_min = y_obst - self.wall_thickness
            y_max = y_obst + self.wall_thickness

            if (curr_x >= x_min and curr_x <= x_max and curr_y >= y_min and curr_y <= y_max) or \
                (next_x >= x_min and next_x<= x_max and next_y >= y_min and next_y <= y_max):
                whether_collision = True
                return whether_collision

            if whether_horizontal_line:
                if b >= y_min and b <= y_max:
                    if not (max(curr_x, next_x) < x_min or min(curr_x, next_x) > x_max):
                        whether_collision = True
                        return whether_collision
            elif whether_vertical_line:
                if curr_x >= x_min and curr_x <= x_max:
                    if (y_min >= min(curr_y, next_y) and y_min <= max(curr_y, next_y)) or \
                       (y_max >= min(curr_y, next_y) and y_max <= max(curr_y, next_y)):
                        whether_collision = True
                        return whether_collision
            else:
                x1 = (y_min - b) / k
                x2 = (y_max - b) / k
                if x1 >= min(curr_x, next_x) and x1 <= max(curr_x, next_x):
                    if x1 >= x_min and x1 <= x_max:
                        whether_collision = True
                        return whether_collision

                if x2 >= min(curr_x, next_x) and x2 <= max(curr_x, next_x):
                    if x2 >= x_min and x2 <= x_max:
                        whether_collision = True
                        return whether_collision

        if self.x <= 0.0 or self.x >= 20.0 or \
           self.y <= 0.0 or self.y >= 20.0:
            whether_collision = True
            return whether_collision

        return whether_collision



if __name__ == '__main__':
    env = NavEnv(render=False)
    check_env(env, warn=True)