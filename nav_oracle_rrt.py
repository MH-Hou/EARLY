import numpy as np
from scipy import interpolate
import math
from time import sleep

import sys
# sys.path.append('/home/ullrich/catkin_ws/src/rrt-algorithms')
sys.path.append('/home/oem/catkin_ws/src/rrt-algorithms')
import src

from src.rrt.rrt_star import RRTStar
from src.rrt.rrt_star_bid import RRTStarBidirectional
from src.search_space.search_space import SearchSpace
from src.utilities.plotting import Plot
from nav_env import NavEnv


class RRT_Oracle():
    def __init__(self, wall_thickness=0.0):
        self.X_dimensions = np.array([(0, 20), (0, 20)])  # dimensions of Search Space
        self.wall_thickness =wall_thickness

        # obstacles
        if self.wall_thickness == 0.0:
            self.obstacles = np.array([(5, 10, 13, 12), (15, 10, 17, 12)])
        else:
            # self.obstacles = np.array([(5, 10, 13, 10 + wall_thickness), (15, 10, 17, 10 + wall_thickness)])
            self.obstacles = np.array([(5 - wall_thickness, 10 - wall_thickness, 13 + wall_thickness, 10 + wall_thickness),
                                       (15 - wall_thickness, 10 - wall_thickness, 17 + wall_thickness, 10 + wall_thickness)])

        self.Q = np.array([(1, 1)])  # length of tree edges
        self.r = 1  # length of smallest edge to check for intersection with obstacles
        self.max_samples = 2000  # max number of samples to take before timing out
        self.rewire_count = 32  # optional, number of nearby branches to rewire
        self.prc = 0.1  # probability of checking for a connection to goal

        # create Search Space
        self.X = SearchSpace(self.X_dimensions, self.obstacles)

    def path_planning(self, pos_init, pos_goal=(10, 16), bidirectional=False):
        while True:
            if bidirectional:
                rrt = RRTStarBidirectional(self.X, self.Q, pos_init, pos_goal,
                                           self.max_samples, self.r, self.prc, self.rewire_count)
                path = rrt.rrt_star_bidirectional()
            else:
                rrt = RRTStar(self.X, self.Q, pos_init, pos_goal,
                              self.max_samples, self.r, self.prc, self.rewire_count)
                path = rrt.rrt_star()

            interpolated_path = self.path_interpolate(path=path)

            oracle_states, oracle_actions = self.recover_demo(path=interpolated_path,
                                                              delta_t=1.0,
                                                              pos_goal=pos_goal)

            whether_collide = self.collision_check(oracle_states=oracle_states,
                                                   oracle_actions=oracle_actions,
                                                   pos_goal=pos_goal)
            if whether_collide:
                print("Planned path failed in nav env! Going to plan again...")
                pass
            else:
                # print("Planned path is successful")
                break

        return interpolated_path

    def path_interpolate(self, path):
        step_max_size = 1.0
        interpolated_path = None # will be a 2d np array in the form of (num_of_points, pos_dimension)
        total_waypoints_num = len(path)
        for i in range(total_waypoints_num - 1):
            start = np.array(path[i]) # 1d np array of (x, y)
            end = np.array(path[i + 1]) # 1d np array of (x, y)
            points_to_interpolate = math.ceil(np.linalg.norm(end - start) / step_max_size) + 1 # including start and end points

            # when the line is not vertical
            if not (start[0] - end[0] == 0.0):
                f = interpolate.interp1d(x=[start[0], end[0]],
                                         y=[start[1], end[1]])

                # 1d np array, and exclude the start and end points
                interpolated_xs = np.linspace(start=start[0], stop=end[0], num=points_to_interpolate)[1:-1]
                interpolated_ys = f(interpolated_xs)
            else:
                interpolated_xs = np.ones(points_to_interpolate)[1:-1]
                interpolated_ys = np.linspace(start=start[1], stop=end[1], num=points_to_interpolate)[1:-1]


            # 2d np array in the form of (num_of_interpolated_points, pos_dimension)
            interpolated_points = np.stack((interpolated_xs, interpolated_ys), axis=1)

            if interpolated_path is None:
                # 2d np array in the form of (num_of_points, pos_dimension)
                interpolated_path = np.concatenate((np.array([start]),
                                                    interpolated_points,
                                                    np.array([end])),
                                                   axis=0)
            else:
                interpolated_path = np.concatenate((interpolated_path,
                                                    np.array([start]),
                                                    interpolated_points,
                                                    np.array([end])),
                                                   axis=0)
        return interpolated_path

    def recover_demo(self, path, delta_t=1.0, pos_goal=(10, 16)):
        states = [] # will be a 2d np array in the form of (num_of_points, state_dimension)
        actions = [] # will be a 2d np array in the form of (num_of_points, action_dimension)
        total_points_num = np.shape(path)[0]

        for i in range(total_points_num):
            state = np.array([pos_goal[0], pos_goal[1], path[i][0], path[i][1]]) # 1d np array as (state_dim,)
            states.append(state)

            if i < (total_points_num - 1):
                vx = (path[i + 1][0] - path[i][0]) / delta_t
                vy = (path[i + 1][1] - path[i][1]) / delta_t
            else:
                vx = 0.0
                vy = 0.0
            action = np.array([vx, vy])
            actions.append(action)

        states = np.array(states)
        actions = np.array(actions)

        return states, actions

    def collision_check(self, oracle_states, oracle_actions, pos_goal=(10, 16)):
        nav_env = NavEnv(render=False, wall_thickness=self.wall_thickness)
        pos_init = oracle_states[0][2:4]
        obs = nav_env.reset(random=False, initial_x=pos_init[0], initial_y=pos_init[1], random_goal=False, goal_x=pos_goal[0], goal_y=pos_goal[1])
        done = False
        step = 0
        reward = 0.0
        while not done:
            action = oracle_actions[step]
            obs, reward, done, _ = nav_env.step(action=action)
            step += 1

        # pos_goal = np.array([nav_env.target_x, nav_env.target_y])
        # pos_terminate = np.array([nav_env.x, nav_env.y])
        # if np.linalg.norm(pos_goal - pos_terminate) <= nav_env.arrival_thres:
        if reward == 1000.0:
            whether_collide = False
        else:
            whether_collide = True

        # print("Goal pos: {}".format(pos_goal))
        # print("Terminate pos: {}".format(pos_terminate))
        # print("distance: {}".format(np.linalg.norm(pos_goal - pos_terminate)))

        return whether_collide


def plot_path(rrt_oracle, path, pos_init, pos_goal):
    # plot
    plot = Plot("rrt_star_2d")
    # plot.plot_tree(X, rrt.trees)
    if path is not None:
        plot.plot_path(rrt_oracle.X, path)
    plot.plot_obstacles(rrt_oracle.X, rrt_oracle.obstacles)
    plot.plot_start(rrt_oracle.X, pos_init)
    plot.plot_goal(rrt_oracle.X, pos_goal)
    plot.draw(auto_open=True)


def rrt_oracle_test():
    env = NavEnv(render=True)
    rrt_oracle = RRT_Oracle()

    total_test_episodes = 20
    for i in range(total_test_episodes):
        obs = env.reset(random=True)

        pos_init = tuple(obs[2:4])
        pos_goal = tuple(obs[0:2])
        path = rrt_oracle.path_planning(pos_init=pos_init,
                                        pos_goal=pos_goal)
        _ , oracle_actions = rrt_oracle.recover_demo(path=path,
                                                 delta_t=1.0,
                                                 pos_goal=pos_goal)

        env.render()
        done = False
        episode_rewards = 0.0
        step = 0

        while not done:
            action = oracle_actions[step]
            obs, reward, done, _ = env.step(action=action)
            env.render()
            episode_rewards += reward

            step += 1
            sleep(0.05)

        print("[Episode {}]: Finished! Episode reward is {}".format(i + 1, episode_rewards))
        print("****************************")

    env.close()
    print("All episodes finished!")


def main():
    rrt_oracle = RRT_Oracle()

    pos_init = (np.random.default_rng().uniform(0.01, 19.99, 1)[0], 4)  # starting location
    pos_goal = (10, 16)  # goal location

    path = rrt_oracle.path_planning(pos_init=pos_init,
                                    pos_goal=pos_goal)

    plot_path(rrt_oracle=rrt_oracle,
              path=path,
              pos_init=pos_init,
              pos_goal=pos_goal)



if __name__ == '__main__':
    # main()
    rrt_oracle_test()