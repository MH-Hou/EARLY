import numpy as np
from stable_baselines3 import PPO
from datetime import datetime

from nav_env import NavEnv

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())


def main():
    # create gym environment
    env = NavEnv(render=False)

    total_episode_num = int(1e4)
    steps_per_episode = 200
    # nav_oracle_policy = PPO("MlpPolicy", env, verbose=1,
    #                         tensorboard_log="./tensorboard/",
    #                         gamma=1.0)

    # start to learn
    nav_oracle_policy = PPO.load("rrt_oracle/bc_wo_ppo2023-06-14T15-33-13")
    nav_oracle_policy.set_env(env)
    nav_oracle_policy.learn(total_timesteps=total_episode_num * steps_per_episode)
    nav_oracle_policy.save("rrt_oracle/bc_ppo_" + TIMESTAMP)
    print("Training finished!")



if __name__ == '__main__':
    main()