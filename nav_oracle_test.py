import numpy as np
from stable_baselines3 import PPO
from time import sleep

from nav_env import NavEnv



def main():
    env = NavEnv(render=False, target_pos=(10, 16))
    oracle_policy = PPO.load("rrt_oracle/clean_bc2023-08-10T13-34-17", env=env)
    # oracle_policy = PPO.load('rrt_oracle/bc_wo_ppo_init_uniform2023-06-15T14-38-00', env=env)
    # oracle_policy = PPO.load('rrt_oracle/bc_wo_ppo_mode_uniform2023-06-15T14-38-00', env=env)

    total_test_episodes = 100
    total_success_episodes = 0
    all_episodes_rewards = 0.0
    init_xs = np.linspace(start=0.01, stop=19.99, num=total_test_episodes)
    for i in range(total_test_episodes):
        init_x = init_xs[i]
        init_y = 4.0
        # obs = env.reset(random=False, initial_x=init_x, initial_y=init_y)
        obs = env.reset(random=True)
        env.render()
        done = False
        episode_rewards = 0.0

        while not done:
            action, _ = oracle_policy.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action=action)
            env.render()
            episode_rewards += reward

            sleep(0.05)

        if reward == 1000.0:
            total_success_episodes += 1

        all_episodes_rewards += episode_rewards

        print("[Episode {}]: Finished! Episode reward is {}".format(i + 1, episode_rewards))
        print("****************************")

    env.close()
    print("All episodes finished!")
    print("Success rate: {} success out of {} test episodes, ratio is {}".format(total_success_episodes,
                                                                                 total_test_episodes,
                                                                                 1.0*total_success_episodes/total_test_episodes))
    print("average episode rewards: {}".format(all_episodes_rewards / total_test_episodes))


if __name__ == '__main__':
    main()





