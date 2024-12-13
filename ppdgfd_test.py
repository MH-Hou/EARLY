import gym

from stable_baselines3 import SAC

env = gym.make("Pendulum-v1")

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000, log_interval=4)
model.save("sac_pendulum")

del model # remove to demonstrate saving and loading

model = SAC.load("sac_pendulum")

total_test_rewards = 0.0
total_test_episodes = 20

for i in range(total_test_episodes):
    done = False
    episode_reward = 0.0
    obs = env.reset()
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

    print("[Episode {}]: total reward is {}".format(i+1, episode_reward))
    total_test_rewards+= episode_reward

print("All finished! Average episode reward is {}".format(total_test_rewards/total_test_episodes))
