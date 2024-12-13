import numpy as np
import gym

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data.types import Trajectory
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.algorithms.adversarial.gail import GAIL
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from imitation.data import rollout
from imitation.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper



TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())


class EvaluationLogger():
    def __init__(self, writer, gail_trainer, test_env):
        self.writer = writer
        self.gail_trainer = gail_trainer
        self.test_env = test_env

        self.before_training_evaluation()

    def before_training_evaluation(self):
        env_step = 0
        average_episode_reward = self.evaluate_GAIL_policy()

        self.writer.add_scalar(tag='episode_reward/train/per_env_step', scalar_value=average_episode_reward,
                               global_step=env_step)
        print("[{} environment steps finished]: Average episode reward is {}".format(env_step,
                                                                                     average_episode_reward))

    def evaluate_GAIL_policy(self):
        total_test_num = 20
        total_reward = 0.0
        for i in range(total_test_num):
            obs = self.test_env.reset()
            done = False
            episode_reward = 0.0
            while not done:
                action, _ = self.gail_trainer.policy.predict(obs, deterministic=True)
                next_obs, reward, done, info = self.test_env.step(action)
                episode_reward += reward

                obs = next_obs

            total_reward += episode_reward

        average_episode_reward = total_reward / total_test_num

        return average_episode_reward

    def callback(self, round_num):
        env_step = self.gail_trainer.gen_train_timesteps * (round_num + 1)

        print("round num: {}".format(round_num))
        print("env step: {}".format(env_step))
        print("global step of gail trainer: {}".format(self.gail_trainer._global_step))

        average_episode_reward = self.evaluate_GAIL_policy()

        self.writer.add_scalar(tag='episode_reward/train/per_env_step', scalar_value=average_episode_reward,global_step=env_step)
        print("[{} environment steps finished]: Average episode reward is {}".format(env_step,
                                                                                     average_episode_reward))


def evaluate_policy(model, env):
    total_test_num = 20
    total_reward = 0.0
    for i in range(total_test_num):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action, _= model.predict(obs)
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward

            obs = next_obs

        total_reward += episode_reward

        print("[Test episode {}]: episode reward is {}".format(i + 1, episode_reward))

    print("average episode reward: {}".format(total_reward / total_test_num))


def generate_demonstrations(env, model):
    trajs_list = []
    trajs_num = 60

    for traj_id in range(trajs_num):
        states = []
        actions = []
        state = env.reset()

        done = False
        episode_reward = 0.0
        while not done:
            states.append(state)
            action, _= model.predict(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            actions.append(action)

            state = next_state

        states.append(state)

        states = np.array(states)
        actions = np.array(actions)
        traj = Trajectory(obs=states, acts=actions, infos=None, terminal=True)
        trajs_list.append(traj)

        print("[GAIL]: demo {} is collected, final reward is {}".format(traj_id + 1, episode_reward))

    return trajs_list


def main():
    # train expert policy
    env = gym.make('Pendulum-v1')
    # env = gym.make('CartPole-v0')
    test_env = gym.make('Pendulum-v1')
    # test_env = gym.make('CartPole-v0')
    venv = DummyVecEnv([lambda: gym.make('Pendulum-v1')])
    # venv = DummyVecEnv([lambda: gym.make('CartPole-v0')])

    model = PPO("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=int(400*1e3))
    # model.learn(total_timesteps=int(50 * 1e3))

    # model.save("ppo_cartpole")
    # del model  # remove to demonstrate saving and loading
    # model = PPO.load("ppo_cartpole")
    model = PPO.load("ppo_pendulum")

    # evaluate expert performance
    evaluate_policy(model=model, env=env)

    # collect demo
    trajectories = generate_demonstrations(env=env, model=model)
    print("Finished generating expert demonstrations")
    print("***********************************")

    ''' Train learner policy with GAIL '''
    learner = PPO(
        env=venv,
        policy=MlpPolicy,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0004,
        gamma=0.95,
        n_epochs=5,
    )
    reward_net = BasicRewardNet(
        venv.observation_space, venv.action_space
    )
    gail_trainer = GAIL(
        demonstrations=trajectories,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=8,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True
    )

    print("Going to train learner policy via GAIL ... ")
    writer = SummaryWriter('logs/new/' + 'gail_test' + '/' + TIMESTAMP)
    gail_logger = EvaluationLogger(writer=writer, gail_trainer=gail_trainer, test_env=test_env)

    # max_env_steps = int(10 * 1e5)
    max_env_steps = 800000
    gail_trainer.train(max_env_steps, callback=gail_logger.callback)


if __name__ == "__main__":
    main()




