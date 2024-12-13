import gym
import random
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO, SAC
from stable_baselines3.ppo import MlpPolicy
import seals  # needed to load environments
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
import numpy as np
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.data.types import Trajectory
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback
import os

from nav_env import NavEnv
from nav_oracle_rrt import RRT_Oracle
from activesac_nav import argparser

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())


def evaluate_policy_(test_env, policy, evaluation_res, env_step, test_episode_num=10, total_seed_num=10, random_goal_task=False, random_y=False, success_evaluation_res=None):
    test_episode_reward = 0.0
    total_success_num = 0.0
    # total_seed_num = 10
    res = []
    success_res = []
    res.append(env_step)
    success_res.append(env_step)
    for seed in range(total_seed_num):
        # np.random.seed(seed)
        sampled_xs = np.random.default_rng(seed).uniform(low=0.01, high=19.9, size=test_episode_num)
        if random_goal_task:
            sampled_goal_xs = np.random.default_rng(seed).uniform(low=0.01, high=19.9, size=test_episode_num)
            if random_y:
                sampled_ys = np.random.default_rng(seed).uniform(low=1.0, high=8.0, size=test_episode_num)
                sampled_goal_ys = np.random.default_rng(seed).uniform(low=12.0, high=19.0, size=test_episode_num)
        for i in range(test_episode_num):
            if random_goal_task:
                if random_y:
                    state_ = test_env.reset(random=False, initial_x=sampled_xs[i], initial_y=sampled_ys[i],
                                            random_goal=False,
                                            goal_x=sampled_goal_xs[i], goal_y=sampled_goal_ys[i], random_y=False)
                else:
                    state_ = test_env.reset(random=False, initial_x=sampled_xs[i], initial_y=4.0, random_goal=False,
                                            goal_x=sampled_goal_xs[i], goal_y=16.0, random_y=False)
            else:
                state_ = test_env.reset(random=False, initial_x=sampled_xs[i], initial_y=4.0)
            done_ = False
            episode_reward = 0.0
            whether_success = 0.0
            while not done_:
                action_, _ = policy.predict(state_, deterministic=True)
                next_state_, reward_, done_, _ = test_env.step(action_.numpy())
                episode_reward += reward_
                test_episode_reward += reward_
                state_ = next_state_

                if reward_ == 1000.0:
                    whether_success = 1.0
                    total_success_num += 1.0

            res.append(episode_reward)
            success_res.append(whether_success)

    evaluation_res.append(res)
    average_episode_reward = test_episode_reward / (test_episode_num * total_seed_num)

    success_evaluation_res.append(success_res)
    average_success_rate = total_success_num / (test_episode_num * total_seed_num)

    return average_episode_reward, evaluation_res, average_success_rate, success_evaluation_res


def evaluate_policy_new(test_env, policy, test_episode_num=10, total_seed_num=10):
    test_episode_reward = 0.0
    total_success_num = 0.0
    for seed in range(total_seed_num):
        sampled_xs = np.random.default_rng(seed).uniform(low=0.01, high=19.9, size=test_episode_num)
        for i in range(test_episode_num):
            state_ = test_env.reset(random=False, initial_x=sampled_xs[i], initial_y=4.0)
            done_ = False
            while not done_:
                action_, _ = policy.predict(state_, deterministic=True)
                next_state_, reward_, done_, _ = test_env.step(action_)
                test_episode_reward += reward_
                state_ = next_state_

    average_episode_reward = test_episode_reward / (test_episode_num * total_seed_num)

    return average_episode_reward


class CustomEvaluateCallback(BaseCallback):
    def __init__(self, task_name, method, test_env, plot_interval=1000, verbose=0):
        super(CustomEvaluateCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

        self.writer = SummaryWriter('logs/' + task_name + '/' + method + '/' + TIMESTAMP)
        self.test_env = test_env
        self.plot_interval = plot_interval

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        if (self.num_timesteps -1) % self.plot_interval == 0:
            average_episode_reward = evaluate_policy_new(test_env=self.test_env,
                                                            policy=self.model,
                                                            test_episode_num=10,
                                                            total_seed_num=10)

            self.writer.add_scalar(tag='episode_reward/train/per_env_step', scalar_value=average_episode_reward,
                                    global_step=self.num_timesteps)

        return True


class EvaluationLogger():
    def __init__(self, writer, gail_trainer, test_env, task_name, method, max_demo_num, random_goal, random_y):
        self.writer = writer
        self.gail_trainer = gail_trainer
        self.test_env = test_env
        self.task_name = task_name
        self.method = method
        self.max_demo_num = max_demo_num
        self.random_goal = random_goal
        self.random_y = random_y

        self.evaluate_res_per_step = []
        self.success_evaluate_res_per_step = []

        self.before_training_evaluation()

    def before_training_evaluation(self):
        env_step = 0
        average_episode_reward, average_success_rate = self.evaluate_GAIL_policy(env_step=env_step,
                                                                                 test_episode_num=10,
                                                                                 total_seed_num=10)

        self.writer.add_scalar(tag='episode_reward/train/per_env_step', scalar_value=average_episode_reward,
                               global_step=env_step)
        print("[{} environment steps finished]: Average episode reward is {}".format(env_step,
                                                                                     average_episode_reward))

    def evaluate_GAIL_policy(self, env_step, test_episode_num=10, total_seed_num=10):
        test_episode_reward = 0.0
        total_success_num = 0.0
        # total_seed_num = 10
        res = []
        success_res = []
        res.append(env_step)
        success_res.append(env_step)
        for seed in range(total_seed_num):
            # np.random.seed(seed)
            sampled_xs = np.random.default_rng(seed).uniform(low=0.01, high=19.9, size=test_episode_num)
            if self.random_goal:
                sampled_goal_xs = np.random.default_rng(seed).uniform(low=0.01, high=19.9, size=test_episode_num)
                if self.random_y:
                    sampled_ys = np.random.default_rng(seed).uniform(low=1.0, high=8.0, size=test_episode_num)
                    sampled_goal_ys = np.random.default_rng(seed).uniform(low=12.0, high=19.0, size=test_episode_num)
            for i in range(test_episode_num):
                if self.random_goal:
                    if self.random_y:
                        state_ = self.test_env.reset(random=False, initial_x=sampled_xs[i], initial_y=sampled_ys[i],
                                                random_goal=False,
                                                goal_x=sampled_goal_xs[i], goal_y=sampled_goal_ys[i], random_y=False)
                    else:
                        state_ = self.test_env.reset(random=False, initial_x=sampled_xs[i], initial_y=4.0, random_goal=False,
                                                goal_x=sampled_goal_xs[i], goal_y=16.0, random_y=False)
                else:
                    state_ = self.test_env.reset(random=False, initial_x=sampled_xs[i], initial_y=4.0)
                done_ = False
                episode_reward = 0.0
                whether_success = 0.0
                while not done_:
                    action_, _ = self.gail_trainer.policy.predict(state_, deterministic=True)
                    next_state_, reward_, done_, _ = self.test_env.step(action_)
                    episode_reward += reward_
                    test_episode_reward += reward_
                    state_ = next_state_

                    if reward_ == 1000.0:
                        whether_success = 1.0
                        total_success_num += 1.0

                res.append(episode_reward)
                success_res.append(whether_success)

        self.evaluate_res_per_step.append(res)
        average_episode_reward = test_episode_reward / (test_episode_num * total_seed_num)

        self.success_evaluate_res_per_step.append(success_res)
        average_success_rate = total_success_num / (test_episode_num * total_seed_num)

        return average_episode_reward, average_success_rate

    def callback(self, round_num):
        env_step = self.gail_trainer.gen_train_timesteps * (round_num + 1)

        print("round num: {}".format(round_num))
        print("env step: {}".format(env_step))
        print("global step of gail trainer: {}".format(self.gail_trainer._global_step))

        average_episode_reward, average_success_rate = self.evaluate_GAIL_policy(env_step=env_step ,
                                                                                 test_episode_num=10,
                                                                                 total_seed_num=10)

        self.writer.add_scalar(tag='episode_reward/train/per_env_step', scalar_value=average_episode_reward,global_step=env_step)
        print("[{} environment steps finished]: Average episode reward is {}".format(env_step,
                                                                                     average_episode_reward))

    def save_results(self):
        res_per_step_path = 'evaluation_res/new/' + self.task_name + '/' + self.method + '/max_demo_' + str(
            self.max_demo_num) + '/'
        if not os.path.exists(res_per_step_path):
            os.makedirs(res_per_step_path)

        np.savetxt(res_per_step_path + 'res_per_step_new.csv', self.evaluate_res_per_step, delimiter=' ')
        np.savetxt(res_per_step_path + 'success_res_per_step_new.csv', self.success_evaluate_res_per_step, delimiter=' ')

        print("[Evaluation Logger]: All results are saved!")

def vectorize_nav_env(env_num=8, max_episode_steps=200):
    def _make_env():
        """Helper function to create a single environment. Put any logic here, but make sure to return a RolloutInfoWrapper."""
        _env = NavEnv(render=False)
        _env = TimeLimit(_env, max_episode_steps=max_episode_steps)
        _env = RolloutInfoWrapper(_env)
        return _env

    venv = SubprocVecEnv([_make_env for _ in range(env_num)])

    return venv


def generate_demonstrations_rrt(env, trajs_num, random_goal, random_y):
    # a list of Trajectory instances
    trajs_list = []
    rrt_oracle = RRT_Oracle()

    for traj_id in range(trajs_num):
        states = []
        actions = []
        state = env.reset(random=True, random_goal=random_goal, random_y=random_y)
        # print("sampled initial state: {}".format(state))

        pos_goal = (state[0], state[1])
        pos_init = (state[2], state[3])
        path = rrt_oracle.path_planning(pos_init=pos_init,
                                        pos_goal=pos_goal)

        # 2d np array of (total_step_num, action/state_dimension)
        oracle_states, oracle_actions = rrt_oracle.recover_demo(path=path,
                                                                delta_t=1.0,
                                                                pos_goal=pos_goal)

        done = False
        step = 0
        while not done:
            states.append(state)

            action = oracle_actions[step]
            next_state, reward, done, _ = env.step(action)

            actions.append(action)

            state = next_state
            step += 1

        states.append(state)

        states = np.array(states)
        actions = np.array(actions)
        traj = Trajectory(obs=states, acts=actions, infos=None, terminal=True)
        trajs_list.append(traj)

        print("[GAIL]: demo {} is collected, final reward is {}".format(traj_id + 1, reward))

    return trajs_list



def generate_demonstrations(env, policy, trajs_num):
    # a list of Trajectory instances
    trajs_list = []

    for traj_id in range(trajs_num):
        states = []
        actions = []
        obs = env.reset()
        done = False

        while not done:
            # collect s_t
            states.append(obs)

            act, _ = policy.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action=act)

            # collect a_t
            actions.append(act)

        # collect the terminal state
        states.append(obs)

        states = np.array(states)
        actions = np.array(actions)
        traj = Trajectory(obs=states, acts=actions, infos=None, terminal=True)
        trajs_list.append(traj)

    return trajs_list


def evaluate_gail_policy(test_env, gail_policy, evaluation_res, env_step, test_episode_num=10):
    res = []
    res.append(env_step)
    test_episode_reward = 0.0
    for _ in range(test_episode_num):
        state_ = test_env.reset()
        done_ = False
        episode_reward = 0.0
        while not done_:
            action_, _ = gail_policy.predict(state_, deterministic=True)
            next_state_, reward_, done_, _ = test_env.step(action_)
            episode_reward += reward_
            test_episode_reward += reward_
            state_ = next_state_

        res.append(episode_reward)

    evaluation_res.append(res)
    average_episode_reward = test_episode_reward / test_episode_num

    return average_episode_reward, evaluation_res


def train_and_eval_sac():
    env = NavEnv(render=False)
    test_env = NavEnv(render=False)
    max_env_steps = int(5 * 1e4)
    custom_evaluate_callback = CustomEvaluateCallback(task_name='nav',
                                                      method='pure_sac',
                                                      test_env=test_env)

    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=max_env_steps, log_interval=4, callback=custom_evaluate_callback)




def main():
    args = argparser()
    seed = 6
    random.seed(seed)
    np.random.seed(seed)
    max_demo_num = 60

    """ Train expert policy """
    task_name = args.task_name
    if task_name == 'nav_1':
        random_goal = False
        random_y = False
        max_env_steps = int(10 * 1e4) + 2048
        ratio = 0.35
    elif task_name == 'nav_2':
        random_goal = True
        random_y = False
        max_env_steps = int(10 * 1e4) + 2048
        ratio = 0.3
    else:
        random_goal = True
        random_y = True
        max_env_steps = int(15 * 1e4) + 2048
        ratio = 0.4

    demo_env = NavEnv(render=False)
    test_env = NavEnv(render=False)
    venv = DummyVecEnv([lambda: NavEnv(render=False, task_name=task_name)])

    # expert = PPO(
    #     policy=MlpPolicy,
    #     env=venv,
    #     seed=0,
    #     batch_size=64,
    #     ent_coef=0.0,
    #     learning_rate=0.0003,
    #     n_epochs=10,
    #     n_steps=64,
    # )
    # print("Going to training expert policy ... ")
    # expert.learn(200*10000)  # Note: set to 100000 to train a proficient expert
    # print("Finished training expert policy")
    # print("********************************")

    ''' Generate expert demonstrations '''
    # rng = np.random.default_rng()
    # rollouts = rollout.rollout(
    #     expert,
    #     make_vec_env(
    #         "seals/CartPole-v0",
    #         n_envs=5,
    #         post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
    #         rng=rng,
    #     ),
    #     rollout.make_sample_until(min_timesteps=None, min_episodes=60),
    #     rng=rng,
    # )
    #
    # print(type(rollouts[0]))

    # trajectories = generate_demonstrations(env=env, policy=expert, trajs_num=60)
    trajectories = generate_demonstrations_rrt(env=demo_env, trajs_num=max_demo_num, random_goal=random_goal, random_y=random_y)
    print("Finished generating expert demonstrations")
    print("***********************************")

    ''' Train learner policy with GAIL '''
    # venv = make_vec_env("seals/CartPole-v0", n_envs=8, rng=rng)
    # venv = vectorize_nav_env(env_num=10, max_episode_steps=200)
    learner = PPO(
        env=venv,
        policy=MlpPolicy,
        batch_size=128,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
    )
    reward_net = BasicRewardNet(
        venv.observation_space, venv.action_space
    )
    gail_trainer = GAIL(
        demonstrations=trajectories,
        demo_batch_size=128,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=1,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True
    )

    ''' Evaluate learner policy before and after training '''
    # learner_rewards_before_training, _ = evaluate_policy(
    #     learner, venv, 100, return_episode_rewards=True
    # )
    print("Going to train learner policy via GAIL ... ")
    method = 'gail'
    writer = SummaryWriter('logs/new/' + task_name + '/' + method + '/' + TIMESTAMP)
    gail_logger = EvaluationLogger(writer=writer, gail_trainer=gail_trainer, test_env=test_env,
                                   task_name=task_name, method=method, max_demo_num=max_demo_num,
                                   random_goal=random_goal, random_y=random_y)

    def callback(round_num: int, /) -> None:
        if round_num % 10 == 0:
            print("round num: {}".format(round_num))
            average_episode_r = evaluate_policy_new(test_env=test_env,
                                                    policy=gail_trainer.policy,
                                                    test_episode_num=10,
                                                    total_seed_num=10)

            writer.add_scalar(tag='episode_reward/train/per_env_step', scalar_value=average_episode_r,
                                   global_step=round_num)

    # gail_trainer.train(max_env_steps, callback=callback)
    gail_trainer.train(max_env_steps, callback=gail_logger.callback)
    gail_logger.save_results()


    # while env_step < max_env_steps:
    #     gail_trainer.train(2500)
    #     env_step += 2500
    #
    #     average_episode_reward, evaluate_res_per_step = evaluate_gail_policy(test_env=test_env,
    #                                                                          gail_policy=learner,
    #                                                                          evaluation_res=evaluate_res_per_step,
    #                                                                          env_step=env_step,
    #                                                                          test_episode_num=100)
    #     writer.add_scalar(tag='episode_reward/train/per_env_step', scalar_value=average_episode_reward,
    #                       global_step=env_step)
    #     print("[{} environment steps finished]: Average episode reward is {}".format(env_step,
    #                                                                                  average_episode_reward))
    # # gail_trainer.train(600000)  # Note: set to 300000 for better results
    # print("Finished GAIL training")
    # print("*************************")
    #
    # np.savetxt('evaluation_res/' + method + '/res_per_step.csv', evaluate_res_per_step, delimiter=' ')


    # learner_rewards_after_training, _ = evaluate_policy(
    #     learner, venv, 100, return_episode_rewards=True
    # )
    # expert_rewards_after_training, _ = evaluate_policy(
    #     expert, venv, 100, return_episode_rewards=True
    # )

    # print("Going to show the evaluation results ... ")
    # print('mean rewards of expert:')
    # print(np.mean(expert_rewards_after_training))
    # print("mean rewards before GAIL training")
    # print(np.mean(learner_rewards_before_training))
    # print("mean rewards after GAIL training:")
    # print(np.mean(learner_rewards_after_training))
    #
    #
    # plt.hist(
    #     [learner_rewards_before_training, learner_rewards_after_training, expert_rewards_after_training],
    #     label=["untrained", "trained", "expert"],
    # )
    # plt.legend()
    # plt.show()



if __name__ == '__main__':
    main()
    # venv = vectorize_nav_env(env_num=10, max_episode_steps=200)
    # train_and_eval_sac()