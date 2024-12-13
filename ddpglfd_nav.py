import os
import copy
import random
from collections import deque
from typing import Deque, Dict, List, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from time import sleep
import argparse

from segment_tree import MinSegmentTree, SumSegmentTree

from nav_env import NavEnv
from nav_oracle_rrt import RRT_Oracle
from activesac_nav import query_strategy
from scripts.joystick import Joystick



TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# seed = 777
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)


class ReplayBuffer:
    """A numpy replay buffer with demonstrations."""

    def __init__(
            self,
            obs_dim: int,
            act_dim:int,
            size: int,
            batch_size: int = 32,
            gamma: float = 0.99,
            demo: list = None,
            n_step: int = 1,
    ):
        """Initialize."""
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size = 0, 0

        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

        # for demonstration
        self.demo_size = len(demo) if demo else 0
        self.demo = demo

        self.demo_step_ids = []

        if self.demo:
            self.ptr += self.demo_size
            self.size += self.demo_size
            for ptr, d in enumerate(self.demo):
                state, action, reward, next_state, done = d
                self.obs_buf[ptr] = state
                # self.acts_buf[ptr] = np.array(action)
                self.acts_buf[ptr] = action
                self.rews_buf[ptr] = reward
                self.next_obs_buf[ptr] = next_state
                self.done_buf[ptr] = done
                self.demo_step_ids.append(ptr)

    def store(
            self,
            obs: np.ndarray,
            act: np.ndarray,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
            role
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        """Store the transition in buffer."""
        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()

        # make a n-step transition
        rew, next_obs, done = self._get_n_step_info()
        obs, act = self.n_step_buffer[0][:2]

        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        if role == 'teacher':
            self.demo_step_ids.append(self.ptr)

        self.ptr += 1
        self.ptr = 0 if self.ptr % self.max_size == 0 else self.ptr
        while self.ptr in self.demo_step_ids:
            self.ptr += 1
            self.ptr = 0 if self.ptr % self.max_size == 0 else self.ptr

        # self.ptr += 1
        # self.ptr = self.demo_size if self.ptr % self.max_size == 0 else self.ptr
        self.size = min(self.size + 1, self.max_size)

        return self.n_step_buffer[0]

    def sample_batch(self, indices: List[int] = None) -> Dict[str, np.ndarray]:
        """Randomly sample a batch of experiences from memory."""
        assert len(self) >= self.batch_size

        if indices is None:
            indices = np.random.choice(
                len(self), size=self.batch_size, replace=False
            )

        return dict(
            obs=self.obs_buf[indices],
            next_obs=self.next_obs_buf[indices],
            acts=self.acts_buf[indices],
            rews=self.rews_buf[indices],
            done=self.done_buf[indices],
            # for N-step learning
            indices=indices,
        )

    def _get_n_step_info(self) -> Tuple[np.int64, np.ndarray, bool]:
        """Return n step rew, next_obs, and done."""
        # info of the last transition
        rew, next_obs, done = self.n_step_buffer[-1][-3:]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]

            rew = r + self.gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer with demonstrations."""

    def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            size: int,
            batch_size: int = 32,
            gamma: float = 0.99,
            alpha: float = 0.6,
            epsilon_d: float = 1.0,
            demo: list = None,
    ):
        """Initialize."""
        assert alpha >= 0

        super(PrioritizedReplayBuffer, self).__init__(
            obs_dim, act_dim, size, batch_size, gamma, demo, n_step=1
        )
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        self.epsilon_d = epsilon_d

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

        # for init priority of demo
        self.tree_ptr = self.demo_size
        for i in range(self.demo_size):
            self.sum_tree[i] = self.max_priority ** self.alpha
            self.min_tree[i] = self.max_priority ** self.alpha

    def store(
            self,
            obs: np.ndarray,
            act: np.ndarray,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
            role
    ):
        """Store experience and priority."""
        transition = super().store(obs, act, rew, next_obs, done, role)

        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha

            self.tree_ptr += 1
            if self.tree_ptr % self.max_size == 0:
                self.tree_ptr = self.demo_size

        return transition

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional()

        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        epsilon_d = np.array(
            [self.epsilon_d if i in self.demo_step_ids else 0.0 for i in indices]
        )
        # epsilon_d = np.array(
        #     [self.epsilon_d if i < self.demo_size else 0.0 for i in indices]
        # )

        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            epsilon_d=epsilon_d,
            indices=indices,
        )

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight


class OUNoise:
    """Ornstein-Uhlenbeck process.
    Taken from Udacity deep-reinforcement-learning github repository:
    https://github.com/udacity/deep-reinforcement-learning/blob/master/
    ddpg-pendulum/ddpg_agent.py
    """

    def __init__(
        self,
        size: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        seed=6
    ):
        """Initialize parameters and noise process."""
        self.state = np.float64(0.0)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self) -> np.ndarray:
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
            [random.random() for _ in range(len(x))]
        )
        self.state = x + dx
        return self.state


class Actor(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            init_w: float = 3e-3,
    ):
        """Initialize."""
        super(Actor, self).__init__()

        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, out_dim)

        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        action = self.out(x).tanh()

        return action



class Critic(nn.Module):
    def __init__(
            self,
            in_dim: int,
            init_w: float = 3e-3,
    ):
        """Initialize."""
        super(Critic, self).__init__()

        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)

        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(
            self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Forward method implementation."""
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        value = self.out(x)

        return value


class DDPGfDAgent:
    """DDPGfDAgent interacting with environment.

    Attribute:
        env (gym.Env): openAI Gym environment
        actor (nn.Module): target actor model to select actions
        actor_target (nn.Module): actor model to predict next actions
        actor_optimizer (Optimizer): optimizer for training actor
        critic (nn.Module): critic model to predict state values
        critic_target (nn.Module): target critic model to predict state values
        critic_optimizer (Optimizer): optimizer for training critic
        demo (list): demonstration
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        gamma (float): discount factor
        tau (float): parameter for soft target update
        initial_random_steps (int): initial random action steps
        pretrain_step (int): the number of step for pre-training
        n_step (int): the number of multi step
        use_n_step (bool): whether to use n_step memory
        prior_eps (float): guarantees every transitions can be sampled
        lambda1 (float): n-step return weight
        lambda2 (float): l2 regularization weight
        lambda3 (float): actor loss contribution of prior weight
        noise (OUNoise): noise generator for exploration
        device (torch.device): cpu / gpu
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
    """

    def __init__(
            self,
            env: gym.Env,
            memory_size: int,
            batch_size: int,
            ou_noise_theta: float,
            ou_noise_sigma: float,
            demo: list,
            pretrain_step: int,
            gamma: float = 0.99,
            tau: float = 5e-3,
            initial_random_steps: int = 1e4,
            # PER parameters
            alpha: float = 0.3,
            beta: float = 1.0,
            prior_eps: float = 1e-6,
            # N-step Learning
            n_step: int = 3,
            # loss parameters
            lambda1: float = 1.0,  # N-step return weight
            lambda2: float = 1e-4,  # l2 regularization weight
            lambda3: float = 1.0,  # actor loss contribution of prior weight
            writer=None,
            test_env=None,
            plot_internal=None,
            seed=6
    ):
        """Initialize."""
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        env.action_space.np_random.seed(seed)

        self.env = env
        self.batch_size = batch_size
        self.pretrain_step = pretrain_step
        self.gamma = gamma
        self.tau = tau
        self.initial_random_steps = initial_random_steps
        self.lambda1 = lambda1
        self.lambda3 = lambda3

        self.writer = writer
        self.test_env = test_env
        self.plot_interval = plot_internal

        self.demo = demo
        demos_1_step, demos_n_step = [], []
        if self.demo:
            demos_1_step, demos_n_step = self._get_n_step_info_from_demo(
                demo, n_step
            )

        # PER
        # memory for 1-step Learning
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(
            obs_dim, action_dim, memory_size, batch_size, gamma, alpha, demo=demos_1_step
        )

        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(
                obs_dim,
                action_dim,
                memory_size,
                batch_size,
                gamma,
                demos_n_step,
                self.n_step
            )

        # noise
        self.noise = OUNoise(
            action_dim,
            theta=ou_noise_theta,
            sigma=ou_noise_sigma,
            seed=seed
        )

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # self.device = torch.device("cpu")
        print(self.device)

        # networks
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizer
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=3e-4,
            weight_decay=lambda2,
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=1e-3,
            weight_decay=lambda2,
        )

        # transition to store in memory
        self.transition = list()

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # if initial random action should be conducted
        if self.total_step < self.initial_random_steps and not self.is_test:
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.actor(
                torch.FloatTensor(state).to(self.device)
            ).detach().cpu().numpy()

        # add noise for exploration during training
        if not self.is_test:
            noise = self.noise.sample()
            selected_action = np.clip(selected_action + noise, -1.0, 1.0)

        self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray, role) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done, _ = self.env.step(action)

        if not self.is_test:
            self.transition += [reward, next_state, done]

            # N-step transition
            transition = self.transition

            if self.use_n_step:
                self.transition.append(role)
                transition = self.memory_n.store(*self.transition)

            # add a single step transition
            if transition:
                transition = list(transition)
                transition.append(role)
                self.memory.store(*transition)

        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, ...]:
        """Update the model by gradient descent."""
        device = self.device  # for shortening the following lines

        samples = self.memory.sample_batch(self.beta)
        state = torch.FloatTensor(samples["obs"]).to(device)
        action = torch.FloatTensor(samples["acts"]).to(device)

        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(device)
        epsilon_d = samples["epsilon_d"]
        indices = samples["indices"]

        # train critic
        # 1-step loss
        critic_loss_element_wise = self._get_critic_loss(samples, self.gamma)
        critic_loss = torch.mean(critic_loss_element_wise * weights)

        # n-step loss
        if self.use_n_step:
            samples_n = self.memory_n.sample_batch(indices)
            n_gamma = self.gamma ** self.n_step
            critic_loss_n_element_wise = self._get_critic_loss(
                samples_n, n_gamma
            )

            # to update loss and priorities
            critic_loss_element_wise += (
                    critic_loss_n_element_wise * self.lambda1
            )
            critic_loss = torch.mean(critic_loss_element_wise * weights)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # train actor
        actor_loss_element_wise = -self.critic(state, self.actor(state))
        actor_loss = torch.mean(actor_loss_element_wise * weights)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # target update
        self._target_soft_update()

        # PER: update priorities
        new_priorities = critic_loss_element_wise
        new_priorities += self.lambda3 * actor_loss_element_wise.pow(2)
        new_priorities += self.prior_eps
        new_priorities = new_priorities.data.cpu().numpy().squeeze()
        new_priorities += epsilon_d
        self.memory.update_priorities(indices, new_priorities)

        # check the number of sampling demos
        demo_idxs = np.where(epsilon_d != 0.0)
        n_demo = demo_idxs[0].size

        return actor_loss.data, critic_loss.data, n_demo

    def _pretrain(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Pretraining steps."""
        actor_losses = []
        critic_losses = []
        print("Pre-Train %d step." % self.pretrain_step)
        for _ in range(1, self.pretrain_step + 1):
            actor_loss, critic_loss, _ = self.update_model()
            actor_losses.append(actor_loss.data)
            critic_losses.append(critic_loss.data)
        print("Pre-Train Complete!\n")
        return actor_losses, critic_losses

    def train(self, num_frames: int,
              method,
              task_name,
              rrt_oracle,
              evaluation_res,
              success_evaluation_res,
              query_pool_size=20,
              ratio=0.3,
              max_demo_num=60,
              random_goal=False,
              random_y=False,
              whether_render=False,
              sub_id=0
              ):
        traj_list = []
        uncertainty_list = []
        init_state_query_list = []
        teacher_demo_num = 0

        """Train the agent."""
        self.is_test = False

        # if self.demo:
        #     output = self._pretrain()

        while self.total_step < num_frames:
            # start a new episode
            state = self.env.reset(random=True, random_goal=random_goal, random_y=random_y)
            init_state = state.copy()
            done = False
            traj = []

            while not done:
                if whether_render:
                    self.env.render()
                    sleep(0.001)

                action = self.select_action(state)
                next_state, reward, done = self.step(action, role='learner')
                traj.append((state, action, reward, next_state, float(done)))

                # PER: increase beta
                fraction = min(self.total_step / num_frames, 1.0)
                self.beta = self.beta + fraction * (1.0 - self.beta)

                # if training is ready
                if (
                        len(self.memory) >= self.batch_size
                        and self.total_step > self.initial_random_steps
                ):
                    actor_loss, critic_loss, n_demo = self.update_model()

                if self.total_step % self.plot_interval == 0:
                    average_episode_reward, evaluation_res, average_success_rate, success_evaluation_res = self.evaluate_policy(evaluation_res=evaluation_res,
                                                                                                                                success_evaluation_res=success_evaluation_res,
                                                                                                                                env_step=self.total_step,
                                                                                                                                test_episode_num=100,
                                                                                                                                total_seed_num=10,
                                                                                                                                random_goal_task=random_goal,
                                                                                                                                random_y=random_y)
                    self.writer.add_scalar(tag='episode_reward/train/per_env_step', scalar_value=average_episode_reward,
                                           global_step=self.total_step)
                    self.writer.add_scalar(tag='success_rate/train/per_env_step', scalar_value=average_success_rate,
                                           global_step=self.total_step)
                    print("[{} environment steps finished]: Average episode reward is {}".format(self.total_step, average_episode_reward))


                    if method == 'ddpg_lfd_human':
                        model_path = 'models/' + task_name + '/' + method + '/' + 'sub_' + str(sub_id) + '/max_demo_' + str(max_demo_num) + '/'
                    else:
                        model_path = 'models/' + task_name + '/' + method + '/max_demo_' + str(max_demo_num) + '/'

                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    torch.save(self.actor.state_dict(), model_path + str(self.total_step) + '.pth')


                state = next_state
                self.total_step += 1

            ''' Estimate uncertainty for latest learner roll-out '''
            if method == 'active_ddpg':
                uncertainty = self.estimate_traj_uncertainty_td(traj=traj)
                traj_list.append(traj)
                whether_query = query_strategy(uncertainty_list, uncertainty, max_history_len=query_pool_size, ratio=ratio)

                uncertainty_list.append(uncertainty)
                init_state_query_list.append(init_state)
            elif method == 'active_ddpg_bern':
                seed = np.random.uniform(0, 1)
                whether_query = (seed <= ratio)
            else:
                whether_query = False

            ''' Query teacher for demo '''
            if teacher_demo_num < max_demo_num and whether_query:
                if method == 'active_ddpg':
                    max_uncertain_idx = np.array(uncertainty_list).argmax()
                    state = init_state_query_list[max_uncertain_idx]
                else:
                    state = self.env.reset(random=True, random_goal=random_goal, random_y=random_y)

                # generate oracle demonstrations
                pos_goal = (state[0], state[1])
                pos_init = (state[2], state[3])
                path = rrt_oracle.path_planning(pos_init=pos_init,
                                                pos_goal=pos_goal)

                # 2d np array of (total_step_num, action/state_dimension)
                oracle_states, oracle_actions = rrt_oracle.recover_demo(path=path,
                                                                        delta_t=1.0,
                                                                        pos_goal=pos_goal)

                state = self.env.reset(random=False, initial_x=pos_init[0], initial_y=pos_init[1], random_goal=False,
                                  goal_x=pos_goal[0], goal_y=pos_goal[1], random_y=False)
                done = False
                step = 0
                while not done:
                    if whether_render:
                        self.env.render()
                        sleep(0.001)

                    action = oracle_actions[step]
                    self.transition = [state, action]
                    next_state, reward, done = self.step(action, role='teacher')

                    # if training is ready
                    if (
                            len(self.memory) >= self.batch_size
                            and self.total_step > self.initial_random_steps
                    ):
                        actor_loss, critic_loss, n_demo = self.update_model()

                    if self.total_step % self.plot_interval == 0:
                        average_episode_reward, evaluation_res, average_success_rate, success_evaluation_res = self.evaluate_policy(
                            evaluation_res=evaluation_res,
                            success_evaluation_res=success_evaluation_res,
                            env_step=self.total_step,
                            test_episode_num=100,
                            total_seed_num=10,
                            random_goal_task=random_goal,
                            random_y=random_y)
                        self.writer.add_scalar(tag='episode_reward/train/per_env_step',
                                               scalar_value=average_episode_reward,
                                               global_step=self.total_step)
                        self.writer.add_scalar(tag='success_rate/train/per_env_step', scalar_value=average_success_rate,
                                               global_step=self.total_step)
                        print("[{} environment steps finished]: Average episode reward is {}".format(self.total_step,
                                                                                                     average_episode_reward))
                        model_path = 'models/' + task_name + '/' + method + '/max_demo_' + str(
                            max_demo_num) + '/ratio_' + str(ratio) + '/'
                        if not os.path.exists(model_path):
                            os.makedirs(model_path)
                        torch.save(self.actor.state_dict(), model_path + str(self.total_step) + '.pth')

                    state = next_state
                    self.total_step += 1
                    step += 1

                teacher_demo_num += 1
                print("[{} demos provided]".format(teacher_demo_num))
                self.writer.add_scalar(tag='total_demo_num/per_env_step', scalar_value=teacher_demo_num,
                                  global_step=self.total_step)

            if len(uncertainty_list) > query_pool_size:
                uncertainty_list.pop(0)
                init_state_query_list.pop(0)
                traj_list.pop(0)

        self.save_results(method=method, task_name=task_name, evaluation_res=evaluation_res, success_evaluation_res=success_evaluation_res, sub_id=sub_id)

        self.env.close()

    def test(self):
        """Test the agent."""
        self.is_test = True

        state = self.env.reset()
        done = False
        score = 0

        frames = []
        while not done:
            frames.append(self.env.render(mode="rgb_array"))
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print("score: ", score)
        self.env.close()

        return frames

    def evaluate_policy(self, evaluation_res, success_evaluation_res, env_step, test_episode_num=10, total_seed_num=10, random_goal_task=False, random_y=False):
        test_episode_reward = 0.0
        total_success_num = 0.0

        res = []
        success_res = []
        res.append(env_step)
        success_res.append(env_step)

        for seed in range(total_seed_num):
            sampled_xs = np.random.default_rng(seed).uniform(low=0.01, high=19.9, size=test_episode_num)
            if random_goal_task:
                sampled_goal_xs = np.random.default_rng(seed).uniform(low=0.01, high=19.9, size=test_episode_num)
                if random_y:
                    sampled_ys = np.random.default_rng(seed).uniform(low=1.0, high=8.0, size=test_episode_num)
                    sampled_goal_ys = np.random.default_rng(seed).uniform(low=12.0, high=19.0, size=test_episode_num)
            for i in range(test_episode_num):
                if random_goal_task:
                    if random_y:
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
                    action_ =  self.actor(torch.FloatTensor(state_).to(self.device)).detach().cpu().numpy()
                    next_state_, reward_, done_, _ = self.test_env.step(action_)
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

    def estimate_traj_uncertainty_td(self, traj):
        cumulated_td_error = 0.0
        episode_length = len(traj)
        for state, action, reward, next_state, done in traj:
            state = torch.FloatTensor(state.reshape(-1, state.shape[0])).to(self.device)
            next_state = torch.FloatTensor(next_state.reshape(-1, next_state.shape[0])).to(self.device)
            action = torch.FloatTensor(action.reshape(-1, action.shape[0])).to(self.device)

            next_action = self.actor_target(next_state)
            next_value = self.critic_target(next_state, next_action).detach().item()
            curr_return = reward + self.gamma * next_value * (1 - done)
            values = self.critic(state, action).detach().item()

            td_error = curr_return - values
            cumulated_td_error += abs(td_error)

        traj_uncertainty = cumulated_td_error / episode_length

        return traj_uncertainty

    def save_results(self, method, task_name, evaluation_res, success_evaluation_res, sub_id=0, max_demo_num=60):
        ''' Save the results '''
        if method == 'ddpg_lfd_human':
            res_per_step_path = 'evaluation_res/new/' + task_name + '/' + method + '/' + 'sub_' + str(
                sub_id) + '/max_demo_' + str(max_demo_num) + '/'
            if not os.path.exists(res_per_step_path):
                os.makedirs(res_per_step_path)
        else:
            res_per_step_path = 'evaluation_res/new/' + task_name + '/' + method + '/max_demo_' + str(
                max_demo_num) + '/'
            if not os.path.exists(res_per_step_path):
                os.makedirs(res_per_step_path)

        np.savetxt(res_per_step_path + 'res_per_step_new.csv', evaluation_res, delimiter=' ')
        np.savetxt(res_per_step_path + 'success_res_per_step_new.csv', success_evaluation_res, delimiter=' ')

    def _get_critic_loss(
            self, samples: Dict[str, np.ndarray], gamma: float
    ) -> torch.Tensor:
        """Return element-wise critic loss."""
        device = self.device  # for shortening the following lines

        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.FloatTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        masks = 1 - done
        next_action = self.actor_target(next_state)
        next_value = self.critic_target(next_state, next_action)
        curr_return = reward + gamma * next_value * masks
        curr_return = curr_return.to(device).detach()

        # train critic
        values = self.critic(state, action)
        critic_loss_element_wise = (values - curr_return).pow(2)

        return critic_loss_element_wise

    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau

        for t_param, l_param in zip(
                self.actor_target.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        for t_param, l_param in zip(
                self.critic_target.parameters(), self.critic.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

    def _get_n_step_info_from_demo(
            self, demo: List, n_step: int
    ) -> Tuple[List, List]:
        """Return 1 step and n step demos."""
        demos_1_step = list()
        demos_n_step = list()
        n_step_buffer: Deque = deque(maxlen=n_step)

        for transition in demo:
            n_step_buffer.append(transition)

            if len(n_step_buffer) == n_step:
                # add a single step transition
                demos_1_step.append(n_step_buffer[0])

                # add a multi step transition
                curr_state, action = n_step_buffer[0][:2]

                # get n-step info
                reward, next_state, done = n_step_buffer[-1][-3:]
                for transition in reversed(list(n_step_buffer)[:-1]):
                    r, n_o, d = transition[-3:]

                    reward = r + self.gamma * reward * (1 - d)
                    next_state, done = (n_o, d) if d else (next_state, done)

                transition = (curr_state, action, reward, next_state, done)
                demos_n_step.append(transition)

        return demos_1_step, demos_n_step

    def _plot(
            self,
            frame_idx: int,
            scores: List[float],
            actor_losses: List[float],
            critic_losses: List[float],
            n_demo: List[int],
    ):
        """Plot the training progresses."""

        def subplot(loc: int, title: str, values: List[float]):
            plt.subplot(loc)
            plt.title(title)
            plt.plot(values)

        subplot_params = [
            (141, f"frame {frame_idx}. score: {np.mean(scores[-10:])}", scores),
            (142, "actor_loss", actor_losses),
            (143, "critic_loss", critic_losses),
            (144, "the number of sampling demos", n_demo),
        ]

        # clear_output(True)
        plt.figure(figsize=(30, 5))
        for loc, title, values in subplot_params:
            subplot(loc, title, values)
        plt.show()


class ActionNormalizer(gym.ActionWrapper):
    """Rescale and relocate the actions."""

    def action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (-1, 1) to (low, high)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (low, high) to (-1, 1)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, -1.0, 1.0)

        return action

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', help='name of task', default='nav_1')
    parser.add_argument('--sub_id', help='id of human subject', default=1, type=int)
    parser.add_argument('--method', help='name of the method')

    return parser.parse_args()

def main():
    args = argparser()

    seed = 6
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ''' prepare the environment '''
    whether_render = False
    nav_env = NavEnv(render=whether_render)
    env = ActionNormalizer(nav_env)
    test_env = NavEnv(render=False)
    test_env = ActionNormalizer(test_env)

    ''' Training hyperparameters'''
    task_name = args.task_name
    # task_name = 'nav_3'

    if task_name == 'nav_1':
        random_goal = False
        random_y = False
        max_env_steps = int(10 * 1e4)
    elif task_name == 'nav_2':
        random_goal = True
        random_y = False
        max_env_steps = int(10 * 1e4)
    else:
        random_goal = True
        random_y = True
        max_env_steps = int(15 * 1e4)

    ou_noise_theta = 1.0
    ou_noise_sigma = 0.1
    max_demo_num = 60
    n_step = 3
    pretrain_step = 1000

    env_step = 0
    batch_size = 128
    teacher_demo_num = 0
    query_pool_size = 200
    ratio = 0.3

    ''' Choose the method to use '''
    method = args.method
    # method = 'active_ddpg'
    # method = 'active_ddpg_bern'
    # method = 'ddpg_lfd'
    # method = 'ddpg_lfd_human'

    # initial_random_steps = int(1e4)
    initial_random_steps = 0

    ''' Choose how to measure uncertainty '''
    uncertainty_method = 'td_error'

    ''' Choose how to generate candidate queries '''
    uniform_init_sampling = True
    max_uncertain_rollout = False

    ''' Choose how to select the query to make '''
    stream_based_query = True

    plot_interval = 1000
    rrt_oracle = RRT_Oracle()
    evaluate_res_per_step = []
    success_evaluate_res_per_step = []
    evaluate_res_per_demo = []
    success_evaluate_res_per_demo = []
    init_state_query_list = []
    init_state_history = []
    uncertainty_list = []
    traj_list = []
    states_traj_list = []
    demo = []

    replay_buffer_size = 1000000
    if method == 'ddpg_lfd_human':
        writer = SummaryWriter(
            'logs/new/' + task_name + '/' + method + '/' + 'sub_' + str(args.sub_id) + '/max_demo_' + str(max_demo_num) + '/' + TIMESTAMP)
    else:
        writer = SummaryWriter('logs/new/' + task_name + '/' + method + '/max_demo_' + str(max_demo_num) + '/' + TIMESTAMP)

    ''' start training '''
    if method == 'ddpg_lfd':
        print("[DDPG LfD]: Start to first collect teacher demo before learning ... ")
        for demo_num in range(max_demo_num):
            state = env.reset(random=True, random_goal=random_goal, random_y=random_y)
            pos_goal = (state[0], state[1])
            pos_init = (state[2], state[3])
            path = rrt_oracle.path_planning(pos_init=pos_init,
                                            pos_goal=pos_goal)

            # 2d np array of (total_step_num, action/state_dimension)
            oracle_states, oracle_actions = rrt_oracle.recover_demo(path=path,
                                                                    delta_t=1.0,
                                                                    pos_goal=pos_goal)
            state = env.reset(random=False, initial_x=pos_init[0], initial_y=pos_init[1], random_goal=False,
                              goal_x=pos_goal[0], goal_y=pos_goal[1], random_y=False)
            done = False
            step = 0
            while not done:
                action = oracle_actions[step]
                next_state, reward, done, _ = env.step(action)

                demo.append((state, action, reward, next_state, done))

                state = next_state
                step += 1

            print("[DDPG LfD]: demo {} is collected".format(demo_num + 1))

        print("[DDPG LfD]: All demo are collected. Going to train the model")
        print("***************************")
    elif method == 'ddpg_lfd_human':
        print("[DDPG LfD Human]: Start to load human joystick demo ...")
        joystick_demo_traj_data_path = 'joystick_trajectory_data/' + task_name + '/' + method + '/' + 'sub_' + str(
            args.sub_id) + '/max_demo_' + str(max_demo_num) + '/'

        demo_state_start_ids = np.genfromtxt(joystick_demo_traj_data_path + 'demo_state_start_ids.csv', delimiter=' ')
        demo_action_start_ids = np.genfromtxt(joystick_demo_traj_data_path + 'demo_action_start_ids.csv', delimiter=' ')
        demo_state_trajs = np.genfromtxt(joystick_demo_traj_data_path + 'joystick_demo_state_trajs.csv', delimiter=' ')
        demo_action_trajs = np.genfromtxt(joystick_demo_traj_data_path + 'joystick_demo_action_trajs.csv', delimiter=' ')

        for demo_num in range(max_demo_num):
            state_start_id = int(demo_state_start_ids[demo_num])
            action_start_id = int(demo_action_start_ids[demo_num])

            if demo_num < max_demo_num - 1:
                state_start_id_next = int(demo_state_start_ids[demo_num + 1])
                action_start_id_next = int(demo_action_start_ids[demo_num + 1])
                oracle_states = demo_state_trajs[(state_start_id + 1):state_start_id_next, :]
                oracle_actions = demo_action_trajs[(action_start_id + 1):action_start_id_next, :]
            else:
                oracle_states = demo_state_trajs[(state_start_id + 1):, :]
                oracle_actions = demo_action_trajs[(action_start_id + 1):, :]

            state = oracle_states[0, :] # get the starting state from the joystick demo
            pos_goal = (state[0], state[1])
            pos_init = (state[2], state[3])
            state = env.reset(random=False, initial_x=pos_init[0], initial_y=pos_init[1], random_goal=False,
                              goal_x=pos_goal[0], goal_y=pos_goal[1], random_y=False)
            done = False
            step = 0
            while not done:
                action = oracle_actions[step]
                next_state, reward, done, _ = env.step(action)

                demo.append((state, action, reward, next_state, done))

                state = next_state
                step += 1

            print("[DDPG LfD Human]: demo {} is collected".format(demo_num + 1))

        print("[DDPG LfD Human]: All demo are collected. Going to train the model")
        print("***************************")

    agent = DDPGfDAgent(
        env=env,
        memory_size=replay_buffer_size,
        batch_size=batch_size,
        ou_noise_theta=ou_noise_theta,
        ou_noise_sigma=ou_noise_sigma,
        demo=demo,
        n_step=n_step,
        pretrain_step=pretrain_step,
        initial_random_steps=initial_random_steps,
        writer=writer,
        test_env=test_env,
        plot_internal=plot_interval,
        seed=seed
    )

    agent.train(num_frames=max_env_steps,
                method=method,
                task_name=task_name,
                rrt_oracle=rrt_oracle,
                evaluation_res=evaluate_res_per_step,
                success_evaluation_res=success_evaluate_res_per_step,
                query_pool_size=query_pool_size,
                ratio=ratio,
                max_demo_num=max_demo_num,
                random_goal=random_goal,
                random_y=random_y,
                whether_render=False,
                sub_id=args.sub_id
                )

    writer.close()
    print("************************")
    print("All training finished")


if __name__ == '__main__':
    main()