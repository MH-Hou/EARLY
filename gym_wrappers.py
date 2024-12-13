from typing import Dict, Union

import gym
import numpy as np
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn


class TimeFeatureWrapper(gym.Wrapper):
    """
    Add remaining, normalized time to observation space for fixed length episodes.
    See https://arxiv.org/abs/1712.00378 and https://github.com/aravindr93/mjrl/issues/13.

    .. note::

        Only ``gym.spaces.Box`` and ``gym.spaces.Dict`` (``gym.GoalEnv``) 1D observation spaces
        are supported for now.

    :param env: Gym env to wrap.
    :param max_steps: Max number of steps of an episode
        if it is not wrapped in a ``TimeLimit`` object.
    :param test_mode: In test mode, the time feature is constant,
        equal to zero. This allow to check that the agent did not overfit this feature,
        learning a deterministic pre-defined sequence of actions.
    """

    def __init__(self, env: gym.Env, max_steps: int = 1000, test_mode: bool = False):
        assert isinstance(
            env.observation_space, (gym.spaces.Box, gym.spaces.Dict)
        ), "`TimeFeatureWrapper` only supports `gym.spaces.Box` and `gym.spaces.Dict` (`gym.GoalEnv`) observation spaces."

        # Add a time feature to the observation
        if isinstance(env.observation_space, gym.spaces.Dict):
            assert "observation" in env.observation_space.spaces, "No `observation` key in the observation space"
            obs_space = env.observation_space.spaces["observation"]
            assert isinstance(
                obs_space, gym.spaces.Box
            ), "`TimeFeatureWrapper` only supports `gym.spaces.Box` observation space."
            obs_space = env.observation_space.spaces["observation"]
        else:
            obs_space = env.observation_space

        assert len(obs_space.shape) == 1, "Only 1D observation spaces are supported"

        low, high = obs_space.low, obs_space.high
        low, high = np.concatenate((low, [0.0])), np.concatenate((high, [1.0]))
        self.dtype = obs_space.dtype

        if isinstance(env.observation_space, gym.spaces.Dict):
            env.observation_space.spaces["observation"] = gym.spaces.Box(low=low, high=high, dtype=self.dtype)
        else:
            env.observation_space = gym.spaces.Box(low=low, high=high, dtype=self.dtype)

        super(TimeFeatureWrapper, self).__init__(env)

        # Try to infer the max number of steps per episode
        try:
            self._max_steps = env.spec.max_episode_steps
        except AttributeError:
            self._max_steps = None

        # Fallback to provided value
        if self._max_steps is None:
            self._max_steps = max_steps

        self._current_step = 0
        self._test_mode = test_mode

    def reset(self, **kwargs) -> GymObs:
        self._current_step = 0
        return self._get_obs(self.env.reset( **kwargs))

    def step(self, action: Union[int, np.ndarray]) -> GymStepReturn:
        self._current_step += 1
        obs, reward, done, info = self.env.step(action)
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs: Union[np.ndarray, Dict[str, np.ndarray]]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Concatenate the time feature to the current observation.

        :param obs:
        :return:
        """
        # Remaining time is more general
        time_feature = 1 - (self._current_step / self._max_steps)
        if self._test_mode:
            time_feature = 1.0
        time_feature = np.array(time_feature, dtype=self.dtype)

        if isinstance(obs, dict):
            obs["observation"] = np.append(obs["observation"], time_feature)
            return obs
        return np.append(obs, time_feature)


class ResetWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        # Call the parent constructor, so we can access self.env later
        super().__init__(env)

    def reset(self, **kwargs):
        self.env.reset()

        whether_random = kwargs.get('whether_random', True)
        with self.env.sim.no_rendering():
            if whether_random:
                self.env.robot.reset()
                self.env.task.reset()
            else:
                self.env.robot.reset()
                goal_pos = kwargs.get('goal_pos') # 1d array of the form (x, y, z)
                object_pos = kwargs.get('object_pos') # 1d array of the form (x, y, z)
                self.env.task.goal = goal_pos
                self.env.task.sim.set_base_pose("target", self.env.task.goal, [0, 0, 0, 1])

                if object_pos is None:
                    pass
                else:
                    self.env.task.sim.set_base_pose("object", object_pos, [0, 0, 0, 1])

        # get obs
        robot_obs = self.env.robot.get_obs()  # robot state
        task_obs = self.env.task.get_obs()  # object position, velococity, etc...
        observation = np.concatenate([robot_obs, task_obs])
        achieved_goal = self.env.task.get_achieved_goal()

        obs = {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": self.task.get_goal(),
               }

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info