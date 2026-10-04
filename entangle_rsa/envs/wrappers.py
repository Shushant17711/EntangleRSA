"""Thin Gymnasium wrappers: fixed observation scaling and per-task metadata used by H3."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class TaskSpec:
    name: str
    gym_id: str
    obs_dim: int
    discrete: bool
    n_actions: int  # number of discrete actions, or action dim for continuous
    scale: tuple[float, ...]
    # Indices into the *physical feature* vector returned by `physical_features` (see below).
    coupled_dims: tuple[int, ...]
    uncoupled_dims: tuple[int, ...]
    feature_names: tuple[str, ...]
    solved_return: float


TASKS: dict[str, TaskSpec] = {
    "cartpole": TaskSpec(
        name="cartpole",
        gym_id="CartPole-v1",
        obs_dim=4,
        discrete=True,
        n_actions=2,
        scale=(2.4, 3.0, 0.21, 3.0),
        # obs = [x, x_dot, theta, theta_dot]; the brief's coupled pair is pole angle x cart velocity
        coupled_dims=(2, 1),
        uncoupled_dims=(0, 3),
        feature_names=("x", "x_dot", "theta", "theta_dot"),
        solved_return=475.0,
    ),
    "pendulum": TaskSpec(
        name="pendulum",
        gym_id="Pendulum-v1",
        obs_dim=3,
        discrete=False,
        n_actions=1,
        scale=(1.0, 1.0, 8.0),
        # physical features = [angle, angular velocity]; the only physical pair is the coupled one,
        # so H3 contrasts joint (both-dimension) differences against single-dimension differences.
        coupled_dims=(0, 1),
        uncoupled_dims=(),
        feature_names=("angle", "theta_dot"),
        solved_return=-200.0,
    ),
    "acrobot": TaskSpec(
        name="acrobot",
        gym_id="Acrobot-v1",
        obs_dim=6,
        discrete=True,
        n_actions=3,
        scale=(1.0, 1.0, 1.0, 1.0, 4 * np.pi, 9 * np.pi),
        # physical features = [theta1, theta2, dtheta1, dtheta2]
        coupled_dims=(0, 1),
        uncoupled_dims=(2, 3),
        feature_names=("theta1", "theta2", "dtheta1", "dtheta2"),
        solved_return=-100.0,
    ),
}


def get_task(name: str) -> TaskSpec:
    try:
        return TASKS[name]
    except KeyError:
        raise ValueError(f"unknown task {name!r}; expected one of {sorted(TASKS)}") from None


def scale_obs(task: TaskSpec, obs: np.ndarray) -> np.ndarray:
    """Fixed, documented observation scaling applied identically to every policy family."""
    return (np.asarray(obs, dtype=np.float32) / np.asarray(task.scale, dtype=np.float32)).astype(np.float32)


def physical_features(task: TaskSpec, obs: np.ndarray) -> np.ndarray:
    """Raw observations -> physically meaningful coordinates used to define H3 probe-pair partitions."""
    obs = np.asarray(obs, dtype=np.float64)
    if task.name == "cartpole":
        return obs
    if task.name == "pendulum":
        return np.stack([np.arctan2(obs[..., 1], obs[..., 0]), obs[..., 2]], axis=-1)
    if task.name == "acrobot":
        t1 = np.arctan2(obs[..., 1], obs[..., 0])
        t2 = np.arctan2(obs[..., 3], obs[..., 2])
        return np.stack([t1, t2, obs[..., 4], obs[..., 5]], axis=-1)
    raise ValueError(task.name)


def make_env(task_name: str, seed: int | None = None) -> gym.Env:
    task = get_task(task_name)
    env = gym.make(task.gym_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def make_vec_env(task_name: str, n_envs: int, seed: int) -> gym.vector.VectorEnv:
    task = get_task(task_name)

    def thunk(i):
        def _make():
            env = gym.make(task.gym_id)
            env.action_space.seed(seed + i)
            return env

        return _make

    envs = gym.vector.SyncVectorEnv([thunk(i) for i in range(n_envs)])
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)
    return envs
