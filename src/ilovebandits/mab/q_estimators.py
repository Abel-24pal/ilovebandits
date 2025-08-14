"""Core and main classes for the MAB problem."""

import random
from typing import Union

import numpy as np


class BaseScaler:
    """Base class for the reward scalers. This base class lets the rewards as it is."""

    def __init__(self) -> None:
        pass

    def scale(self, reward):
        """Scale reward."""
        return reward


class BernoulliBinarizationScaler(BaseScaler):
    """Class to convert rewards from [0,1] domain to {0,1} domain."""

    def __init__(self, seed=42):
        self.seed = seed
        random.seed(self.seed)

    def with_proba(self, epsilon):
        r"""Bernoulli test, with probability :math:`\varepsilon`, return `True`, and with probability :math:`1 - \varepsilon`, return `False`."""
        if epsilon > 1 or epsilon < 0:
            raise Exception(
                f"Error: only bounded rewards in [0, 1] are supported by this Beta posterior right now. We get a reward={epsilon}."
            )
        return random.random() < epsilon  # True with proba epsilon  # noqa: S311

    def scale(self, reward):
        """Scale reward from [0,1] to {0,1}."""
        if reward == 0:
            return 0  # Returns a int!
        elif reward == 1:
            return 1  # Returns a int!
        else:
            return int(self.with_proba(reward))


class QEstBase:
    """Base class for the Q estimators. This base class is just a blueprint."""

    def __init__(
        self,
        arms: int,
        qvals_init=None,
        reward_scaler: Union["BaseScaler", "BernoulliBinarizationScaler", "None"] = None,
    ) -> None:
        self.arms = arms

        if qvals_init is None:
            self.qvals = [0.0 for _ in range(self.arms)]
        else:
            self.qvals = qvals_init

            if len(self.qvals) != self.arms:
                raise ValueError("Number of arms does not match the provided qvals_init length.")

        if reward_scaler is None:
            reward_scaler = BaseScaler()

        self.reward_scaler = reward_scaler
        self.reset_arm_counts()

    def estimate(self, action):
        """Base estimate: update the arm count for the updates."""
        self.arm_count_updates[action] += 1
        pass

    def reset_arm_counts(self):
        """Reset arm count updates."""
        self.arm_count_updates = [0.0 for _ in range(self.arms)]


class QEstMean(QEstBase):
    """Q estimator to estimate reward with sample average estimates."""

    def __init__(
        self,
        arms: int,
        qvals_init=None,
        reward_scaler: Union["BaseScaler", "BernoulliBinarizationScaler", "None"] = None,
    ) -> None:
        super().__init__(arms=arms, qvals_init=qvals_init, reward_scaler=reward_scaler)

    def estimate(self, reward, action):
        """Estimate reward for the given arm/action."""
        super().estimate(action=action)
        reward = self.reward_scaler.scale(reward)

        step_size = 1 / self.arm_count_updates[action]
        self.qvals[action] = self.qvals[action] + step_size * (reward - self.qvals[action])


class QEstFixedStep(QEstBase):
    """Q estimator to estimate reward with constant step size."""

    def __init__(
        self,
        arms: int,
        step_size: float,
        qvals_init=None,
        reward_scaler: Union["BaseScaler", "BernoulliBinarizationScaler", "None"] = None,
    ) -> None:
        super().__init__(arms=arms, qvals_init=qvals_init, reward_scaler=reward_scaler)
        self.step_size = step_size

    def estimate(self, reward, action):
        """Estimate reward for the given arm/action."""
        super().estimate(action=action)
        reward = self.reward_scaler.scale(reward)

        self.qvals[action] = self.qvals[action] + self.step_size * (reward - self.qvals[action])


class QThompSamp(QEstBase):
    """Q estimator to estimate the reward for the Thompson Sampling agent assuming a bernourlli RV for the rewards. In this case, beta is the prior."""

    def __init__(
        self,
        alphas: np.array,
        betas: np.array,
        reward_scaler: Union["BaseScaler", "BernoulliBinarizationScaler", "None"] = None,
    ) -> None:
        if len(alphas) != len(betas):
            raise Exception("betas and alphas should have the same length")

        self.arms = len(alphas)
        self.alphas = alphas
        self.betas = betas

        if reward_scaler is None:
            reward_scaler = BaseScaler()

        self.reward_scaler = reward_scaler

        self.get_expected_values()
        self.reset_arm_counts()

    def get_expected_values(self):
        """Get expected values of the assumed distributions."""
        # here, expected values of the distributions are computed
        self.qvals = [alpha / (alpha + beta) for alpha, beta in zip(self.alphas, self.betas, strict=True)]

    def estimate(self, reward, action):
        """Estimate reward for the given arm/action."""
        super().estimate(action=action)
        reward = self.reward_scaler.scale(reward)

        self.alphas[action] += reward
        self.betas[action] += 1 - reward

        self.get_expected_values()

    def sample_thetas(self):
        """Sample thetas from current distribution."""
        return [np.random.beta(self.alphas[arm], self.betas[arm]) for arm in range(self.arms)]
