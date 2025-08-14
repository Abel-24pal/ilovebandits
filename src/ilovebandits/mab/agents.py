"""Core and main classes for the MAB problem."""

from copy import deepcopy
from typing import Dict, Tuple, Union

import numpy as np

from .q_estimators import QEstFixedStep, QEstMean, QThompSamp
from .utils import argmax, ucb_uncertainty


class BaseAgent:
    """Base class for the agents. This base class is just a blueprint."""

    def __init__(self, q_estimator: Union["QEstMean", "QEstFixedStep"]):
        """Setup for the agent called when the experiment first starts."""
        self.arms = q_estimator.arms
        self.initial_q_estimator = deepcopy(q_estimator)

        self.reset_agent()

    def take_action(self):
        """
        Takes one step for the agent.

        It takes in a reward and observation and
        returns the action the agent chooses at that time step.
        """
        pass

    def reset_only_estimator(self):
        """Reset q_estimator of the agent."""
        self.q_estimator = deepcopy(self.initial_q_estimator)

    def reset_agent(self):
        """Reset agent parameters such as arm counters and q estimations."""
        self.arm_count = [0.0 for _ in range(self.arms)]  # number of times an arm is taken
        self.last_action = None
        self.q_estimator = deepcopy(self.initial_q_estimator)
        # self.q_estimator.reset_arm_counts() # it seems that it is not needed.


class GreedyAgent(BaseAgent):
    """Pure Greedy Agent. Takes always greedy action."""

    def take_action(self) -> Tuple[int, int, float]:  # the "take action" function of the base class is overridden
        """
        Takes one step for the agent.

        It takes in a reward and observation and
        returns the action the agent chooses at that time step.

        Returns
        -------
        int - the index of the current action.
        int - the number of times the current action has been chosen.
        float - the probability of selecting the action.
        """
        current_action, prob_action, _ = argmax(self.q_estimator.qvals)
        self.arm_count[current_action] += 1
        self.last_action = current_action

        return current_action, self.arm_count[current_action], prob_action


class EpsilonGreedyAgent(BaseAgent):
    """Epsilon Greedy Agent. Take Greedy action 1-epsilon% of times. Take random action epsilon% of times."""

    def __init__(self, q_estimator: Union["QEstMean", "QEstFixedStep"], epsilon: float):
        super().__init__(q_estimator)  # it executes the __init__ function of the base class
        self.epsilon = epsilon

    def take_action(self) -> Tuple[int, int, float]:  # the "take action" function of the base class is overridden
        """
        Takes one step for the agent.

        It takes in a reward and observation and
        returns the action the agent chooses at that time step.

        Returns
        -------
        int - the index of the current action.
        int - the number of times the current action has been chosen.
        float - the probability of selecting the action.
        """
        random_number = np.random.random()

        # Select Action:
        greedy_action, partial_greedy_prob, list_greedy_actions = argmax(
            self.q_estimator.qvals
        )  # list_greedy_actions is useful to compute probabilities in case there are ties in the greedy action
        if random_number < self.epsilon:
            current_action = np.random.choice(list(range(len(self.q_estimator.qvals))))
        else:
            current_action = greedy_action

        # Compute probability
        if (
            current_action in list_greedy_actions
        ):  # we selected one of the greedy actions (it can be more than one in case of ties)
            prob_action = (1 - self.epsilon) * partial_greedy_prob + self.epsilon * (1 / len(self.q_estimator.qvals))
        else:  # we selected a non-greedy action
            prob_action = self.epsilon * (1 / len(self.q_estimator.qvals))

        # Update arm count and last action
        self.arm_count[current_action] += 1
        self.last_action = current_action

        return current_action, self.arm_count[current_action], prob_action


class RandomAgent(BaseAgent):
    """Random Agent. It can be used as a baseline."""

    def take_action(self) -> Tuple[int, int, float]:  # the "take action" function of the base class is overridden
        """
        Takes one step for the agent.

        It takes in a reward and observation and
        returns the action the agent chooses at that time step.

        Returns
        -------
        int - the index of the current action.
        int - the number of times the current action has been chosen.
        float - the probability of selecting the action.
        """
        current_action = np.random.choice(list(range(len(self.q_estimator.qvals))))
        prob_action = 1 / len(self.q_estimator.qvals)  # noqa: F841

        self.arm_count[current_action] += 1
        self.last_action = current_action

        return current_action, self.arm_count[current_action], prob_action


class UCBAgent(BaseAgent):
    """Implements the UCB1 Agent."""

    def __init__(self, q_estimator: Union["QEstMean", "QEstFixedStep"], c: float):
        super().__init__(q_estimator)  # it executes the __init__ function of the base class
        self.c = c  # c should be positive
        self.reset_agent()

    def take_action(self) -> Tuple[int, int, float]:  # the "take action" function of the base class is overridden
        """
        Takes one step for the agent.

        It takes in a reward and observation and
        returns the action the agent chooses at that time step.

        Returns
        -------
        int - the index of the current action.
        int - the number of times the current action has been chosen.
        float - the probability of selecting the action.
        """
        uncertainty = ucb_uncertainty(c=self.c, arm_count_updates=self.q_estimator.arm_count_updates)

        # step = sum(self.q_estimator.arm_count_updates) + 1
        # uncertainty = np.array([
        #     self.c * np.sqrt(np.log(step) / arm_count) if arm_count > 0 else float('inf')
        #     for arm_count in self.q_estimator.arm_count_updates
        # ])

        current_action, prob_action, _ = argmax(self.q_estimator.qvals + uncertainty)

        self.last_choice_uncertainties = uncertainty
        self.arm_count[current_action] += 1
        self.last_action = current_action

        return current_action, self.arm_count[current_action], prob_action

    def reset_agent(self):
        """Reset agent parameters such as arm counters and q estimations."""
        super().reset_agent()
        self.last_choice_uncertainties = np.array([float("inf") for _ in range(self.arms)])


class TSAgent(BaseAgent):  # Thompson Sampling Agent for discrete 0-1 rewards
    """Implements Thompson Sampling Agent for discrete 0-1 rewards. Default a_init=1 and b_init=1 correspond to unfiform distribution."""

    def __init__(self, arms: int, a_init: float = 1, b_init: float = 1, samples_for_freq_est: int = 100000):
        self.samples = samples_for_freq_est
        if a_init <= 0 or b_init <= 0:
            raise Exception("No valid parameters for initial beta distribution")

        super().__init__(
            q_estimator=QThompSamp(alphas=a_init * np.ones(arms), betas=b_init * np.ones(arms))
        )  # it executes the __init__ function of the base class
        self.reset_agent()

    def _freq_arms(self) -> Dict[int, float]:
        """Return the chosen frequency of each arm.

        Returns
        -------
        dict[int, float] - dictionary that maps the arm index to the frequency.
        """
        try:
            list_idxs = []
            for _ in range(self.samples):
                idx, _, _ = argmax(self.q_estimator.sample_thetas())
                list_idxs.append(idx)

            unique, counts = np.unique(list_idxs, return_counts=True)
            dic_arm_freq = dict(zip(unique, counts / len(list_idxs), strict=True))

            # if there is an action that never was chosen, we assign a frequency of None. This indicates that we were not able to estimate the frequency with the samples we took.
            dic_arm_freq.update({arm: None for arm in range(self.q_estimator.arms) if arm not in dic_arm_freq.keys()})
        except Exception as e:
            print(f"Error estimating frequency of arms' selection: {e}")
            raise e

        return dic_arm_freq

    def take_action(self) -> Tuple[int, int, float]:  # the "take action" function of the base class is overridden
        """
        Takes one step for the agent.

        It takes in a reward and observation and
        returns the action the agent chooses at that time step.

        Returns
        -------
        int - the index of the current action.
        int - the number of times the current action has been chosen.
        float - the probability of selecting the action.
        """
        theta_samples = self.q_estimator.sample_thetas()

        current_action, _, _ = argmax(
            theta_samples
        )  # IMPORTANT: the prob_action in argmax is not the right probability for Thompson Sampling

        prob_action = self._freq_arms()[current_action]

        self.last_theta_samples = theta_samples
        self.arm_count[current_action] += 1
        self.last_action = current_action

        return (
            current_action,
            self.arm_count[current_action],
            prob_action,
        )

    def reset_agent(self):
        """Reset agent parameters such as arm counters and q estimations."""
        super().reset_agent()
        self.last_theta_samples = [None for _ in range(self.arms)]
