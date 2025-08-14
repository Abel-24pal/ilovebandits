"""Core and main classes for the MAB problem."""

import warnings
from typing import List, Tuple, Union

import numpy as np
from sklearn.base import clone, is_classifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .utils import argmax

####### COSNTANTS FOR ERROR NotEnoughRewardsPerArmError ########
# I force all arms to have at least "MIN_REWARDS_PER_ARM" unique rewards until all arms have "MIN_SAMPLES_TO_IGNORE_ARM" samples. Once we have "MIN_SAMPLES_TO_IGNORE_ARM" samples per arm, the error NotEnoughRewardsPerArmError will not be raised anymore.
MIN_REWARDS_PER_ARM = 2  # Minimum number of unique rewards per arm to avoid NotEnoughRewardsPerArmError
MIN_SAMPLES_TO_IGNORE_ARM = 100  # Minimum number of samples per arm to avoid NotEnoughRewardsPerArmError


class MismatchedArmNumberError(Exception):
    """Exception raised when the number of arms in the agent does not match the number of arms in the training data."""

    def __init__(self, expected_arms: int, actual_arms: int):
        super().__init__(f"Expected {expected_arms} arms, but found {actual_arms} unique arms in a_train.")
        self.expected = expected_arms
        self.actual = actual_arms


class NotEnoughRewardsPerArmError(Exception):
    """Exception raised when there are not enough unique rewards per arm in the training data. We want at least 2 different rewards per arm."""

    def __init__(self, unique_rewards_per_arm: List[int]):
        super().__init__(
            f"There are arms that have less than two unique reards in the training data provided. Here, the unique rewards per arm for each arm:  {unique_rewards_per_arm}."
        )
        self.unique_rewards_per_arm = unique_rewards_per_arm


class BaseContextualAgent:
    """
    Base class for Contextual Bandit Agents.

    Parameters
    ----------
    arms : int
        Number of arms (actions) available to the agent.
    n_rounds_random : int, optional
        Number of rounds to take random actions.
    rng : np.random.Generator, optional
        Random number generator for reproducibility. Default is a random generator with seed 42.
    """

    def __init__(
        self,
        arms: int,
        n_rounds_random=200,
        rng_seed=None,
    ):
        self.arms = arms
        self.idx_arms = list(range(self.arms))  # arms are encoded in the range [0, arms-1]
        self.n_rounds_random = n_rounds_random
        self.update_agent_counts = 0
        if rng_seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed=rng_seed)

        self.reset_agent()

    def take_action(self, context) -> Tuple[int, float]:
        """
        Takes one step for the agent.

        According to the current context, self.n_rounds_random and self.update_agent_counts,
        returns the action the agent chooses at that time step.

        Parameters
        ----------
        context : np.ndarray (2D)
            The context features for which to estimate rewards of each arm. Shape (1, n_context_features).

        Returns
        -------
        int - the index of the current action.
        float - the probability of selecting the action.
        """
        if sum(self.arm_count) < self.n_rounds_random:
            current_action, prob_action = self.take_random_action()
        elif sum(self.arm_count) >= self.n_rounds_random and self.update_agent_counts == 0:
            warnings.warn(
                f"RANDOM ACTION: Agent {self.__class__.__name__} is taking a random action because it has not been updated yet. Please call update_agent() with training data before taking actions.",
                stacklevel=2,
            )
            current_action, prob_action = self.take_random_action()
        else:
            current_action, prob_action = self.take_agent_action(context)

        # Update arm count and last action
        self.arm_count[current_action] += 1
        self.last_action = current_action

        return current_action, prob_action

    def take_random_action(self) -> Tuple[int, float]:
        """
        Returns a random action from the available arms.

        Returns
        -------
        int - the index of the current action.
        float - the probability of selecting the action.
        """
        current_action = self.rng.choice(self.idx_arms)
        prob_action = 1 / self.arms  # noqa: F841

        return current_action, prob_action

    def take_agent_action(self, context: np.ndarray) -> Tuple[int, float]:
        """
        Takes one step for the agent according to the current context.

        Parameters
        ----------
        context : np.ndarray (2D)
            The context features for which to estimate rewards of each arm. Shape (1, n_context_features).

        Returns
        -------
        int - the index of the current action.
        float - the probability of selecting the action.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def update_agent(self, contexts: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> None:
        """
        Update the agent's parameters based on the historical data. This method should be implemented by subclasses.

        Parameters
        ----------
        c_train : np.ndarray (2D). Shape (n_samples, n_context_features).
            The context features for the training data samples.
        a_train : np.ndarray (2D). Shape (n_samples, ).
            The arms selected for each training data sample.
        r_train : np.ndarray. Shape (n_samples, ).
            The obtained reward for each training data sample.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def reset_agent(self):
        """Reset agent parameters such as arm counters."""
        self.arm_count = [0.0 for _ in range(self.arms)]  # number of times an arm is taken
        self.last_action = None
        self.update_agent_counts = 0


class EpsGreedyConAgent(BaseContextualAgent):
    """Epsilon Greedy Agent. Take Greedy action 1-epsilon% of times. Take random action epsilon% of times.

    Parameters
    ----------
    arms : int
        Number of arms (actions) available to the agent.
    n_rounds_random : int, optional
        Number of rounds to take random actions.
    rng : np.random.Generator, optional
        Random number generator for reproducibility. Default is a random generator with seed 42.
    epsilon : float
        Probability of taking a random action. Default is 0.1.
    one_model_per_arm : bool
        If True, the agent will maintain a separate model for each arm. If False, a single model will be used for all arms.
    base_estimator : Union[RandomForestRegressor, RandomForestClassifier]
        The base estimator to be used for fitting the reward model.
    """

    def __init__(  # noqa: PLR0913
        self,
        base_estimator: Union[RandomForestRegressor, RandomForestClassifier],  # NOT PREPARED FOR CLASSIFICATION YET!
        arms: int,
        n_rounds_random: int = 200,
        epsilon: float = 0.1,
        one_model_per_arm: bool = True,
        rng_seed=None,
        min_rewards_per_arm: int = MIN_REWARDS_PER_ARM,
        min_samples_to_ignore_arm: int = MIN_SAMPLES_TO_IGNORE_ARM,
    ):
        super().__init__(
            arms=arms, n_rounds_random=n_rounds_random, rng_seed=rng_seed
        )  # it executes the __init__ function of the base class
        self.epsilon = epsilon
        self.base_estimator = base_estimator
        self.one_model_per_arm = one_model_per_arm

        # Choose the prediction function dynamically (.predict or .predict_proba)
        self.predfun = (
            (lambda m, x: m.predict_proba(x)[:, 1])  # or another slice depending on needs
            if is_classifier(self.base_estimator)
            else (lambda m, x: m.predict(x))
        )

        self.MIN_REWARDS_PER_ARM = min_rewards_per_arm
        self.MIN_SAMPLES_TO_IGNORE_ARM = min_samples_to_ignore_arm

        self.reset_agent()

    def update_agent(self, c_train: np.ndarray, a_train: np.ndarray, r_train: np.ndarray) -> None:
        """
        Update the agent's parameters based on the historical data.This method should be implemented by subclasses.

        Parameters
        ----------
        c_train : np.ndarray (2D). Shape (n_samples, n_context_features).
            The context features for the training data samples.
        a_train : np.ndarray (2D). Shape (n_samples, ).
            The arms selected for each training data sample.
        r_train : np.ndarray. Shape (n_samples, ).
            The obtained reward for each training data sample.
        """
        self.nfeats = c_train.shape[1]

        a_unique, a_counts = np.unique(a_train, return_counts=True)
        idx_arms = list(np.sort(a_unique))
        narms_in_train = len(idx_arms)

        if narms_in_train != self.arms:
            raise MismatchedArmNumberError(expected_arms=self.arms, actual_arms=narms_in_train)
        if not np.issubdtype(a_train.dtype, np.integer):
            raise ValueError(f"Expected a_train to be of integer type, but found {a_train.dtype}.")
        if idx_arms != list(range(self.arms)):
            raise ValueError(
                f"Expected arms to be encoded in the range [0, {self.arms - 1}], but found arms: {self.idx_arms}."
            )

        unique_rewards_per_arm = [len(np.unique(r_train[a_train == arm])) for arm in range(self.arms)]
        if min(unique_rewards_per_arm) < self.MIN_REWARDS_PER_ARM and np.min(a_counts) < self.MIN_SAMPLES_TO_IGNORE_ARM:
            raise NotEnoughRewardsPerArmError(unique_rewards_per_arm=unique_rewards_per_arm)

        if self.one_model_per_arm:
            self.models = []
            # Update the model for each arm separately
            for arm in range(self.arms):
                x_train = c_train[a_train == arm]
                y_train = r_train[a_train == arm]

                self.models.append(
                    clone(self.base_estimator).fit(
                        x_train,
                        y_train,
                    )
                )

        else:
            # Update a single model for all arms
            x_train = np.column_stack((c_train, a_train))
            y_train = r_train
            self.model = clone(self.base_estimator).fit(
                x_train,
                y_train,
            )

        self.update_agent_counts += 1
        self.last_c_train = c_train
        self.last_a_train = a_train
        self.last_r_train = r_train

    def estimate_qvals(self, context: np.ndarray) -> List[np.ndarray]:
        """
        Estimate Q-values for each arm given the context.

        Parameters
        ----------
        context : np.ndarray (2D)
            The context features for which to estimate rewards of each arm. Shape (1, n_context_features).

        Returns
        -------
        List[np.ndarray] - A list of Q-values/rewards for each arm.
            Each element is a numpy array of shape (1,).
            It seems that this arrays with single elements work as a float
            number but with the advantages of being a numpy type, so we left the input this way.
        """
        if self.one_model_per_arm:
            self.qvals = [self.predfun(model, context) for model in self.models]
        else:
            self.qvals = [
                self.predfun(
                    self.model,
                    np.hstack((context, np.array([[arm]]))),
                )
                for arm in self.idx_arms
            ]

    def take_agent_action(
        self, context: np.ndarray
    ) -> Tuple[int, float]:  # the "take action" function of the base class is overridden
        """
        Takes one step for the agent.

        According to the current context,
        returns the action the agent chooses at that time step.

        Parameters
        ----------
        context : np.ndarray (2D)
            The context features for which to estimate rewards of each arm. Shape (1, n_context_features).

        Returns
        -------
        int - the index of the current action.
        float - the probability of selecting the action.
        """
        random_number = self.rng.random()

        self.estimate_qvals(context)

        # Select Action:
        greedy_action, partial_greedy_prob, list_greedy_actions = argmax(
            self.qvals,
            self.rng,
        )  # list_greedy_actions is useful to compute probabilities in case there are ties in the greedy action
        if random_number < self.epsilon:
            current_action = self.rng.choice(self.idx_arms)
        else:
            current_action = greedy_action

        # Compute probability
        if (
            current_action in list_greedy_actions
        ):  # we selected one of the greedy actions (it can be more than one in case of ties)
            prob_action = (1 - self.epsilon) * partial_greedy_prob + self.epsilon * (1 / self.arms)
        else:  # we selected a non-greedy action
            prob_action = self.epsilon * (1 / self.arms)

        return current_action, prob_action

    def reset_agent(self):
        """Reset agent parameters such as arm counters."""
        super().reset_agent()
        self.model = None
        self.models = None
        self.qvals = []


class BootStrapConAgent(BaseContextualAgent):
    """
    Bootstrap Agent. Disjoints models that are updated according to a bootstrapped sample for their training data associated to that arm. It tries to simulate Thompson Sampling by using bootstrapping.

    Parameters
    ----------
    arms : int
        Number of arms (actions) available to the agent.
    n_rounds_random : int, optional
        Number of rounds to take random actions.
    rng : np.random.Generator, optional
        Random number generator for reproducibility. Default is a random generator with seed 42.
    base_estimator : Union[RandomForestRegressor, RandomForestClassifier]
        The base estimator to be used for fitting the reward model.
    divisor_bootstrap : int
        if 1 we always bootstrap when take_agent_action(). If 2, we bootstrap half of the times
        we use take_agent_action(). If 3, we bootstrap a third of the times we use take_agent_action(), etc.

    See the paper for more details:
    https://www.auai.org/uai2017/proceedings/papers/171.pdf
    """

    def __init__(  # noqa: PLR0913
        self,
        base_estimator: Union[RandomForestRegressor, RandomForestClassifier],
        arms: int,
        n_rounds_random: int = 200,
        rng_seed=None,
        divisor_bootstrap: int = 1,  # if 1 we always bootstrap when take_agent_action(). If 2, we bootstrap half of the times we use take_agent_action(). If 3, we bootstrap a third of the times we use take_agent_action()
    ):
        super().__init__(
            arms=arms, n_rounds_random=n_rounds_random, rng_seed=rng_seed
        )  # it executes the __init__ function of the base class
        self.base_estimator = base_estimator
        self.divisor_bootstrap = divisor_bootstrap  # controls how often we bootstrap the models
        # Choose the prediction function dynamically (.predict or .predict_proba)
        self.predfun = (
            (lambda m, x: m.predict_proba(x)[:, 1])  # or another slice depending on needs
            if is_classifier(self.base_estimator)
            else (lambda m, x: m.predict(x))
        )
        self.reset_agent()

    def update_agent(self, c_train: np.ndarray, a_train: np.ndarray, r_train: np.ndarray) -> None:
        """
        Update the agent's parameters based on the historical data. This method should be implemented by subclasses.

        Parameters
        ----------
        c_train : np.ndarray (2D). Shape (n_samples, n_context_features).
            The context features for the training data samples.
        a_train : np.ndarray (2D). Shape (n_samples, ).
            The arms selected for each training data sample.
        r_train : np.ndarray. Shape (n_samples, ).
            The obtained reward for each training data sample.
        """
        self.nfeats = c_train.shape[1]
        idx_arms = list(np.sort(np.unique(a_train)))
        narms_in_train = len(idx_arms)

        if narms_in_train != self.arms:
            raise MismatchedArmNumberError(expected_arms=self.arms, actual_arms=narms_in_train)
        if not np.issubdtype(a_train.dtype, np.integer):
            raise ValueError(f"Expected a_train to be of integer type, but found {a_train.dtype}.")
        if idx_arms != list(range(self.arms)):
            raise ValueError(
                f"Expected arms to be encoded in the range [0, {self.arms - 1}], but found arms: {self.idx_arms}."
            )

        unique_rewards_per_arm = [len(np.unique(r_train[a_train == arm])) for arm in range(self.arms)]
        if min(unique_rewards_per_arm) < MIN_REWARDS_PER_ARM:
            raise NotEnoughRewardsPerArmError(unique_rewards_per_arm=unique_rewards_per_arm)

        self.models = []
        # Update the model for each arm separately
        for arm in range(self.arms):
            c_train_a = c_train[a_train == arm]
            r_train_a = r_train[a_train == arm]

            # bootstrap sampling
            n_train = c_train_a.shape[0]
            n_samples = n_train
            boot_indices = np.random.choice(n_train, size=n_samples, replace=True)

            x_train = c_train_a[boot_indices]
            y_train = r_train_a[boot_indices]

            self.models.append(
                clone(self.base_estimator).fit(
                    x_train,
                    y_train,
                )
            )

        self.update_agent_counts += 1
        self.last_c_train = c_train
        self.last_a_train = a_train
        self.last_r_train = r_train

    def estimate_qvals(self, context: np.ndarray) -> List[np.ndarray]:
        """
        Estimate Q-values for each arm given the context.

        Parameters
        ----------
        context : np.ndarray (2D)
            The context features for which to estimate rewards of each arm. Shape (1, n_context_features).

        Returns
        -------
        List[np.ndarray] - A list of Q-values/rewards for each arm.
            Each element is a numpy array of shape (1,).
            It seems that this arrays with single elements work as a float
            number but with the advantages of being a numpy type, so we left the input this way.
        """
        self.qvals = [self.predfun(model, context) for model in self.models]

    def take_agent_action(
        self, context: np.ndarray
    ) -> Tuple[int, float]:  # the "take action" function of the base class is overridden
        """
        Takes one step for the agent.

        According to the current context,
        returns the action the agent chooses at that time step.

        Parameters
        ----------
        context : np.ndarray (2D)
            The context features for which to estimate rewards of each arm. Shape (1, n_context_features).

        Returns
        -------
        int - the index of the current action.
        float - the probability of selecting the action.
        """
        self.estimate_qvals(context)

        # Select Action:
        current_action, _, _ = argmax(
            self.qvals,
            self.rng,
        )  # IMPORTANT: the prob_action in argmax is not the right probability for BootStrapping/Thompson Sampling

        prob_action = np.nan  # probability is very time-consuming to be computed in this agent, so we set it to NaN.

        if sum(self.arm_count) % self.divisor_bootstrap == 0:  # not executed always to avoid heavy computational cost
            # Bootstrap a new sample of current training and retrain models in each arm for next time that .take_action() is called
            self.update_agent(c_train=self.last_c_train, a_train=self.last_a_train, r_train=self.last_r_train)

        return current_action, prob_action

    def reset_agent(self):
        """Reset agent parameters such as arm counters."""
        super().reset_agent()
        self.models = None
        self.qvals = []
