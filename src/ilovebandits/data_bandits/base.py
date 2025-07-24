"""DATABASED BANDITS. Transform classification datasets into bandit problems."""

import numpy as np
import pandas as pd


class DataBasedBanditFromPandas:
    """
    Initialize the DataBasedbandit with a DataFrame.

    Transforms a classification problem into a bandit problem.
    It assumes the last column of the DataFrame is the target variable.
    Each class is considered an arm. If the action taken mathches the target label, it gives a reward of 1, 0 otherwise.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the data for the bandit. The last column should be the target variable
    reward_delay: int, optional
        Delay in the reward. Default is 0, meaning no delay.
        This can be useful for simulating environments where the reward is not immediate.
    """

    def __init__(self, df: pd.DataFrame, reward_delay: int = 0, random_state=None):
        self.df = df
        self.arms = len(df.iloc[:, -1].unique())  # n_actions
        self.idx_arms = list(np.sort(df.iloc[:, -1].unique()))
        self.nfeats = df.shape[1] - 1  # context_dim
        self.nobs = df.shape[0]
        self.delay = reward_delay
        self.random_state = random_state

        if not np.issubdtype(df.iloc[:, -1].unique().dtype, np.integer):
            raise ValueError(f"Expected labels to be of integer type, but found {df.iloc[:, -1].unique().dtype}.")
        if self.idx_arms != list(range(self.arms)):
            raise ValueError(
                f"Expected arms to be encoded in the range [0, {self.arms - 1}], but found arms: {self.idx_arms}."
            )

        self.reset_env()

    def get_current_context(self):
        """
        Get the current context.

        :return: Current context as a numpy array with shape (1, nfeats).
        """
        return self.df.iloc[self.idx, : self.nfeats].values.reshape(1, -1)

    def take_reward_of_arm(self, action: int) -> int:
        """Compute the reward for a given action.

        Args:
            action (int): The action to compute reward for.

        Returns
        -------
            Tuple[int, int]: Computed reward and associated delay.
        """
        if action not in self.idx_arms:
            raise ValueError(f"Action {action} is not a valid arm. Valid arms are: {self.idx_arms}.")

        label = self.df.iloc[self.idx, -1]
        r = int(label == action)

        self.idx += 1
        if not self.idx < self.nobs:
            self.idx = 0

        return r, self.delay

    def reset_env(self):
        """Reset DataBasedBandit."""
        self.idx = 0
        self.df = self.df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)  # Shuffle the DataFrame
