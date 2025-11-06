from typing import List


class MismatchedArmNumberError(Exception):
    """Exception raised when the number of arms in the agent does not match the number of arms in the training data."""

    def __init__(self, expected_arms: int, actual_arms: int):
        super().__init__(
            f"Expected {expected_arms} arms, but found {actual_arms} unique arms in a_train."
        )
        self.expected = expected_arms
        self.actual = actual_arms


class NotEnoughRewardsPerArmError(Exception):
    """Exception raised when there are not enough unique rewards per arm in the training data. We want at least 2 different rewards per arm."""

    def __init__(self, unique_rewards_per_arm: List[int]):
        super().__init__(
            f"There are arms that have less than two unique reards in the training data provided. Here, the unique rewards per arm for each arm:  {unique_rewards_per_arm}."
        )
        self.unique_rewards_per_arm = unique_rewards_per_arm


class AgentNotFullyUpdatedError(Exception):
    """Agent needs to be updated with update_agent() method at least once."""

    pass
