"""Utils functions for the package."""

from typing import List, Tuple

import numpy as np


# Arg max that solves ties randomly:
def argmax(q_values: List) -> Tuple[int, float, List[int]]:
    """
    Takes in a list of q_values and returns the index of the item with the highest value. Breaks ties randomly.

    Returns
    -------
    int - the index of the highest value in q_values.
    float - the probability of selecting the action.
    list[int] - the list of indices that are tied for the highest value.
    """
    top_value = float("-inf")
    ties = []

    for i in range(len(q_values)):
        # if a value in q_values is greater than the highest value update top and reset ties to zero
        # if a value is equal to top value add the index to ties
        # return a random selection from ties.
        if q_values[i] > top_value:
            top_value = q_values[i]
            ties = []
            ties.append(i)

        elif q_values[i] == top_value:
            ties.append(i)

    prob = 1 / len(ties)  # probability of selecting each tied action

    return np.random.choice(ties), prob, ties


def find_max_numbers(numbers: List) -> List:
    """Returns a list with the max number. In case of tie, the tied numbers are returned."""
    if not numbers:
        return []

    max_value = max(numbers)
    max_numbers = [num for num in numbers if num == max_value]

    return max_numbers


def find_max_indices(numbers: List) -> List:
    """Returns a list with the index of the max number. In case of tie, the indexs of the tied numbers are returned."""
    if not numbers:
        return []

    max_value = max(numbers)
    max_indices = [index for index, value in enumerate(numbers) if value == max_value]

    return max_indices


def ucb_uncertainty(c: float, arm_count_updates: List[int]) -> np.ndarray:
    """Compute uncertainty given a c value and an arm_count vector.

    Arguments
    ----------
    c : c value for UCB.
    arm_count_update: List of arm counts.

    Returns
    -------
    np.ndarray : uncertainty vector.
    """
    step = sum(arm_count_updates) + 1
    uncertainty = np.array(
        [c * np.sqrt(np.log(step) / arm_count) if arm_count > 0 else float("inf") for arm_count in arm_count_updates]
    )

    return uncertainty
