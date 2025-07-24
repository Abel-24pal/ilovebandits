"""Classes to perform simulations."""

import collections
import heapq
from copy import deepcopy
from typing import Tuple

import numpy as np

from .agents import MismatchedArmNumberError, NotEnoughRewardsPerArmError


class NotAbleToUpdateBanditError(Exception):
    """Exception raised when it was not possible to update the bandit during the whole simulation."""

    def __init__(self, info_last_ite_failed: Tuple):
        super().__init__(
            f"Bandit not updated in the whole simulation. Last time the update failed was (ite, exception) = {info_last_ite_failed}."
        )
        self.info_last_ite_failed = info_last_ite_failed


class NoRewardsReceivedError(Exception):
    """Exception raised when no rewards were received during the simulation."""

    def __init__(self):
        super().__init__("No rewards were received during the simulation.")


class SimContBandit:
    """Perform a simulation with delays for contextual bandits."""

    def __init__(self, min_ites_to_train: int, update_factor: int, agent, model_env):
        self.min_ites_to_train = min_ites_to_train
        self.update_factor = update_factor
        self.agent = deepcopy(agent)
        self.model_env = deepcopy(model_env)

    def reset_agent_and_env(self):
        """Reset agent and environment."""
        self.agent.reset_agent()
        self.model_env.reset_env()

    def simulate(self, iterations: int = 1000):
        """
        Perform the simulation for the given number of iterations.

        Parameters
        ----------
        iterations: int
            number of iterations for the simulation.

        """
        self.res_hist = {}

        self.res_hist["rew_agent"] = []
        self.res_hist["actions"] = []
        self.res_hist["prob_actions"] = []
        self.res_hist["ite_updated"] = []
        self.res_hist["ite_failed"] = []

        reward_heap = []
        RI = collections.namedtuple("RI", "ite arm context reward")  # Reward Info tuple
        lc_train, la_train, lr_train = [], [], []

        for ite in range(1, iterations + 1):
            context = self.model_env.get_current_context()
            a_idx, a_prob = self.agent.take_action(context=context)
            r, r_delay = self.model_env.take_reward_of_arm(a_idx)

            # print(f"\n--> Iteration {ite}, context: {context}, action: {a_idx}, a_prob: {a_prob}, reward: {r}, reward delay: {r_delay}")

            heapq.heappush(
                reward_heap,
                RI(
                    ite=ite + r_delay, arm=a_idx, context=context.ravel().tolist(), reward=r
                ),  # better save context as a list instead of np.array
            )

            while reward_heap and reward_heap[0].ite <= ite:
                rinfo = heapq.heappop(reward_heap)
                lc_train.append(rinfo.context)
                la_train.append(rinfo.arm)
                lr_train.append(rinfo.reward)

                self.res_hist["rew_agent"].append(rinfo)

            try:
                if (ite > self.min_ites_to_train) and (ite % self.update_factor == 0) and (len(lc_train) > 0):
                    # print(f"\n--> Iteration {ite}, updating agent with {len(lc_train)} samples")

                    c_train = np.array(lc_train)
                    a_train = np.array(la_train)
                    r_train = np.array(lr_train)

                    self.agent.update_agent(c_train=c_train, a_train=a_train, r_train=r_train)

                    self.res_hist["ite_updated"].append((ite))

            except (NotEnoughRewardsPerArmError, MismatchedArmNumberError) as e:
                # print(f"Failed to update agent at iteration {ite}: {e}")
                self.res_hist["ite_failed"].append((ite, str(e)))

            self.res_hist["actions"].append(a_idx)
            self.res_hist["prob_actions"].append(a_prob)

        self.res_hist["agent"] = deepcopy(self.agent)  # do a deep copy of the class
        self.res_hist["model_env"] = deepcopy(self.model_env)  # do a deep copy of the class
        self.res_hist["reward_heap"] = reward_heap

        print(f"Agent updated in iterations: {self.res_hist['ite_updated']}")

        if len(self.res_hist["rew_agent"]) < 1:
            raise NoRewardsReceivedError()

        if len(self.res_hist["ite_updated"]) < 1:
            raise NotAbleToUpdateBanditError(info_last_ite_failed=self.res_hist["ite_failed"][-1])

        return self.res_hist


class SimMabBandit:
    """Perform a simulation with delays for MAB bandits. Environment should have binary rewards in 0-1."""

    def __init__(self, agent, model_env):
        self.agent = deepcopy(agent)
        self.model_env = deepcopy(model_env)

    def reset_agent_and_env(self):
        """Reset agent and environment."""
        self.agent.reset_agent()
        self.model_env.reset_env()

    def simulate(self, iterations: int = 1000):
        """
        Perform the simulation for the given number of iterations.

        Parameters
        ----------
        iterations: int
            number of iterations for the simulation.

        """
        self.res_hist = {}

        self.res_hist["rew_agent"] = []
        self.res_hist["actions"] = []
        self.res_hist["prob_actions"] = []
        self.res_hist["qvals"] = []
        self.res_hist["qvals"].append(self.agent.initial_q_estimator.qvals)

        reward_heap = []
        RI = collections.namedtuple("RI", "ite arm context reward")  # Reward Info tuple

        for ite in range(1, iterations + 1):
            context = self.model_env.get_current_context()
            a_idx, _, a_prob = self.agent.take_action()
            r, r_delay = self.model_env.take_reward_of_arm(a_idx)

            # print(f"\n--> Iteration {ite}, context: {context}, action: {a_idx}, a_prob: {a_prob}, reward: {r}, reward delay: {r_delay}")

            heapq.heappush(
                reward_heap,
                RI(
                    ite=ite + r_delay, arm=a_idx, context=context.ravel().tolist(), reward=r
                ),  # better save context as a list instead of np.array
            )

            while reward_heap and reward_heap[0].ite <= ite:
                rinfo = heapq.heappop(reward_heap)
                self.agent.q_estimator.estimate(reward=rinfo.reward, action=rinfo.arm)
                self.res_hist["rew_agent"].append(rinfo)

            # As this is outside the inners for-while loops, res_hist["qvals"] represents the qvals after all the updates of the iteration. If we want to represent the evolution of all the partial updates, this should be changed.
            self.res_hist["qvals"].append(deepcopy(self.agent.q_estimator.qvals))
            self.res_hist["actions"].append(a_idx)
            self.res_hist["prob_actions"].append(a_prob)

        self.res_hist["agent"] = deepcopy(self.agent)  # do a deep copy of the class
        self.res_hist["model_env"] = deepcopy(self.model_env)  # do a deep copy of the class
        self.res_hist["reward_heap"] = reward_heap

        return self.res_hist
