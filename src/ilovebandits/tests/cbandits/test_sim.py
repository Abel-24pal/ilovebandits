"""Testing submodule sim.py."""

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.ilovebandits.agents import EpsGreedyConAgent
from src.ilovebandits.data_bandits.base import DataBasedBanditFromPandas
from src.ilovebandits.data_bandits.utils import GenrlBanditDataLoader
from src.ilovebandits.sim import NoRewardsReceivedError, NotAbleToUpdateBanditError, SimContBandit, SimMabBandit
from src.ilovebandits.mab.agents import GreedyAgent
from src.ilovebandits.mab.q_estimators import QEstMean

RANDOM_SEED = 42
RANDOM_STATE = 42


@pytest.fixture(scope="module")
def pars_simcontban():
    """Main parameters for the simulation of a contextual bandit problem."""
    return {
        "iterations": 1000,
        "min_ites_to_train": 30,  # minimum number of iterations to start training the agent
        "update_factor": 28,  # if 1, it updates the model every iteration, if 2, it updates every two iterations, etc.
    }


@pytest.fixture(scope="module")
def pars_simmab():
    """Main parameters for the simulation of a contextual bandit problem."""
    return {
        "iterations": 1000,
    }


@pytest.fixture(scope="module")
def dataset_for_sims():
    """Get the dataset for simulations."""
    return GenrlBanditDataLoader().get_statlog_shuttle_data()


@pytest.fixture(
    scope="function", params=[0, 10]
)  # The fixture is run once per test function because the default scope is 'function'.
def mab_delay_sim(request, pars_simmab, dataset_for_sims):
    """Do a simulation of a contextual bandit problem with different reward delays."""
    reward_delay = request.param

    iterations = pars_simmab["iterations"]

    model_env = DataBasedBanditFromPandas(
        df=dataset_for_sims,
        reward_delay=reward_delay,
        random_state=RANDOM_STATE,
    )
    narms = model_env.arms
    qvals_init = [0] * narms  # Initial Q-values for each arm
    qvals_init[0] = 1  # Set the first arm's initial Q-value to 1 for testing purposes
    agent = GreedyAgent(q_estimator=QEstMean(arms=narms, qvals_init=qvals_init))

    simulator = SimMabBandit(
        agent=agent,
        model_env=model_env,
    )

    res = simulator.simulate(iterations=iterations)
    return {
        "simulator": simulator,
        "res": res,
        "iterations": iterations,
        "reward_delay": reward_delay,
    }


@pytest.fixture(
    scope="function", params=[0, 10]
)  # The fixture is run once per test function because the default scope is 'function'.
def cbandit_delay_sim(request, pars_simcontban, dataset_for_sims):
    """Do a simulation of a contextual bandit problem with different reward delays."""
    reward_delay = request.param

    iterations = pars_simcontban["iterations"]
    min_ites_to_train = pars_simcontban["min_ites_to_train"]  # minimum number of iterations to start training the agent
    update_factor = pars_simcontban[
        "update_factor"
    ]  # if 1, it updates the model every iteration, if 2, it updates every two iterations, etc.

    model_env = DataBasedBanditFromPandas(
        df=dataset_for_sims,
        reward_delay=reward_delay,
        random_state=RANDOM_STATE,
    )
    narms = model_env.arms
    agent = EpsGreedyConAgent(
        arms=narms,
        base_estimator=RandomForestClassifier(random_state=RANDOM_STATE),
        n_rounds_random=50,
        epsilon=0.01,  # low to reduce random component
        one_model_per_arm=False,
        rng_seed=RANDOM_SEED,
        min_samples_to_ignore_arm=10,
    )

    simulator = SimContBandit(
        agent=agent,
        model_env=model_env,
        min_ites_to_train=min_ites_to_train,
        update_factor=update_factor,
    )

    res = simulator.simulate(iterations=iterations)
    return {
        "simulator": simulator,
        "res": res,
        "iterations": iterations,
        "reward_delay": reward_delay,
    }


# @pytest.mark.skip(reason="Time-consuming test, skip for now.")
def test_reset_cban_simulator(cbandit_delay_sim):
    """Test the reset functionality of the simulator."""
    simulator = cbandit_delay_sim["simulator"]

    # Before resetting
    assert not np.all(np.isclose(simulator.agent.arm_count, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], rtol=1e-5))
    if simulator.agent.one_model_per_arm:
        assert simulator.agent.models is not None
    else:
        assert simulator.agent.model is not None
    assert simulator.agent.update_agent_counts != 0
    assert simulator.agent.last_action is not None
    assert simulator.agent.qvals != []
    assert simulator.model_env.idx != 0

    #### RESETTING THE SIMULATOR ####
    simulator.reset_agent_and_env()

    # After resetting:
    assert np.all(np.isclose(simulator.agent.arm_count, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], rtol=1e-5))
    assert simulator.agent.model is None
    assert simulator.agent.models is None
    assert simulator.agent.update_agent_counts == 0
    assert simulator.agent.last_action is None
    assert simulator.agent.qvals == []
    assert simulator.model_env.idx == 0


# @pytest.mark.skip(reason="Time-consuming test, skip for now.")
def test_cban_sim(cbandit_delay_sim):
    """Test a cbandirt simulation with 0 delays and with constant delays."""
    res = cbandit_delay_sim["res"]
    iterations = cbandit_delay_sim["iterations"]
    reward_delay = cbandit_delay_sim["reward_delay"]

    ##### General length check #####
    assert len(res["actions"]) == iterations
    assert len(res["prob_actions"]) == iterations
    assert iterations == sum(res["agent"].arm_count)
    assert len(res["reward_heap"]) == reward_delay  # No reward delay, so the heap should be empty

    actions_taken = pd.DataFrame(res["actions"], columns=["arm"])
    actions_taken["ite"] = actions_taken.index + 1  # Adjust index to start from 1. Iterations start at 1.
    rew_agent = pd.DataFrame(res["rew_agent"])

    assert (
        len(rew_agent) == iterations - reward_delay
    )  # This shopuld be true if delays are 0. Rewards that agent sees is equal to number of iterations.
    # this arrays are equal when there are no reward delays
    np.array_equal(
        actions_taken[["ite", "arm"]].values,
        rew_agent[["ite", "arm"]].values,
    )

    ###### Check the arm and prob_arm taken in the last update ######
    ite_train = res["ite_updated"][-1]  # last time agent is updated

    df_train = rew_agent.query("ite<=@ite_train")
    pred_point = rew_agent.query("ite==@ite_train+1")

    c_train = np.array(df_train["context"].to_list())
    a_train = np.array(df_train["arm"].to_list())
    r_train = np.array(df_train["reward"].to_list())
    context2pred = np.array(pred_point["context"].to_list())

    res["agent"].update_agent(c_train=c_train, a_train=a_train, r_train=r_train)

    a_exp, prob_a_exp = res["agent"].take_action(
        context=context2pred,
    )
    a_act = res[
        "actions"
    ][
        ite_train
    ]  # note: position ite_train corresponds to arm taken at iteration "ite_train + 1". This is because iterations start at 1, but array starts at index 0.
    prob_a_act = res[
        "prob_actions"
    ][
        ite_train
    ]  # note: position ite_train corresponds to prob_arm taken at iteration "ite_train + 1". This is because iterations start at 1, but array starts at index 0.

    assert df_train.shape[0] == ite_train - reward_delay  # this should be true if delays are 0
    assert a_exp == a_act
    assert prob_a_act == pytest.approx(prob_a_exp, rel=1e-6, abs=1e-12)

    ###### Check that last ite failed did it due to MIN_SAMPLES_TO_IGNORE_ARM ######
    take_an_ite_failed = res["ite_failed"][-1][0]
    min_arm_samples = (
        pd.DataFrame(res["rew_agent"])
        .query(f"ite<={take_an_ite_failed}")
        .groupby("arm")
        .agg(
            samples_per_arm=("ite", "count"),
        )
        .reset_index()["samples_per_arm"]
        .min()
    )

    assert min_arm_samples < res["agent"].MIN_SAMPLES_TO_IGNORE_ARM


def test_update_bandit_error(dataset_for_sims):
    """Test that the NotAbleToUpdateBanditError is raised when the agent cannot be updated."""
    iterations = 1000
    min_ites_to_train = iterations // 10  # minimum number of iterations to start training the agent
    update_factor = iterations // 10
    reward_delay = iterations // 10
    min_samples_to_ignore_arm = iterations  # Set a value that will force the error

    model_env = DataBasedBanditFromPandas(
        df=dataset_for_sims,
        reward_delay=reward_delay,
        random_state=RANDOM_STATE,
    )
    narms = model_env.arms
    agent = EpsGreedyConAgent(
        arms=narms,
        base_estimator=RandomForestClassifier(random_state=RANDOM_STATE),
        n_rounds_random=min_ites_to_train,
        epsilon=0.01,  # low to reduce random component
        one_model_per_arm=False,
        rng_seed=RANDOM_SEED,
        min_samples_to_ignore_arm=min_samples_to_ignore_arm,
    )

    simulator = SimContBandit(
        agent=agent,
        model_env=model_env,
        min_ites_to_train=min_ites_to_train,
        update_factor=update_factor,
    )

    with pytest.raises(NotAbleToUpdateBanditError):
        simulator.simulate(iterations=iterations)


def test_no_rewards_error(dataset_for_sims):
    """Test that the NoRewardsReceivedError is raised when no rewards are received."""
    iterations = 1000
    min_ites_to_train = iterations // 10  # minimum number of iterations to start training the agent
    update_factor = iterations // 10
    reward_delay = iterations + 1  # Set a reward delay longer than the number of iterations to force the error

    model_env = DataBasedBanditFromPandas(
        df=dataset_for_sims,
        reward_delay=reward_delay,
        random_state=RANDOM_STATE,
    )
    narms = model_env.arms
    agent = EpsGreedyConAgent(
        arms=narms,
        base_estimator=RandomForestClassifier(random_state=RANDOM_STATE),
        n_rounds_random=min_ites_to_train,
        epsilon=0.01,  # low to reduce random component
        one_model_per_arm=False,
        rng_seed=RANDOM_SEED,
    )

    simulator = SimContBandit(
        agent=agent,
        model_env=model_env,
        min_ites_to_train=min_ites_to_train,
        update_factor=update_factor,
    )

    with pytest.raises(NoRewardsReceivedError):
        simulator.simulate(iterations=iterations)


def test_reset_mab_simulator(mab_delay_sim):
    """Test the reset functionality of the simulator."""
    simulator = mab_delay_sim["simulator"]

    # Before resetting
    assert not np.all(np.isclose(simulator.agent.arm_count, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], rtol=1e-5))
    assert not np.all(
        np.isclose(simulator.agent.q_estimator.arm_count_updates, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], rtol=1e-5)
    )
    assert simulator.agent.last_action is not None
    assert simulator.model_env.idx != 0

    #### RESETTING THE SIMULATOR ####
    simulator.reset_agent_and_env()

    # After resetting:
    assert np.all(np.isclose(simulator.agent.arm_count, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], rtol=1e-5))
    assert np.all(
        np.isclose(simulator.agent.q_estimator.arm_count_updates, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], rtol=1e-5)
    )
    assert simulator.agent.last_action is None
    assert simulator.model_env.idx == 0


def test_mab_sim(mab_delay_sim):
    """Test a mab simulation with 0 delays and with constant delays."""
    res = mab_delay_sim["res"]
    iterations = mab_delay_sim["iterations"]
    reward_delay = mab_delay_sim["reward_delay"]

    ##### General length check #####
    assert len(res["actions"]) == iterations
    assert len(res["prob_actions"]) == iterations
    assert iterations == sum(res["agent"].arm_count)
    assert (iterations - reward_delay) == sum(res["agent"].q_estimator.arm_count_updates)
    assert len(res["reward_heap"]) == reward_delay  # No reward delay, so the heap should be empty

    actions_taken = pd.DataFrame(res["actions"], columns=["arm"])
    actions_taken["ite"] = actions_taken.index + 1  # Adjust index to start from 1. Iterations start at 1.
    rew_agent = pd.DataFrame(res["rew_agent"])

    assert (
        len(rew_agent) == iterations - reward_delay
    )  # This shopuld be true if delays are 0. Rewards that agent sees is equal to number of iterations.
    # this arrays are equal when there are no reward delays
    np.array_equal(
        actions_taken[["ite", "arm"]].values,
        rew_agent[["ite", "arm"]].values,
    )

    exp_qvals = [rew_agent.query("arm==@arm")["reward"].mean() for arm in range(res["agent"].arms)]
    exp_qvals = [0 if np.isnan(x) else x for x in exp_qvals]  # if action does not exist, it will be NaN
    npt.assert_allclose(actual=res["qvals"][-1], desired=exp_qvals, rtol=1e-7, atol=1e-8)
