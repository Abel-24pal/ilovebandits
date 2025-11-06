"""Testing submodule agents.py."""

import numpy as np
import numpy.testing as npt
import pytest
from scipy.special import expit

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from src.ilovebandits.agents import (
    EpsGreedyConAgent,
    MismatchedArmNumberError,
    NotEnoughRewardsPerArmError,
    RandomForestTsAgent,
    RandomForestUcbAgent,
)
from src.ilovebandits.utils import argmax
from src.ilovebandits.exceptions import AgentNotFullyUpdatedError
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone

RANDOM_SEED = 42

dic_eps_pars = {
    "eps_pars_disjoint": {
        "n_rounds_random": 5,
        "epsilon": 0.1,
        "one_model_per_arm": True,
        "base_estimator": LinearRegression(),
    },
    "eps_pars_hybrid": {
        "n_rounds_random": 5,
        "epsilon": 0.1,
        "one_model_per_arm": False,
        "base_estimator": LinearRegression(),
    },
    "eps_pars_hybrid_clf": {
        "n_rounds_random": 5,
        "epsilon": 0.1,
        "one_model_per_arm": False,
        "base_estimator": LogisticRegression(
            penalty=None, fit_intercept=False, solver="lbfgs"
        ),
    },
}

pars_agents_err = {
    "rfts": RandomForestTsAgent(
        arms=4,
        vpar=1.0,
        base_model=RandomForestRegressor(max_samples=None, random_state=RANDOM_SEED),
        rng_seed=42,
        one_model_per_arm=False,
        samples_for_freq_est=1,
    ),
    "rfts_disjoint": RandomForestTsAgent(
        arms=4,
        vpar=1.0,
        base_model=RandomForestRegressor(max_samples=None, random_state=RANDOM_SEED),
        rng_seed=42,
        one_model_per_arm=True,
        samples_for_freq_est=1,
    ),
    "rfucb": RandomForestUcbAgent(
        arms=4,
        vpar=1.0,
        base_model=RandomForestRegressor(max_samples=None, random_state=RANDOM_SEED),
        rng_seed=42,
        one_model_per_arm=False,
    ),
    "rfucb_disjoint": RandomForestUcbAgent(
        arms=4,
        vpar=1.0,
        base_model=RandomForestRegressor(max_samples=None, random_state=RANDOM_SEED),
        rng_seed=42,
        one_model_per_arm=True,
    ),
}


@pytest.fixture(scope="module")
def eps_pars_disjoint():
    """Return the parameters for the disjoint epsilon-greedy agents (one model per arm). Regression problem."""
    return dic_eps_pars["eps_pars_disjoint"]


@pytest.fixture(scope="module")
def eps_pars_hybrid():
    """Return the parameters for the hybrid epsilon-greedy agent (one model for all arms). Regression problem."""
    return dic_eps_pars["eps_pars_hybrid"]


@pytest.fixture(scope="module")
def eps_pars_hybrid_clf():
    """Return the parameters for the hybrid epsilon-greedy agent (one model for all arms). Classifier problem."""
    return dic_eps_pars["eps_pars_hybrid_clf"]


@pytest.mark.parametrize(
    "eps_pars,arms,feats",
    [
        (dic_eps_pars["eps_pars_disjoint"], 3, 4),
        (dic_eps_pars["eps_pars_hybrid"], 3, 4),
    ],
)
def test_eps_agent_initialization(eps_pars, arms, feats):
    """Test GreedyAgent."""
    ####### CRETAE AGENT ########

    eps_agent = EpsGreedyConAgent(
        arms=arms,
        base_estimator=eps_pars["base_estimator"],
        n_rounds_random=eps_pars["n_rounds_random"],
        epsilon=eps_pars["epsilon"],
        one_model_per_arm=eps_pars["one_model_per_arm"],
        rng_seed=RANDOM_SEED,
    )

    # Test Initial Agent Attributes
    assert eps_agent.arms == arms
    assert eps_agent.idx_arms == list(range(arms))
    assert eps_agent.n_rounds_random == eps_pars["n_rounds_random"]
    assert eps_agent.epsilon == eps_pars["epsilon"]
    assert eps_agent.one_model_per_arm == eps_pars["one_model_per_arm"]
    assert eps_agent.model is None
    assert eps_agent.models is None
    assert eps_agent.update_agent_counts == 0

    assert eps_agent.arm_count == [0.0 for _ in range(eps_agent.arms)]
    assert eps_agent.last_action is None
    ####### END --- CREATE AGENT ########
    dummy_context = np.ones((1, feats))

    a1, p1 = eps_agent.take_action(context=dummy_context)
    a2, p2 = eps_agent.take_action(context=dummy_context)
    a3, p3 = eps_agent.take_action(context=dummy_context)
    a4, p4 = eps_agent.take_action(context=dummy_context)
    a5, p5 = eps_agent.take_action(context=dummy_context)

    manual_arm_count = [0 for _ in range(arms)]
    for ai in [a1, a2, a3, a4, a5]:
        manual_arm_count[ai] += 1

    # check prob estimation of random action
    npt.assert_allclose(
        actual=1 / arms * np.ones(5), desired=[p1, p2, p3, p4, p5], rtol=1e-7, atol=1e-8
    )

    # check arm count logic
    npt.assert_array_equal(
        np.array(eps_agent.arm_count),
        np.array(manual_arm_count),
    )
    # check warning if agent is nor fitted and needs to be updated due to agent exceeding n_rounds_random
    with pytest.warns(UserWarning, match="RANDOM ACTION:") as w:  # noqa: F841
        _, _ = eps_agent.take_action(context=dummy_context)


def test_eps_disjoint_with_linear_regressor(data_disjoint, eps_pars_disjoint):
    """Test GreedyAgent."""
    c_train = data_disjoint["c_train"]
    a_train = data_disjoint["a_train"]
    r_train = data_disjoint["r_train"]
    feats = data_disjoint["feats"]
    arms = data_disjoint["arms"]
    coefs = data_disjoint["coefs"]

    ####### CREATE AGENT ########
    eps_agent = EpsGreedyConAgent(
        arms=arms,
        base_estimator=eps_pars_disjoint["base_estimator"],
        n_rounds_random=eps_pars_disjoint["n_rounds_random"],
        epsilon=eps_pars_disjoint["epsilon"],
        one_model_per_arm=eps_pars_disjoint["one_model_per_arm"],
        rng_seed=RANDOM_SEED,
    )
    ####### END --- CREATE AGENT ########

    dummy_context = np.ones((1, feats.shape[1]))
    expr_dummy = np.array(
        [np.dot(dummy_context, np.array(coef)) for coef in coefs]
    )  # expected reward of the dummy context for each arm

    a1, _ = eps_agent.take_action(context=dummy_context)
    a2, _ = eps_agent.take_action(context=dummy_context)
    a3, _ = eps_agent.take_action(context=dummy_context)
    a4, _ = eps_agent.take_action(context=dummy_context)
    a5, _ = eps_agent.take_action(context=dummy_context)

    # check seed produces the expected random behaviour
    assert a1 == 0
    assert a2 == 3
    assert a3 == 2
    assert a4 == 1
    assert a5 == 1

    ########### FIT AGENT ##########
    eps_agent.update_agent(c_train=c_train, a_train=a_train, r_train=r_train)

    # check model attributes after the fit:
    assert eps_agent.update_agent_counts == 1
    assert eps_agent.model is None
    assert len(eps_agent.models) == arms
    assert eps_agent.nfeats == feats.shape[1]
    npt.assert_allclose(
        actual=c_train, desired=eps_agent.last_c_train, rtol=1e-7, atol=1e-8
    )
    npt.assert_allclose(
        actual=a_train, desired=eps_agent.last_a_train, rtol=1e-7, atol=1e-8
    )
    npt.assert_allclose(
        actual=r_train, desired=eps_agent.last_r_train, rtol=1e-7, atol=1e-8
    )

    # check coefficients of the linear regression models
    for model, coef in zip(eps_agent.models, coefs, strict=False):
        npt.assert_allclose(actual=model.coef_, desired=coef, rtol=1e-7, atol=1e-8)
    ########### END --- FIT AGENT ##########

    ########### PREDICT AGENT ##########
    (
        a7,
        p7,
    ) = eps_agent.take_action(context=dummy_context)

    assert np.argmax(expr_dummy) == a7
    npt.assert_allclose(
        actual=eps_agent.qvals,
        desired=expr_dummy,
        rtol=1e-7,
        atol=1e-8,
    )

    for _ in range(10000):
        _, _ = eps_agent.take_action(context=dummy_context)

    expected_proba_greedy = 1 - eps_agent.epsilon + eps_agent.epsilon / eps_agent.arms
    assert (
        pytest.approx(
            eps_agent.arm_count[np.argmax(expr_dummy)] / sum(eps_agent.arm_count),
            abs=0.01,
        )
        == expected_proba_greedy
    )
    assert pytest.approx(p7, abs=0.01) == expected_proba_greedy
    ########### END --- PREDICT AGENT ##########


def test_eps_hybrid_with_linear_regressor(data_hybrid, eps_pars_hybrid):
    """Test GreedyAgent."""
    c_train = data_hybrid["c_train"]
    a_train = data_hybrid["a_train"]
    r_train = data_hybrid["r_train"]
    arms = data_hybrid["arms"]
    coefs = data_hybrid["coefs"]

    ####### CREATE AGENT ########
    eps_agent = EpsGreedyConAgent(
        arms=arms,
        base_estimator=eps_pars_hybrid["base_estimator"],
        n_rounds_random=eps_pars_hybrid["n_rounds_random"],
        epsilon=eps_pars_hybrid["epsilon"],
        one_model_per_arm=eps_pars_hybrid["one_model_per_arm"],
        rng_seed=RANDOM_SEED,
    )
    ####### END --- CREATE AGENT ########

    dummy_context = np.ones((1, c_train.shape[1]))
    expr_dummy = []
    for arm in range(arms):
        arm_context = np.hstack((dummy_context, np.array([[arm]])))
        expr_dummy.append(np.dot(arm_context, np.array(coefs)))
    expr_dummy = np.array(
        expr_dummy
    )  # expected reward of the dummy context for each arm

    a1, _ = eps_agent.take_action(context=dummy_context)
    a2, _ = eps_agent.take_action(context=dummy_context)
    a3, _ = eps_agent.take_action(context=dummy_context)
    a4, _ = eps_agent.take_action(context=dummy_context)
    a5, _ = eps_agent.take_action(context=dummy_context)

    # check seed produces the expected random behaviour
    assert a1 == 0
    assert a2 == 3
    assert a3 == 2
    assert a4 == 1
    assert a5 == 1

    ########### FIT AGENT ##########
    eps_agent.update_agent(c_train=c_train, a_train=a_train, r_train=r_train)

    # check model attributes after the fit:
    assert eps_agent.update_agent_counts == 1
    assert eps_agent.models is None
    assert eps_agent.model is not None
    assert eps_agent.nfeats == c_train.shape[1]
    npt.assert_allclose(
        actual=c_train, desired=eps_agent.last_c_train, rtol=1e-7, atol=1e-8
    )
    npt.assert_allclose(
        actual=a_train, desired=eps_agent.last_a_train, rtol=1e-7, atol=1e-8
    )
    npt.assert_allclose(
        actual=r_train, desired=eps_agent.last_r_train, rtol=1e-7, atol=1e-8
    )

    # check coefficients of the linear regression model
    npt.assert_allclose(
        actual=eps_agent.model.coef_, desired=coefs, rtol=1e-7, atol=1e-8
    )
    ########### END --- FIT AGENT ##########

    ########### PREDICT AGENT ##########
    (
        a7,
        p7,
    ) = eps_agent.take_action(context=dummy_context)

    assert np.argmax(expr_dummy) == a7
    npt.assert_allclose(
        actual=eps_agent.qvals,
        desired=expr_dummy,
        rtol=1e-7,
        atol=1e-8,
    )

    for _ in range(10000):
        _, _ = eps_agent.take_action(context=dummy_context)

    expected_proba_greedy = 1 - eps_agent.epsilon + eps_agent.epsilon / eps_agent.arms
    assert (
        pytest.approx(
            eps_agent.arm_count[np.argmax(expr_dummy)] / sum(eps_agent.arm_count),
            abs=0.01,
        )
        == expected_proba_greedy
    )
    assert pytest.approx(p7, abs=0.01) == expected_proba_greedy
    # ########### END --- PREDICT AGENT ##########


def test_eps_hybrid_clf(data_hybrid_clf, eps_pars_hybrid_clf):
    """Test GreedyAgent."""
    c_train = data_hybrid_clf["c_train"]
    a_train = data_hybrid_clf["a_train"]
    r_train = data_hybrid_clf["r_train"]
    arms = data_hybrid_clf["arms"]
    coefs = data_hybrid_clf["coefs"]

    ####### CREATE AGENT ########
    eps_agent = EpsGreedyConAgent(
        arms=arms,
        base_estimator=eps_pars_hybrid_clf["base_estimator"],
        n_rounds_random=eps_pars_hybrid_clf["n_rounds_random"],
        epsilon=eps_pars_hybrid_clf["epsilon"],
        one_model_per_arm=eps_pars_hybrid_clf["one_model_per_arm"],
        rng_seed=RANDOM_SEED,
    )
    ####### END --- CREATE AGENT ########

    dummy_context = np.ones((1, c_train.shape[1]))
    expr_dummy = []
    for arm in range(arms):
        arm_context = np.hstack((dummy_context, np.array([[arm]])))
        expr_dummy.append(expit(np.dot(arm_context, np.array(coefs))))
    expr_dummy = np.array(
        expr_dummy
    )  # expected reward of the dummy context for each arm

    _, _ = eps_agent.take_action(context=dummy_context)
    _, _ = eps_agent.take_action(context=dummy_context)
    _, _ = eps_agent.take_action(context=dummy_context)
    _, _ = eps_agent.take_action(context=dummy_context)
    _, _ = eps_agent.take_action(context=dummy_context)

    ########### FIT AGENT ##########
    eps_agent.update_agent(c_train=c_train, a_train=a_train, r_train=r_train)

    # check model attributes after the fit:
    assert eps_agent.update_agent_counts == 1
    assert eps_agent.models is None
    assert eps_agent.model is not None
    assert eps_agent.nfeats == c_train.shape[1]
    npt.assert_allclose(
        actual=c_train, desired=eps_agent.last_c_train, rtol=1e-7, atol=1e-8
    )
    npt.assert_allclose(
        actual=a_train, desired=eps_agent.last_a_train, rtol=1e-7, atol=1e-8
    )
    npt.assert_allclose(
        actual=r_train, desired=eps_agent.last_r_train, rtol=1e-7, atol=1e-8
    )

    # check coefficients of the linear regression model
    npt.assert_allclose(
        actual=eps_agent.model.coef_,
        desired=np.array([coefs]),
        rtol=0.01,
        atol=0,
    )
    ########### END --- FIT AGENT ##########

    ########### PREDICT AGENT ##########
    (
        a7,
        p7,
    ) = eps_agent.take_action(context=dummy_context)

    # import pdb; pdb.set_trace()
    assert np.argmax(expr_dummy) == a7
    npt.assert_allclose(
        actual=eps_agent.qvals,
        desired=expr_dummy,
        rtol=1e-4,
        atol=1e-4,
    )

    for _ in range(10000):
        _, _ = eps_agent.take_action(context=dummy_context)

    expected_proba_greedy = 1 - eps_agent.epsilon + eps_agent.epsilon / eps_agent.arms
    assert (
        pytest.approx(
            eps_agent.arm_count[np.argmax(expr_dummy)] / sum(eps_agent.arm_count),
            abs=0.01,
        )
        == expected_proba_greedy
    )
    assert pytest.approx(p7, abs=0.01) == expected_proba_greedy
    # ########### END --- PREDICT AGENT ##########


def test_mismatched_arm_number_error(data_hybrid, eps_pars_hybrid):
    """Test GreedyAgent."""
    c_train = data_hybrid["c_train"]
    a_train = data_hybrid["a_train"]
    r_train = data_hybrid["r_train"]
    arms = data_hybrid["arms"]

    arms = arms - 1  # intentionally set to arms - 1 to trigger error

    ####### CREATE AGENT ########
    eps_agent = EpsGreedyConAgent(
        arms=arms,
        base_estimator=eps_pars_hybrid["base_estimator"],
        n_rounds_random=eps_pars_hybrid["n_rounds_random"],
        epsilon=eps_pars_hybrid["epsilon"],
        one_model_per_arm=eps_pars_hybrid["one_model_per_arm"],
        rng_seed=RANDOM_SEED,
    )

    rfucb_agent = RandomForestUcbAgent(
        arms=arms,
        vpar=1.0,
        base_model=RandomForestClassifier(
            criterion="log_loss",
            min_samples_leaf=20,
            max_samples=None,  # very important _set_train_data_per_tree logic is created based on this assumption
            max_depth=3,
            random_state=42,
        ),
        rng_seed=RANDOM_SEED,
        one_model_per_arm=eps_pars_hybrid["one_model_per_arm"],
    )

    rfts_agent = RandomForestTsAgent(
        arms=arms,
        vpar=1.0,
        base_model=RandomForestRegressor(
            criterion="squared_error",
            min_samples_leaf=20,
            max_samples=None,  # very important _set_train_data_per_tree logic is created based on this assumption
            max_depth=3,
            random_state=42,
        ),
        rng_seed=RANDOM_SEED,
        one_model_per_arm=eps_pars_hybrid["one_model_per_arm"],
        samples_for_freq_est=1,
    )

    ########### FIT AGENT ##########
    list_agents = [eps_agent, rfucb_agent, rfts_agent]
    for agent in list_agents:
        with pytest.raises(MismatchedArmNumberError):
            agent.update_agent(c_train=c_train, a_train=a_train, r_train=r_train)


def test_not_enough_rewards_per_arm_error(data_hybrid, eps_pars_hybrid):
    """Test GreedyAgent."""
    c_train = data_hybrid["c_train"]
    a_train = data_hybrid["a_train"]
    r_train = data_hybrid["r_train"]
    arms = data_hybrid["arms"]

    # Reduce the number of samples to trigger the error
    n_samples = c_train.shape[0]
    c_train = c_train[: n_samples // 2]  # reduce context samples
    a_train = a_train[: n_samples // 2]  # reduce arm samples
    r_train = r_train[: n_samples // 2]  # reduce reward samples

    ####### CREATE AGENTS ########
    eps_agent = EpsGreedyConAgent(
        arms=arms,
        base_estimator=eps_pars_hybrid["base_estimator"],
        n_rounds_random=eps_pars_hybrid["n_rounds_random"],
        epsilon=eps_pars_hybrid["epsilon"],
        one_model_per_arm=eps_pars_hybrid["one_model_per_arm"],
        rng_seed=RANDOM_SEED,
    )

    rfucb_agent = RandomForestUcbAgent(
        arms=arms,
        vpar=1.0,
        base_model=RandomForestClassifier(
            criterion="log_loss",
            min_samples_leaf=20,
            max_samples=None,  # very important _set_train_data_per_tree logic is created based on this assumption
            max_depth=3,
            random_state=42,
        ),
        rng_seed=RANDOM_SEED,
        one_model_per_arm=eps_pars_hybrid["one_model_per_arm"],
    )
    rfts_agent = RandomForestTsAgent(
        arms=arms,
        vpar=1.0,
        base_model=RandomForestRegressor(
            criterion="squared_error",
            min_samples_leaf=20,
            max_samples=None,  # very important _set_train_data_per_tree logic is created based on this assumption
            max_depth=3,
            random_state=42,
        ),
        rng_seed=RANDOM_SEED,
        one_model_per_arm=eps_pars_hybrid["one_model_per_arm"],
        samples_for_freq_est=1,
    )

    ########### FIT AGENT ##########
    list_agents = [eps_agent, rfucb_agent, rfts_agent]
    for agent in list_agents:
        with pytest.raises(NotEnoughRewardsPerArmError):
            agent.update_agent(c_train=c_train, a_train=a_train, r_train=r_train)


@pytest.mark.parametrize(
    "agent",
    (
        pars_agents_err["rfts"],
        pars_agents_err["rfts_disjoint"],
        pars_agents_err["rfucb"],
        pars_agents_err["rfucb_disjoint"],
    ),
)
def test_agent_notfully_updated_error(agent, get_regression_data):
    c_train = get_regression_data["c_train"]
    y_train = get_regression_data["y_train"]
    a_train = get_regression_data["a_train"]
    with pytest.raises(AgentNotFullyUpdatedError):
        agent.partial_fast_update(c_train=c_train, a_train=a_train, r_train=y_train)
        agent.partial_fast_update(c_train=c_train, a_train=a_train, r_train=y_train)


@pytest.mark.parametrize(
    "agent",
    (
        pars_agents_err["rfts"],
        pars_agents_err["rfts_disjoint"],
        pars_agents_err["rfucb"],
        pars_agents_err["rfucb_disjoint"],
    ),
)
def test_agent_fitted(agent, get_regression_data):
    c_train = get_regression_data["c_train"]
    a_train = get_regression_data["a_train"]
    y_train = get_regression_data["y_train"]

    agent.update_agent(c_train=c_train, a_train=a_train, r_train=y_train)
    with pytest.raises(NotFittedError):
        check_is_fitted(agent.base_estimator)
        check_is_fitted(agent.base_estimator.base_model)

    if not agent.one_model_per_arm:
        check_is_fitted(agent.model.model_)
        check_is_fitted(agent.model)
    else:
        for model in agent.models:
            check_is_fitted(model.model_)
            check_is_fitted(model)


#### NOTE: maybe "test_agent_hybrid_and_disjoint_base_model_preds" is not needed as I have also the test "test_agent_hybrid_and_disjoint_handler_preds".
@pytest.mark.parametrize(
    "agent",
    (
        pars_agents_err["rfts"],
        pars_agents_err["rfts_disjoint"],
        pars_agents_err["rfucb"],
        pars_agents_err["rfucb_disjoint"],
    ),
)
def test_agent_hybrid_and_disjoint_base_model_preds(agent, get_regression_data):
    x_train = get_regression_data["x_train"]
    c_train = get_regression_data["c_train"]
    a_train = get_regression_data["a_train"]
    y_train = get_regression_data["y_train"]
    x_test = get_regression_data["x_test"]
    c_test = get_regression_data["c_test"]
    a_test = get_regression_data["a_test"]

    rf_model = clone(agent.base_estimator.base_model)
    agent.update_agent(c_train=c_train, a_train=a_train, r_train=y_train)

    if not agent.one_model_per_arm:
        preds_ilove = agent.model.predict(x_test)[0]
        preds_ref = rf_model.fit(x_train, y_train).predict(x_test)
        assert np.allclose(preds_ilove, preds_ref)
    else:
        for arm in agent.idx_arms:
            preds_ilove = agent.models[arm].predict(c_test[a_test == arm])[0]
            preds_ref = rf_model.fit(
                c_train[a_train == arm], y_train[a_train == arm]
            ).predict(c_test[a_test == arm])
            assert np.allclose(preds_ilove, preds_ref)


@pytest.mark.parametrize(
    "agent",
    (
        pars_agents_err["rfts"],
        pars_agents_err["rfts_disjoint"],
        pars_agents_err["rfucb"],
        pars_agents_err["rfucb_disjoint"],
    ),
)
def test_agent_hybrid_and_disjoint_handler_preds(agent, get_regression_data):
    x_train = get_regression_data["x_train"]
    c_train = get_regression_data["c_train"]
    a_train = get_regression_data["a_train"]
    y_train = get_regression_data["y_train"]
    x_test = get_regression_data["x_test"]
    c_test = get_regression_data["c_test"]
    a_test = get_regression_data["a_test"]

    rf_handler = clone(agent.base_estimator)
    agent.update_agent(c_train=c_train, a_train=a_train, r_train=y_train)

    if not agent.one_model_per_arm:
        preds_ilove = agent.model.predict(x_test)[0]
        preds_ref = rf_handler.fit(x_train, y_train).predict(x_test)[0]

        c_check = c_test[0, :].reshape(1, -1)
        agent.estimate_means_vars(c_check)
        ref_means = [
            rf_handler.predict(np.concatenate((c_check, np.array([[arm]])), axis=1))[0]
            for arm in agent.idx_arms
        ]
        ref_var = [
            rf_handler.predict(np.concatenate((c_check, np.array([[arm]])), axis=1))[1]
            for arm in agent.idx_arms
        ]
        ref_count = [
            rf_handler.predict(np.concatenate((c_check, np.array([[arm]])), axis=1))[2]
            for arm in agent.idx_arms
        ]

        assert np.allclose(preds_ilove, preds_ref)
        assert np.allclose(agent.means, ref_means)
        assert np.allclose(agent.vars, ref_var)
        assert np.allclose(agent.counts, ref_count)
    else:
        c_check = c_test[0, :].reshape(1, -1)
        agent.estimate_means_vars(c_check)

        for arm in agent.idx_arms:
            rf_handler.fit(c_train[a_train == arm], y_train[a_train == arm])

            preds_ilove = agent.models[arm].predict(c_test[a_test == arm])[0]
            preds_ref = rf_handler.predict(c_test[a_test == arm])[0]

            assert np.allclose(agent.means[arm], rf_handler.predict(c_check)[0])
            assert np.allclose(agent.vars[arm], rf_handler.predict(c_check)[1])
            assert np.allclose(agent.counts[arm], rf_handler.predict(c_check)[2])
            assert np.allclose(preds_ilove, preds_ref)


def test_tree_ensemble_ts(rfhandler_toy_prob):
    """Test the specific functionalities of tree ensemble for ThompSamp."""
    vpar = 1.0
    n_estimators = 2
    min_samples_leaf = 2
    max_depth = 1

    c_red = rfhandler_toy_prob["c_red"]
    a_red = rfhandler_toy_prob["a_red"]
    y_red = rfhandler_toy_prob["y_red"]
    arms = rfhandler_toy_prob["arms"]
    nfeats = rfhandler_toy_prob["nfeats"]
    c_check = np.array([1.0] * nfeats).reshape(1, -1)

    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        criterion="squared_error",
        max_samples=None,
        # n_jobs=N_CORES,
        random_state=42,
    )

    rfts_agent = RandomForestTsAgent(
        arms=arms,
        vpar=vpar,
        base_model=clone(rf_model),
        rng_seed=42,
        one_model_per_arm=False,
        samples_for_freq_est=20_000,
        n_rounds_random=0,
        min_rewards_per_arm=0,
    )

    rfts_agent.update_agent(c_train=c_red, a_train=a_red, r_train=y_red)

    ### TEST .sample_qvals()
    rfts_agent.estimate_means_vars(c_check)
    list_actions = []
    for _ in range(200_000):
        list_actions.append(rfts_agent.sample_qvals())

    a0, a1, a2, a3 = zip(*list_actions)

    sampled_means = np.array(
        [np.mean(a0), np.mean(a1), np.mean(a2), np.mean(a3)]
    ).reshape(-1, 1)
    sampled_vars = np.array([np.var(a0), np.var(a1), np.var(a2), np.var(a3)]).reshape(
        -1, 1
    )

    npt.assert_allclose(
        actual=rfts_agent.means, desired=sampled_means, rtol=1e-3, atol=1e-8
    )

    npt.assert_allclose(
        actual=rfts_agent.vars, desired=sampled_vars, rtol=1e-2, atol=1e-8
    )

    ### TEST ._freq_arms()
    # overwrite means and vars to force equal probability to choose each arm
    rfts_agent.means = [1] * arms
    rfts_agent.vars = [0.25] * arms
    dic_freq_arms = rfts_agent._freq_arms()
    for est_proba in dic_freq_arms.values():
        assert est_proba == pytest.approx(1 / arms, rel=1e-2)


def test_tree_ensemble_ucb(rfhandler_toy_prob):
    """Test the specific functionalities of tree ensemble for UCB.

    Note: Be sure that you are testing the UCB formula correctly.
    If you use other UCB formula, please update the test accordingly or it will fail.
    """
    vpar = 1.0
    n_estimators = 2
    min_samples_leaf = 2
    max_depth = 1

    c_red = rfhandler_toy_prob["c_red"]
    a_red = rfhandler_toy_prob["a_red"]
    y_red = rfhandler_toy_prob["y_red"]
    arms = rfhandler_toy_prob["arms"]
    nfeats = rfhandler_toy_prob["nfeats"]
    c_check = np.array([1.0] * nfeats).reshape(1, -1)

    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        criterion="squared_error",
        max_samples=None,
        random_state=42,
    )

    rfucb_agent = RandomForestUcbAgent(
        arms=arms,
        vpar=1.0,
        base_model=rf_model,
        rng_seed=42,
        one_model_per_arm=False,
        n_rounds_random=0,
        min_rewards_per_arm=0,
    )

    rfucb_agent.update_agent(c_train=c_red, a_train=a_red, r_train=y_red)
    rfucb_agent.estimate_means_vars(c_check)

    ### TEST UCB FORMULA ###
    ref_means = rfucb_agent.means
    ref_var = rfucb_agent.vars
    ref_count = rfucb_agent.counts
    # ** UCB FORMULA TO CHECK. Insert here **#
    total_learning = len(a_red)
    ref_qval_samples = []
    for arm in range(arms):
        arm_count = ref_count[arm]
        ref_qval_samples.append(
            ref_means[arm]
            + np.sqrt(vpar * ref_var[arm] * np.log(total_learning - 1) / arm_count)
        )
    # ** UCB FORMULA TO CHECK. Finish Insert **#

    _, prob_action = rfucb_agent.take_action(c_check)
    _, ref_prob, _ = argmax(
        ref_qval_samples,
        rfucb_agent.rng,
    )

    npt.assert_allclose(
        actual=rfucb_agent.last_qval_samples,
        desired=ref_qval_samples,
        rtol=1e-2,
        atol=1e-8,
    )

    assert pytest.approx(ref_prob, abs=0.01) == prob_action
