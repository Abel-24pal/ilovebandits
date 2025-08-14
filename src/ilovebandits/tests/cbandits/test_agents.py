"""Testing submodule agents.py."""

import numpy as np
import numpy.testing as npt
import pytest
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.ilovebandits.agents import EpsGreedyConAgent, MismatchedArmNumberError, NotEnoughRewardsPerArmError

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
        "base_estimator": LogisticRegression(penalty=None, fit_intercept=False, solver="lbfgs"),
    },
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


@pytest.fixture(scope="module")
def data_disjoint():
    """Return the data for the disjoint epsilon-greedy agents (one model per arm). Linear Regression problem."""
    coefs = [[1, 2, -1], [2, -3, 4], [3, -1, 8], [2, -1, 0.5]]
    arms = len(coefs)
    feats = np.array(
        [
            [1, -1, 2],
            [2, 3, 4],
            [3, -3, 8],
            [4, 8, 10],
        ]
    )

    c_list = []
    a_list = []
    r_list = []

    for idx_a in range(arms):
        r_list.append(coefs[idx_a][0] * feats[:, 0] + coefs[idx_a][1] * feats[:, 1] + coefs[idx_a][2] * feats[:, 2])
        c_list.append(feats)
        a_list.append(np.ones(feats.shape[0]).astype(int) * idx_a)

    c_train = np.concatenate(c_list, axis=0)
    a_train = np.concatenate(a_list, axis=0)
    r_train = np.concatenate(r_list, axis=0)

    return {
        "c_train": c_train,
        "a_train": a_train,
        "r_train": r_train,
        "feats": feats,
        "arms": arms,
        "coefs": coefs,
    }


@pytest.fixture(scope="module")
def data_hybrid():
    """Return the data for the hybrid epsilon-greedy agent (one model for all arms). Linear Regression problem."""
    arms = 4
    coefs = [1, 2, -1, 4]
    a_train = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    c_train = np.array(
        [
            [-10, -1, 2],
            [2, 3, -4],
            [3, -3, 8],
            [-4, 8, 0],
            [0, 8, 2],
            [0, 8, 2],
            [8, 0, 2],
            [2, 8, 0],
        ]
    )
    r_train = coefs[0] * c_train[:, 0] + coefs[1] * c_train[:, 1] + coefs[2] * c_train[:, 2] + coefs[3] * a_train

    return {
        "c_train": c_train,
        "a_train": a_train,
        "r_train": r_train,
        "arms": arms,
        "coefs": coefs,
    }


@pytest.fixture(scope="module")
def data_hybrid_clf():
    """Return the data for the hybrid epsilon-greedy agent (one model for all arms). Classification problem."""
    rng = np.random.default_rng(RANDOM_SEED)
    arms = 4
    samples = 800000  # it should be divisible by 4
    coefs = [1, 2, -1, 4]
    a_train = np.tile(np.array([[0, 1, 2, 3]]), reps=(1, int(samples / 4))).T
    c_train = rng.normal(0, 1, size=(samples, len(coefs) - 1))  # design matrix

    xmatrix = np.hstack((c_train, a_train))
    logits = xmatrix @ np.array(coefs)  # or equivalent: np.dot(xmatrix, np.array(coefs))
    probs = expit(logits)
    r_train = (rng.random(samples) < probs).astype(int)  # Convert to binary classification problem

    return {
        "c_train": c_train,
        "a_train": a_train.ravel(),
        "r_train": r_train,
        "arms": arms,
        "coefs": coefs,
    }


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
    npt.assert_allclose(actual=1 / arms * np.ones(5), desired=[p1, p2, p3, p4, p5], rtol=1e-7, atol=1e-8)

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
    npt.assert_allclose(actual=c_train, desired=eps_agent.last_c_train, rtol=1e-7, atol=1e-8)
    npt.assert_allclose(actual=a_train, desired=eps_agent.last_a_train, rtol=1e-7, atol=1e-8)
    npt.assert_allclose(actual=r_train, desired=eps_agent.last_r_train, rtol=1e-7, atol=1e-8)

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
        pytest.approx(eps_agent.arm_count[np.argmax(expr_dummy)] / sum(eps_agent.arm_count), abs=0.01)
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
    expr_dummy = np.array(expr_dummy)  # expected reward of the dummy context for each arm

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
    npt.assert_allclose(actual=c_train, desired=eps_agent.last_c_train, rtol=1e-7, atol=1e-8)
    npt.assert_allclose(actual=a_train, desired=eps_agent.last_a_train, rtol=1e-7, atol=1e-8)
    npt.assert_allclose(actual=r_train, desired=eps_agent.last_r_train, rtol=1e-7, atol=1e-8)

    # check coefficients of the linear regression model
    npt.assert_allclose(actual=eps_agent.model.coef_, desired=coefs, rtol=1e-7, atol=1e-8)
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
        pytest.approx(eps_agent.arm_count[np.argmax(expr_dummy)] / sum(eps_agent.arm_count), abs=0.01)
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
    expr_dummy = np.array(expr_dummy)  # expected reward of the dummy context for each arm

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
    npt.assert_allclose(actual=c_train, desired=eps_agent.last_c_train, rtol=1e-7, atol=1e-8)
    npt.assert_allclose(actual=a_train, desired=eps_agent.last_a_train, rtol=1e-7, atol=1e-8)
    npt.assert_allclose(actual=r_train, desired=eps_agent.last_r_train, rtol=1e-7, atol=1e-8)

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
        pytest.approx(eps_agent.arm_count[np.argmax(expr_dummy)] / sum(eps_agent.arm_count), abs=0.01)
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

    ####### CREATE AGENT ########
    eps_agent = EpsGreedyConAgent(
        arms=arms - 1,  # intentionally set to arms - 1 to trigger error
        base_estimator=eps_pars_hybrid["base_estimator"],
        n_rounds_random=eps_pars_hybrid["n_rounds_random"],
        epsilon=eps_pars_hybrid["epsilon"],
        one_model_per_arm=eps_pars_hybrid["one_model_per_arm"],
        rng_seed=RANDOM_SEED,
    )

    ########### FIT AGENT ##########
    with pytest.raises(MismatchedArmNumberError):
        eps_agent.update_agent(c_train=c_train, a_train=a_train, r_train=r_train)


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

    ####### CREATE AGENT ########
    eps_agent = EpsGreedyConAgent(
        arms=arms,
        base_estimator=eps_pars_hybrid["base_estimator"],
        n_rounds_random=eps_pars_hybrid["n_rounds_random"],
        epsilon=eps_pars_hybrid["epsilon"],
        one_model_per_arm=eps_pars_hybrid["one_model_per_arm"],
        rng_seed=RANDOM_SEED,
    )

    ########### FIT AGENT ##########
    with pytest.raises(NotEnoughRewardsPerArmError):
        eps_agent.update_agent(c_train=c_train, a_train=a_train, r_train=r_train)
