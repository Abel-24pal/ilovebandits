import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from scipy.special import expit
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._forest import _generate_sample_indices


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
        r_list.append(
            coefs[idx_a][0] * feats[:, 0]
            + coefs[idx_a][1] * feats[:, 1]
            + coefs[idx_a][2] * feats[:, 2]
        )
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
    r_train = (
        coefs[0] * c_train[:, 0]
        + coefs[1] * c_train[:, 1]
        + coefs[2] * c_train[:, 2]
        + coefs[3] * a_train
    )

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
    rng = np.random.default_rng(42)
    arms = 4
    samples = 800000  # it should be divisible by 4
    coefs = [1, 2, -1, 4]
    a_train = np.tile(np.array([[0, 1, 2, 3]]), reps=(1, int(samples / 4))).T
    c_train = rng.normal(0, 1, size=(samples, len(coefs) - 1))  # design matrix

    xmatrix = np.hstack((c_train, a_train))
    logits = xmatrix @ np.array(
        coefs
    )  # or equivalent: np.dot(xmatrix, np.array(coefs))
    probs = expit(logits)
    r_train = (rng.random(samples) < probs).astype(
        int
    )  # Convert to binary classification problem

    return {
        "c_train": c_train,
        "a_train": a_train.ravel(),
        "r_train": r_train,
        "arms": arms,
        "coefs": coefs,
    }


##################################################################


@pytest.fixture(scope="module")
def get_classification_data():
    """Note about n_features: number of features including the column
    that indicates the arm selection.
    """
    random_state, n_features, n_samples = 42, 5, 5000

    # Generate a binary classification dataset.
    x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_clusters_per_class=1,
        n_informative=n_features,
        n_redundant=0,
        n_repeated=0,
        random_state=random_state,
    )

    # Use last column as arm column. Convert values to integers in the range [0, n_features-1].
    q1 = pd.DataFrame(x).describe().loc["25%", n_features - 1]
    q2 = pd.DataFrame(x).describe().loc["50%", n_features - 1]
    q3 = pd.DataFrame(x).describe().loc["75%", n_features - 1]

    x[:, -1] = np.where(
        x[:, -1] < q1, 0, np.where(x[:, -1] < q2, 1, np.where(x[:, -1] < q3, 2, 3))
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    c_train = x_train[:, :-1]  # Contextual features
    a_train = x_train[:, -1].astype(int)  # Arm selected

    c_test = x_test[:, :-1]  # Contextual features
    a_test = x_test[:, -1].astype(int)  # Arm selected

    data = {}
    data["x_train"] = x_train
    data["c_train"] = c_train
    data["a_train"] = a_train
    data["y_train"] = y_train
    data["x_test"] = x_test
    data["c_test"] = c_test
    data["a_test"] = a_test
    data["y_test"] = y_test

    return data


@pytest.fixture(scope="module")
def get_regression_data():
    """Note about n_features: number of features including the column
    that indicates the arm selection.
    """
    random_state, n_features, n_samples = 42, 5, 5000

    # Generate a regression dataset.
    x, y = make_regression(
        n_samples=n_samples, n_features=n_features, noise=1, random_state=random_state
    )

    # Use last column as arm column. Convert values to integers in the range [0, n_features-1].
    q1 = pd.DataFrame(x).describe().loc["25%", n_features - 1]
    q2 = pd.DataFrame(x).describe().loc["50%", n_features - 1]
    q3 = pd.DataFrame(x).describe().loc["75%", n_features - 1]

    # print(f"Q1: {q1}, Q2: {q2}, Q3: {q3}")

    x[:, -1] = np.where(
        x[:, -1] < q1, 0, np.where(x[:, -1] < q2, 1, np.where(x[:, -1] < q3, 2, 3))
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    c_train = x_train[:, :-1]  # Contextual features
    a_train = x_train[:, -1].astype(int)  # Arm selected

    c_test = x_test[:, :-1]  # Contextual features
    a_test = x_test[:, -1].astype(int)  # Arm selected

    data = {}
    data["x_train"] = x_train
    data["c_train"] = c_train
    data["a_train"] = a_train
    data["y_train"] = y_train
    data["x_test"] = x_test
    data["c_test"] = c_test
    data["a_test"] = a_test
    data["y_test"] = y_test

    return data


@pytest.fixture(scope="module")
def rfhandler_toy_prob():
    """ "This contains the reference data for a random forest regression problem. Here, it is computed the mean, count and variances in each tree and leaf nodes.
    This is computed assuming the used parameters below. Things like the decision splits to be found are hardcoded."""
    rf_pars = {
        "n_estimators": 2,
        "min_samples_leaf": 2,
        "max_depth": 1,
        "criterion": "squared_error",
        "random_state": 42,
        "max_samples": None,
    }
    rf_pars["rf_avg_"] = 1.0 / rf_pars["n_estimators"]

    x_red = np.array(
        [
            [-1.03235211, 0.01772303, 0.0],
            [-0.67657322, 0.20022868, 1.0],
            [1.38427582, -0.6641666, 1.0],
            [0.42262633, 0.32744341, 1.0],
            [1.67628789, -0.91734343, 2.0],
            [1.629431, -1.34926428, 0.0],
            [-0.1284255, -1.47691442, 3.0],
            [-0.2713071, 2.45868241, 3.0],
            [-1.2639901, 0.53765843, 1.0],
            [-1.59933577, 1.71283289, 0.0],
        ]
    )

    y_red = np.array(
        [
            -20.06877974,
            -21.82492121,
            143.91112466,
            157.82843687,
            -71.18482613,
            -133.64017262,
            -46.1331373,
            39.40058786,
            -32.30772486,
            -44.25425066,
        ]
    )

    c_red = x_red[:, :2]
    a_red = x_red[:, 2].astype(int)
    arms = len(np.unique(a_red))
    nfeats = c_red.shape[1]

    rf_model = RandomForestRegressor(
        n_estimators=rf_pars["n_estimators"],
        min_samples_leaf=rf_pars[
            "min_samples_leaf"
        ],  # important for the assumptions of sample mean and sample standard deviation in leaf nodes for TEUCB (Tree Ensemble UCB) and TETS (Tree Ensemble Thompson Sampling)
        max_depth=rf_pars["max_depth"],
        criterion=rf_pars["criterion"],
        max_samples=rf_pars["max_samples"],
        random_state=rf_pars["random_state"],
    )

    rf_model.fit(x_red, y_red)

    # Obtain bootstrapped samples for each tree (this is computed assuming max_samples=None)

    n_samples = len(x_red)
    n_samples_bootstrap = n_samples

    train_indices = [
        _generate_sample_indices(
            rf_model.estimators_[sel_tree].random_state, n_samples, n_samples_bootstrap
        )
        for sel_tree in range(len(rf_model.estimators_))
    ]

    x_train_trees = [x_red[indices] for indices in train_indices]
    y_train_trees = [rf_pars["rf_avg_"] * y_red[indices] for indices in train_indices]

    # Obtain reference values for each leaf node:

    tree_1_cond = (
        x_train_trees[0][:, 1] <= 0.433
    )  # This is the decision split for first tree
    tree_2_cond = (
        x_train_trees[1][:, 2] <= 0.5
    )  # This is the decision split for first tree

    tree1_leaf1_count = len(y_train_trees[0][tree_1_cond])
    tree1_leaf2_count = len(y_train_trees[0][~tree_1_cond])
    tree2_leaf1_count = len(y_train_trees[1][tree_2_cond])
    tree2_leaf2_count = len(y_train_trees[1][~tree_2_cond])

    tree1_leaf1_mean = np.mean(y_train_trees[0][tree_1_cond])
    tree1_leaf2_mean = np.mean(y_train_trees[0][~tree_1_cond])
    tree2_leaf1_mean = np.mean(y_train_trees[1][tree_2_cond])
    tree2_leaf2_mean = np.mean(y_train_trees[1][~tree_2_cond])

    tree1_leaf1_var = np.var(y_train_trees[0][tree_1_cond])
    tree1_leaf2_var = np.var(y_train_trees[0][~tree_1_cond])
    tree2_leaf1_var = np.var(y_train_trees[1][tree_2_cond])
    tree2_leaf2_var = np.var(y_train_trees[1][~tree_2_cond])

    # Reference values for a inference point:
    x_check = np.array([[1.0, 1.0, 1.0]])  # Assumed reference value

    # Computation assuming previous decision splits of the trees in the ensemble
    ref_count = tree1_leaf2_count + tree2_leaf2_count
    ref_mean = tree1_leaf2_mean + tree2_leaf2_mean
    ref_var = tree1_leaf2_var / tree1_leaf2_count + tree2_leaf2_var / tree2_leaf2_count

    inference_point = {
        "x_check": x_check,
        "ref_count": ref_count,
        "ref_mean": ref_mean,
        "ref_var": ref_var,
    }

    return {
        "rf_pars": rf_pars,
        "x_red": x_red,
        "y_red": y_red,
        "c_red": c_red,
        "a_red": a_red,
        "tree1_leaf1_count": tree1_leaf1_count,
        "tree1_leaf2_count": tree1_leaf2_count,
        "tree2_leaf1_count": tree2_leaf1_count,
        "tree2_leaf2_count": tree2_leaf2_count,
        "tree1_leaf1_mean": tree1_leaf1_mean,
        "tree1_leaf2_mean": tree1_leaf2_mean,
        "tree2_leaf1_mean": tree2_leaf1_mean,
        "tree2_leaf2_mean": tree2_leaf2_mean,
        "tree1_leaf1_var": tree1_leaf1_var,
        "tree1_leaf2_var": tree1_leaf2_var,
        "tree2_leaf1_var": tree2_leaf1_var,
        "tree2_leaf2_var": tree2_leaf2_var,
        "inference_point": inference_point,
        "arms": arms,
        "nfeats": nfeats,
    }
