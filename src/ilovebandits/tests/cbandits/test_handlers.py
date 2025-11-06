from src.ilovebandits.handlers import RandForestHandler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import clone
import numpy as np
import pytest

pars_handlers = {
    "handler_rfclf": RandForestHandler(
        base_model=RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=20,
            max_depth=3,
            criterion="log_loss",
            max_samples=None,
            random_state=42,
        )
    ),
    "handler_rfreg": RandForestHandler(
        base_model=RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=2,  # important for the assumptions of sample mean and sample standard deviation in leaf nodes for TEUCB (Tree Ensemble UCB) and TETS (Tree Ensemble Thompson Sampling)
            max_depth=5,
            criterion="squared_error",
            max_samples=None,
            random_state=42,
        )
    ),
}


@pytest.mark.parametrize("handler_to_test", [pars_handlers["handler_rfclf"]])
def test_handler_equal_preds_clf(handler_to_test, get_classification_data):
    """Test that means are equal to preds in classification models."""
    x_train = get_classification_data["x_train"]
    y_train = get_classification_data["y_train"]
    x_test = get_classification_data["x_test"]

    base_model = clone(handler_to_test.base_model)

    base_model.fit(x_train, y_train)
    handler_to_test.fit(x_train, y_train)

    # Assuming binary classification, we take the probability of the positive class
    base_model_preds = base_model.predict_proba(x_test)[:, 1]
    # Assuming binary classification, we take the probability of the positive class
    means, _, _ = handler_to_test.predict(x_test)

    assert np.allclose(base_model_preds, means)


@pytest.mark.parametrize("handler_to_test", [pars_handlers["handler_rfreg"]])
def test_handler_equal_preds_reg(handler_to_test, get_regression_data):
    """Test that rf handler gets the correct data."""
    x_train = get_regression_data["x_train"]
    y_train = get_regression_data["y_train"]
    x_test = get_regression_data["x_test"]

    base_model = clone(handler_to_test.base_model)

    base_model.fit(x_train, y_train)
    handler_to_test.fit(x_train, y_train)

    # Assuming binary classification, we take the probability of the positive class
    base_model_preds = base_model.predict(x_test)

    # Assuming binary classification, we take the probability of the positive class
    means, _, _ = handler_to_test.predict(x_test)

    assert np.allclose(base_model_preds, means)


def test_rfhandler_stats_toy_problem(rfhandler_toy_prob):
    rf_pars = rfhandler_toy_prob["rf_pars"]
    x_red = rfhandler_toy_prob["x_red"]
    y_red = rfhandler_toy_prob["y_red"]
    tree1_leaf1_count = rfhandler_toy_prob["tree1_leaf1_count"]
    tree1_leaf2_count = rfhandler_toy_prob["tree1_leaf2_count"]
    tree2_leaf1_count = rfhandler_toy_prob["tree2_leaf1_count"]
    tree2_leaf2_count = rfhandler_toy_prob["tree2_leaf2_count"]
    tree1_leaf1_mean = rfhandler_toy_prob["tree1_leaf1_mean"]
    tree1_leaf2_mean = rfhandler_toy_prob["tree1_leaf2_mean"]
    tree2_leaf1_mean = rfhandler_toy_prob["tree2_leaf1_mean"]
    tree2_leaf2_mean = rfhandler_toy_prob["tree2_leaf2_mean"]
    tree1_leaf1_var = rfhandler_toy_prob["tree1_leaf1_var"]
    tree1_leaf2_var = rfhandler_toy_prob["tree1_leaf2_var"]
    tree2_leaf1_var = rfhandler_toy_prob["tree2_leaf1_var"]
    tree2_leaf2_var = rfhandler_toy_prob["tree2_leaf2_var"]
    inference_point = rfhandler_toy_prob["inference_point"]

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

    rf_handler = RandForestHandler(base_model=rf_model)
    rf_handler.fit(X=x_red, y=y_red)

    # Check means, vars, counts at each leaf node

    assert rf_handler.uc_data_[0]["leaf_counts"][1] == pytest.approx(
        tree1_leaf1_count, rel=1e-6
    )
    assert rf_handler.uc_data_[0]["leaf_counts"][2] == pytest.approx(
        tree1_leaf2_count, rel=1e-6
    )
    assert rf_handler.uc_data_[1]["leaf_counts"][1] == pytest.approx(
        tree2_leaf1_count, rel=1e-6
    )
    assert rf_handler.uc_data_[1]["leaf_counts"][2] == pytest.approx(
        tree2_leaf2_count, rel=1e-6
    )

    assert rf_handler.uc_data_[0]["leaf_avg_vals"][1] == pytest.approx(
        tree1_leaf1_mean, rel=1e-6
    )
    assert rf_handler.uc_data_[0]["leaf_avg_vals"][2] == pytest.approx(
        tree1_leaf2_mean, rel=1e-6
    )
    assert rf_handler.uc_data_[1]["leaf_avg_vals"][1] == pytest.approx(
        tree2_leaf1_mean, rel=1e-6
    )
    assert rf_handler.uc_data_[1]["leaf_avg_vals"][2] == pytest.approx(
        tree2_leaf2_mean, rel=1e-6
    )

    assert (
        rf_handler.uc_data_[0]["leaf_var_vals"][1]
        == pytest.approx(
            tree1_leaf1_var
            / tree1_leaf1_count,  # remeber that the variance of the leaf output is the sample_variance/n_samples. See paper.
            rel=1e-5,
        )
    )
    assert rf_handler.uc_data_[0]["leaf_var_vals"][2] == pytest.approx(
        tree1_leaf2_var / tree1_leaf2_count, rel=1e-5
    )
    assert rf_handler.uc_data_[1]["leaf_var_vals"][1] == pytest.approx(
        tree2_leaf1_var / tree2_leaf1_count, rel=1e-5
    )
    assert rf_handler.uc_data_[1]["leaf_var_vals"][2] == pytest.approx(
        tree2_leaf2_var / tree2_leaf2_count, rel=1e-5
    )

    # Now test predictions at the inference point
    x_check = inference_point["x_check"]
    ref_mean = inference_point["ref_mean"]
    ref_var = inference_point["ref_var"]
    ref_count = inference_point["ref_count"]

    pred_mean, pred_var, pred_count = rf_handler.predict(x_check)

    assert ref_count == pytest.approx(pred_count, rel=1e-6)
    assert ref_mean == pytest.approx(pred_mean, rel=1e-6)
    assert ref_var == pytest.approx(pred_var, rel=1e-6)
