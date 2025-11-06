import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from typing import Tuple, Union
from sklearn.ensemble._forest import _generate_sample_indices
from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


# Handlers provide uncertainty estimates.
# They must follow scikit-learn BaseEstimator conventions
# This is needed to allow the handlers to be cloned.
# You can check that your handler follows scikit-learn conventions by using:
# --> from sklearn.utils.estimator_checks import check_estimator
# --> check_estimator(YourHandler(...))
# RandForestHandler:
#   passes the test if first predict method is uncommented and we do the test with it.
#   Once the test passed, we can comment it and use the second predict method instead for ilovebandits
class RandForestHandler(BaseEstimator):
    def __init__(
        self,
        base_model: Union[RandomForestClassifier, RandomForestRegressor] = None,
    ):
        self.base_model = base_model
        # self.uc_data_ = {}  # to be defined in fit. I cannot initialize it here if this is a sklearn.base.BaseEstimator
        # self.model_ = None  # to be defined in fit. I cannot initialize it here if this is a sklearn.base.BaseEstimator
        # self.x_train_ = None  # to be defined in fit. I cannot initialize it here if this is a sklearn.base.BaseEstimator. Arm column is the last column of this array.
        # self.y_train_ = None  # to be defined in fit. I cannot initialize it here if this is a sklearn.base.BaseEstimator

    def fit(self, X: np.array, y: np.array):
        """ "Fit Random Forest model and extract training data points per tree.

        Parameters
        ----------
        x_train : np.array of shape (n_samples, n_features)
            Features for training.
        y_train : np.array of shape (n_samples, )
            Rewards obtained for each sample in training.
        """
        if (
            self.base_model is None
            or isinstance(
                self.base_model, (RandomForestClassifier, RandomForestRegressor)
            )
            is False
        ):
            raise ValueError(
                "Couldn't fit. Please provide a valid RandomForestClassifier or RandomForestRegressor as base_model."
            )

        if self.base_model.max_samples is not None:
            raise ValueError(
                "Please set max_samples=None in the RandomForest model to ensure correct uncertainty estimation."
            )

        X, y = check_X_y(X, y, force_all_finite=True)
        self.n_features_in_ = X.shape[1]
        self.x_train_ = X
        self.y_train_ = y

        self.model_ = clone(self.base_model).fit(
            self.x_train_,
            self.y_train_,
        )

        self.rf_avg_ = 1 / self.model_.n_estimators

        n_samples = len(self.x_train_)
        n_samples_bootstrap = (
            n_samples  # This is true if max_samples=None in RandomForestRegressor.
        )
        # Be careful if other RandomForestRegressor parameters of boosting are used, because this can change the logic of extracting the training samples of a given tree.

        train_indices = [
            _generate_sample_indices(
                self.model_.estimators_[sel_tree].random_state,
                n_samples,
                n_samples_bootstrap,
            )
            for sel_tree in range(len(self.model_.estimators_))
        ]

        x_train_trees = [self.x_train_[indices] for indices in train_indices]
        y_train_trees = [self.y_train_[indices] for indices in train_indices]

        self.uc_data_ = {}  # initialize this attribute
        self.set_data_per_tree(x_pertree=x_train_trees, y_pertree=y_train_trees)

        return self

    # ######### TEST NOT OPTIMIZED: #############
    # def set_data_per_tree(self, x_pertree: List[np.array], y_pertree: List[np.array]):
    #     """Compute necessary self.uc_data object to do the uncetainty estimation. In this case, we use the provided x_val, y_val. we do not use the bootstrap samples.

    #     Parameters
    #     ----------
    #     x_pertree : List of np.arrays of shape (n_samples, n_features)
    #         Feature data points for each tree in the forest. The position in the list corresponds to the tree index whose points belong to.
    #         Different data points can be used for each tree or the same data points can be used for all trees.
    #     y_pertree : List of np.arrays of shape (n_samples, )
    #         Target/reward data points for each tree in the forest. The position in the list corresponds to the tree index whose points belong to.
    #         Different data points can be used for each tree or the same data points can be used for all trees.

    #     Notes
    #     ----------
    #     self.uc_data : dict
    #         Dictionary where each key is a tree index and the value is another dictionary containing:
    #         - 'leaf_counts': dict mapping leaf index to the number of samples in that leaf.
    #         - 'leaf_avg_vals': dict mapping leaf index to the average value of the samples in that leaf.
    #         - 'leaf_var_vals': dict mapping leaf index to the variance of the values of the samples in that leaf.
    #         - 'x_samples': np.array of shape (n_samples, n_features)
    #             Features for the training samples of the tree.
    #         - 'y_samples': np.array of shape (n_samples, )
    #             Rewards for the training samples of the tree.
    #         - 'leaf_ids_train': np.array of shape (n_samples, )
    #             Leaf index for each sample in the training set (x_samples) of the tree.
    #     """
    #     #####OPTION what it seems researchers do:
    #     y_pertree = [ytree * self.rf_avg_ for ytree in y_pertree]

    #     for sel_tree in range(len(self.model_.estimators_)):
    #         self.uc_data_[sel_tree] = {}

    #         leaf_ids = self.model_.estimators_[
    #             sel_tree
    #         ].apply(
    #             x_pertree[sel_tree]
    #         )  # This is the leaf index for each sample in the training set of the tree sel_tree

    #         leaf_tree_idxs, leaf_tree_count = np.unique(leaf_ids, return_counts=True)

    #         self.uc_data_[sel_tree]["leaf_counts"] = dict(
    #             zip(leaf_tree_idxs, leaf_tree_count)
    #         )

    #         self.uc_data_[sel_tree]["leaf_avg_vals"] = dict(
    #             zip(
    #                 leaf_tree_idxs,
    #                 [
    #                     y_pertree[sel_tree][leaf_ids == leaf_idx].mean()
    #                     for leaf_idx in leaf_tree_idxs
    #                 ],
    #             )
    #         )

    #         ####OPTION what it seems researchers do:
    #         dic_leaf_tree_count = self.uc_data_[sel_tree]["leaf_counts"]
    #         self.uc_data_[sel_tree]["leaf_var_vals"] = dict(
    #             zip(
    #                 leaf_tree_idxs,
    #                 [
    #                     y_pertree[sel_tree][leaf_ids == leaf_idx].var()
    #                     / dic_leaf_tree_count[leaf_idx]
    #                     for leaf_idx in leaf_tree_idxs
    #                 ],
    #             )
    #         )

    #         self.uc_data_[sel_tree]["x_samples"] = x_pertree[sel_tree]
    #         self.uc_data_[sel_tree]["y_samples"] = y_pertree[sel_tree]
    #         self.uc_data_[sel_tree]["leaf_ids_train"] = (
    #             leaf_ids  # This is the leaf index for each sample in the training set of the tree sel_tree
    #         )

    ######### TEST OPTIMIZATION: #############
    def set_data_per_tree(
        self, x_pertree: list[np.ndarray], y_pertree: list[np.ndarray]
    ):
        """Compute necessary self.uc_data object to do the uncetainty estimation. In this case, we use the provided x_val, y_val. we do not use the bootstrap samples.

        Optimized version: computes per-leaf statistics (counts, means, variances)
        efficiently using vectorized numpy operations.

        Parameters
        ----------
        x_pertree : List of np.arrays of shape (n_samples, n_features)
            Feature data points for each tree in the forest. The position in the list corresponds to the tree index whose points belong to.
            Different data points can be used for each tree or the same data points can be used for all trees.
        y_pertree : List of np.arrays of shape (n_samples, )
            Target/reward data points for each tree in the forest. The position in the list corresponds to the tree index whose points belong to.
            Different data points can be used for each tree or the same data points can be used for all trees.

        Notes
        ----------
        self.uc_data : dict
            Dictionary where each key is a tree index and the value is another dictionary containing:
            - 'leaf_counts': dict mapping leaf index to the number of samples in that leaf.
            - 'leaf_avg_vals': dict mapping leaf index to the average value of the samples in that leaf.
            - 'leaf_var_vals': dict mapping leaf index to the variance of the values of the samples in that leaf.
            - 'x_samples': np.array of shape (n_samples, n_features)
                Features for the training samples of the tree.
            - 'y_samples': np.array of shape (n_samples, )
                Rewards for the training samples of the tree.
            - 'leaf_ids_train': np.array of shape (n_samples, )
                Leaf index for each sample in the training set (x_samples) of the tree.
        """
        y_pertree = [ytree * self.rf_avg_ for ytree in y_pertree]

        for sel_tree, (x_tree, y_tree) in enumerate(zip(x_pertree, y_pertree)):
            estimator = self.model_.estimators_[sel_tree]
            leaf_ids = estimator.apply(x_tree)  # shape (n_samples,)

            # Map leaf_ids to consecutive integers for bincount
            unique_leaves, inv = np.unique(leaf_ids, return_inverse=True)

            counts = np.bincount(inv)
            sums = np.bincount(inv, weights=y_tree)
            means = sums / counts

            # Variance calculation: var = E[y^2] - (E[y])^2
            sums_sq = np.bincount(inv, weights=y_tree**2)
            vars_ = (sums_sq / counts) - (means**2)
            vars_ /= counts  # match your normalization convention

            # Store results
            self.uc_data_[sel_tree] = {
                "leaf_counts": dict(zip(unique_leaves, counts)),
                "leaf_avg_vals": dict(zip(unique_leaves, means)),
                "leaf_var_vals": dict(zip(unique_leaves, vars_)),
                "x_samples": x_tree,
                "y_samples": y_tree,
                "leaf_ids_train": leaf_ids,
            }

    def set_whole_train_data_per_tree(self):
        """Compute necessary self.uc_data object to do the uncetainty estimation. In this case, we use the whole training data, not just the bootstrap samples.

        Notes
        ----------
        self.uc_data : dict
            Dictionary where each key is a tree index and the value is another dictionary containing:
            - 'leaf_counts': dict mapping leaf index to the number of samples in that leaf.
            - 'leaf_avg_vals': dict mapping leaf index to the average value of the samples in that leaf.
            - 'leaf_var_vals': dict mapping leaf index to the variance of the values of the samples in that leaf.
            - 'x_samples': np.array of shape (n_samples, n_features)
                Features for the training samples of the tree.
            - 'y_samples': np.array of shape (n_samples, )
                Rewards for the training samples of the tree.
            - 'leaf_ids_train': np.array of shape (n_samples, )
                Leaf index for each sample in the training set (x_samples) of the tree.
        """
        x_pertree = [self.x_train_ for _ in range(len(self.model_.estimators_))]
        y_pertree = [self.y_train_ for _ in range(len(self.model_.estimators_))]
        self.set_data_per_tree(x_pertree=x_pertree, y_pertree=y_pertree)

    ### Predict method, just for evaluating "from sklearn.utils.estimator_checks import check_estimator", "check_estimator(RandForestHandler(base_model=RandomForestRegressor(random_state=42)))"
    # def predict(self, x_test: np.array) -> np.array:
    #     """ "Compute Mean and Variance prediction of the points contained in x_test.

    #     Parameters & Outputs
    #     --------------------
    #     x_test : np.array of shape (n_samples, n_features)
    #         Features for the test samples.
    #     mean_pred : np.array of shape (n_samples, )
    #         Mean prediction for each sample in x_test.
    #     var_pred : np.array of shape (n_samples, )
    #         Variance prediction for each sample in x_test.
    #     """
    #     if self.model_ is None:
    #         raise ValueError("Model has not been fitted yet.")
    #     # Ensure model is fitted
    #     check_is_fitted(self, "model_")
    #     # Validate input data for prediction
    #     x_test = check_array(x_test, force_all_finite=True)

    #     n_samples = len(x_test)
    #     n_trees = self.model_.n_estimators
    #     tree_preds = np.zeros((n_samples, n_trees))

    #     for sel_tree in range(n_trees):
    #         leaf_indices = self.model_.estimators_[sel_tree].apply(x_test)

    #         leaf_vals = self.uc_data_[sel_tree]["leaf_avg_vals"]
    #         tree_preds[:, sel_tree] = [leaf_vals[leaf_idx] for leaf_idx in leaf_indices]

    #     # #####OPTION what it seems researchers do:
    #     mean_pred = np.sum(tree_preds, axis=1)

    #     return mean_pred

    ### Predict method desired. uncomment once previous predict passed the "check_estimator" checks.
    def predict(self, x_test: np.array) -> Tuple[np.array, np.array, np.array]:
        """ "Compute Mean and Variance prediction of the points contained in x_test.

        Parameters & Outputs
        --------------------
        x_test : np.array of shape (n_samples, n_features)
            Features for the test samples.
        mean_pred : np.array of shape (n_samples, )
            Mean prediction for each sample in x_test.
        var_pred : np.array of shape (n_samples, )
            Variance prediction for each sample in x_test.
        """
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet.")
        # Ensure model is fitted
        check_is_fitted(self, "model_")
        # Validate input data for prediction
        x_test = check_array(x_test, force_all_finite=True)

        n_samples = len(x_test)
        n_trees = self.model_.n_estimators
        tree_preds = np.zeros((n_samples, n_trees))
        tree_var_preds = np.zeros((n_samples, n_trees))
        tree_count_samples = np.zeros((n_samples, n_trees))

        for sel_tree in range(n_trees):
            leaf_indices = self.model_.estimators_[sel_tree].apply(x_test)

            leaf_vals = self.uc_data_[sel_tree]["leaf_avg_vals"]
            tree_preds[:, sel_tree] = [leaf_vals[leaf_idx] for leaf_idx in leaf_indices]

            leaf_var_vals = self.uc_data_[sel_tree]["leaf_var_vals"]
            tree_var_preds[:, sel_tree] = [
                leaf_var_vals[leaf_idx] for leaf_idx in leaf_indices
            ]

            leaf_count_vals = self.uc_data_[sel_tree]["leaf_counts"]
            tree_count_samples[:, sel_tree] = [
                leaf_count_vals[leaf_idx] for leaf_idx in leaf_indices
            ]

        # #####OPTION ABEL 1st attempt:
        # mean_pred = np.mean(tree_preds, axis=1)
        # var_pred = 1/(n_trees**2)*np.sum(tree_var_preds, axis=1)
        # #####OPTION what it seems researchers do:
        mean_pred = np.sum(tree_preds, axis=1)
        var_pred = np.sum(tree_var_preds, axis=1)
        count_samples = np.sum(tree_count_samples, axis=1)

        return mean_pred, var_pred, count_samples
