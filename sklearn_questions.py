"""Assignment - making a sklearn estimator and cv splitter.

The goal of this assignment is to implement by yourself:

- a scikit-learn estimator for the KNearestNeighbors for classification
  tasks and check that it is working properly.
- a scikit-learn CV splitter where the splits are based on a Pandas
  DateTimeIndex.

Detailed instructions for question 1:
The nearest neighbor classifier predicts for a point X_i the target y_k of
the training sample X_k which is the closest to X_i. We measure proximity with
the Euclidean distance. The model will be evaluated with the accuracy (average
number of samples corectly classified). You need to implement the `fit`,
`predict` and `score` methods for this class. The code you write should pass
the test we implemented. You can run the tests by calling at the root of the
repo `pytest test_sklearn_questions.py`. Note that to be fully valid, a
scikit-learn estimator needs to check that the input given to `fit` and
`predict` are correct using the `validate_data, check_is_fitted` functions
imported in this file.
You can find more information on how they should be used in the following doc:
https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator.
Make sure to use them to pass `test_nearest_neighbor_check_estimator`.


Detailed instructions for question 2:
The data to split should contain the index or one column in
datatime format. Then the aim is to split the data between train and test
sets when for each pair of successive months, we learn on the first and
predict of the following. For example if you have data distributed from
november 2020 to march 2021, you have have 4 splits. The first split
will allow to learn on november data and predict on december data, the
second split to learn december and predict on january etc.

We also ask you to respect the pep8 convention: https://pep8.org. This will be
enforced with `flake8`. You can check that there is no flake8 errors by
calling `flake8` at the root of the repo.

Finally, you need to write docstrings for the methods you code and for the
class. The docstring will be checked using `pydocstyle` that you can also
call at the root of the repo.

Hints
-----
- You can use the function:

from sklearn.metrics.pairwise import pairwise_distances

to compute distances between 2 sets of samples.
"""
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.multiclass import type_of_target

from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import validate_data
from sklearn.metrics.pairwise import pairwise_distances


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1):  # noqa: D107
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fitting function.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to train the model.
        y : ndarray, shape (n_samples,)
            Labels associated with the training data.

        Returns
        -------
        self : instance of KNearestNeighbors
            The current instance of the classifier
        """
        if not isinstance(self.n_neighbors, int) or self.n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer.")

        X_valid, y_valid = validate_data(self, X, y=y)

        target_type = type_of_target(y_valid)
        if target_type not in ("binary", "multiclass"):
            raise ValueError(
                f"Unknown label type: {target_type}. "
                "This estimator supports only classification targets."
            )

        self.X_ = X_valid
        self.y_ = y_valid
        self.classes_ = np.unique(y_valid)

        return self

    def predict(self, X):
        """Predict function.

        Parameters
        ----------
        X : ndarray, shape (n_test_samples, n_features)
            Data to predict on.

        Returns
        -------
        y : ndarray, shape (n_test_samples,)
            Predicted class labels for each test data sample.
        """
        check_is_fitted(self, attributes=("X_", "y_", "classes_"))
        X_valid = validate_data(self, X, reset=False)

        distances = pairwise_distances(X_valid, self.X_)
        n_train = self.X_.shape[0]
        k = min(self.n_neighbors, n_train)

        y_pred = np.empty(X_valid.shape[0], dtype=self.y_.dtype)

        for i in range(X_valid.shape[0]):
            nn_idx = np.argpartition(distances[i], k - 1)[:k]
            nn_labels = self.y_[nn_idx]

            labels, counts = np.unique(nn_labels, return_counts=True)
            y_pred[i] = labels[np.argmax(counts)]

        return y_pred

    def score(self, X, y):
        """Calculate the score of the prediction.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to score on.
        y : ndarray, shape (n_samples,)
            target values.

        Returns
        ----------
        score : float
            Accuracy of the model computed for the (X, y) pairs.
        """
        X_valid, y_valid = validate_data(self, X, y=y, reset=False)
        y_pred = self.predict(X_valid)
        a = np.mean(y_pred == y_valid)
        return float(a)


class MonthlySplit(BaseCrossValidator):
    """CrossValidator based on monthly split.

    Split data based on the given `time_col` (or default to index). Each split
    corresponds to one month of data for the training and the next month of
    data for the test.

    Parameters
    ----------
    time_col : str, defaults to 'index'
        Column of the input DataFrame that will be used to split the data. This
        column should be of type datetime. If split is called with a DataFrame
        for which this column is not a datetime, it will raise a ValueError.
        To use the index as column just set `time_col` to `'index'`.
    """

    def __init__(self, time_col='index'):  # noqa: D107
        self.time_col = time_col

    def _get_time_values(self, X):
        if self.time_col == "index":
            if not hasattr(X, "index"):
                raise ValueError("X has no index; cannot use time_col='index'."
                                 )
            time_values = X.index
        else:
            if not hasattr(X, "__getitem__"):
                raise ValueError("X must support column access for time_col.")
            time_values = X[self.time_col]

        if isinstance(time_values, pd.DatetimeIndex):
            return time_values

        if isinstance(time_values, pd.Series):
            if not pd.api.types.is_datetime64_any_dtype(time_values.dtype):
                raise ValueError("time_col must be datetime dtype.")
            return pd.DatetimeIndex(time_values)

        raise ValueError("time_col must be a pandas datetime index/series.")

    def _get_month_periods(self, X):
        dt_index = self._get_time_values(X)
        return dt_index.to_period("M")

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            The number of splits.
        """
        periods = self._get_month_periods(X)
        unique_months = pd.Index(periods.unique()).sort_values()
        return max(int(len(unique_months) - 1), 0)

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Yields
        ------
        idx_train : ndarray
            The training set indices for that split.
        idx_test : ndarray
            The testing set indices for that split.
        """

        periods = self._get_month_periods(X)
        unique_months = pd.Index(periods.unique()).sort_values()

        for i in range(len(unique_months) - 1):
            train_month = unique_months[i]
            test_month = unique_months[i + 1]

            idx_train = np.flatnonzero(periods == train_month)
            idx_test = np.flatnonzero(periods == test_month)

            yield idx_train, idx_test
