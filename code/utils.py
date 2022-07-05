import numpy as np
from numpy import ma
import pandas as pd
from sklearn.utils import check_random_state
from sklearn.tree import BaseDecisionTree
from sklearn.tree import DecisionTreeRegressor
from forest import ForestRegressor
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def quantile_loss(y, pred, q):
    '''
    q: Quantile to be evaluated, e.g., 0.5 for median.
    y: True value.
    pred: Fitted (predicted) value.
    '''
    e = y - pred
    
    return np.maximum(q * e, (q - 1) * e).mean()


def find_path_skTree(node_numb, path, leaf, children_left, children_right):
    '''
    This function is used to find the path of nodes that are visited before reaching a leaf
    '''
    path.append(node_numb)
    if node_numb == leaf:
        return True
    left = False
    right = False
    if (children_left[node_numb] != -1):
        left = find_path_skTree(children_left[node_numb], path, leaf, children_left, children_right)
    if (children_right[node_numb] != -1):
        right = find_path_skTree(children_right[node_numb], path, leaf, children_left, children_right)
    if left or right:
        return True
    path.remove(node_numb)
    return False


def get_rule_skTree(leaf, path, column_names, columns, ID, children_left, feature, threshold):
    '''
    This functions transform the list of nodes composing a path into a set of constraints
    '''
    constraints_leaf = pd.DataFrame(columns=columns)
    for index, node in enumerate(path):
        constraint = pd.DataFrame(data=np.zeros(len(columns)).reshape(1, -1), columns=columns)
        # We check if we are not in the leaf
        if node != leaf:
            # Do we go under or over the threshold ?
            if (children_left[node] == path[index + 1]):
                constraint[column_names[feature[node]]] = 1
                constraint['threshold'] = threshold[node]
            else:
                constraint[column_names[feature[node]]] = -1
                constraint['threshold'] = -(threshold[node] + 0.000001)
            constraint['ID'] = ID
            constraint['leaf'] = leaf
            constraints_leaf = pd.concat([constraints_leaf, constraint])
    return constraints_leaf



def get_rule_skTree2(leaf, path, column_names, columns, ID, children_left, feature, threshold):
    '''
    This functions transform the list of nodes composing a path into a set of constraints
    '''
    constraints_leaf = pd.DataFrame(columns=columns)
    for index, node in enumerate(path):
        constraint = pd.DataFrame(data=np.zeros(len(columns)).reshape(1, -1), columns=columns)
        # We check if we are not in the leaf
        if node != leaf:
            # Do we go under or over the threshold ?
            if (children_left[node] == path[index + 1]):
                constraint[column_names[feature[node]]] = 1
                constraint['threshold'] = threshold[node]
            else:
                constraint[column_names[feature[node]]] = -1
                constraint['threshold'] = -(threshold[node] + 0.000001)
            constraint['ID'] = ID
            #constraint['leaf'] = leaf
            constraints_leaf = pd.concat([constraints_leaf, constraint])
    return constraints_leaf


def generate_sample_indices(random_state, n_samples):
    """
    Generates bootstrap indices for each tree fit.
    Parameters
    ----------
    random_state: int, RandomState instance or None
        If int, random_state is the seed used by the random number generator.
        If RandomState instance, random_state is the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.
    n_samples: int
        Number of samples to generate from each tree.
    Returns
    -------
    sample_indices: array-like, shape=(n_samples), dtype=np.int32
        Sample indices.
    """
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples)
    return sample_indices


def weighted_percentile(a, q, weights=None, sorter=None):
    """
    Returns the weighted percentile of a at q given weights.

    Parameters
    ----------
    a: array-like, shape=(n_samples,)
        samples at which the quantile.

    q: int
        quantile.

    weights: array-like, shape=(n_samples,)
        weights[i] is the weight given to point a[i] while computing the
        quantile. If weights[i] is zero, a[i] is simply ignored during the
        percentile computation.

    sorter: array-like, shape=(n_samples,)
        If provided, assume that a[sorter] is sorted.

    Returns
    -------
    percentile: float
        Weighted percentile of a at q.

    References
    ----------
    1. https://en.wikipedia.org/wiki/Percentile#The_Weighted_Percentile_method

    Notes
    -----
    Note that weighted_percentile(a, q) is not equivalent to
    np.percentile(a, q). This is because in np.percentile
    sorted(a)[i] is assumed to be at quantile 0.0, while here we assume
    sorted(a)[i] is given a weight of 1.0 / len(a), hence it is at the
    1.0 / len(a)th quantile.
    """
    if weights is None:
        weights = np.ones_like(a)
    if q > 100 or q < 0:
        raise ValueError("q should be in-between 0 and 100, "
                        "got %d" % q)

    a = np.asarray(a, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if len(a) != len(weights):
        raise ValueError("a and weights should have the same length.")

    if sorter is not None:
        a = a[sorter]
        weights = weights[sorter]

    nz = weights != 0
    a = a[nz]
    weights = weights[nz]

    if sorter is None:
        sorted_indices = np.argsort(a)
        sorted_a = a[sorted_indices]
        sorted_weights = weights[sorted_indices]
    else:
        sorted_a = a
        sorted_weights = weights

    # Step 1
    sorted_cum_weights = np.cumsum(sorted_weights)
    total = sorted_cum_weights[-1]

    # Step 2
    partial_sum = 100.0 / total * (sorted_cum_weights - sorted_weights / 2.0)
    start = np.searchsorted(partial_sum, q) - 1
    if start == len(sorted_cum_weights) - 1:
        return sorted_a[-1]
    if start == -1:
        return sorted_a[0]

    # Step 3.
    fraction = (q - partial_sum[start]) / (partial_sum[start + 1] - partial_sum[start])
    return sorted_a[start] + fraction * (sorted_a[start + 1] - sorted_a[start])



class BaseTreeQuantileRegressor(BaseDecisionTree):
    def predict(self, X, quantile=None, check_input=False):
        """
        Predict regression value for X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        quantile : int, optional
            Value ranging from 0 to 100. By default, the mean is returned.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        y : array of shape = [n_samples]
            If quantile is set to None, then return E(Y | X). Else return
            y such that F(Y=y | x) = quantile.
        """
        # apply method requires X to be of dtype np.float32
        if quantile is None:
            return super(BaseTreeQuantileRegressor, self).predict(X, check_input=check_input)

        quantiles = np.zeros(X.shape[0])
        X_leaves = self.apply(X)
        unique_leaves = np.unique(X_leaves)
        for leaf in unique_leaves:
            quantiles[X_leaves == leaf] = weighted_percentile(
                self.y_train_[self.y_train_leaves_ == leaf], quantile)
        return quantiles

    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None):
        """
        Build a decision tree classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels) as integers or strings.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        X_idx_sorted : array-like, shape = [n_samples, n_features], optional
            The indexes of the sorted training input samples. If many tree
            are grown on the same dataset, this allows the ordering to be
            cached between trees. If None, the data will be sorted here.
            Don't use this parameter unless you know what to do.

        Returns
        -------
        self : object
            Returns self.
        """
        # y passed from a forest is 2-D. This is to silence the
        # annoying data-conversion warnings.
        y = np.asarray(y)
        if np.ndim(y) == 2 and y.shape[1] == 1:
            y = np.ravel(y)
        # apply method requires X to be of dtype np.float32
        super(BaseTreeQuantileRegressor, self).fit(
            X, y, sample_weight=sample_weight, check_input=check_input,
            X_idx_sorted=X_idx_sorted)
        self.y_train_ = y

        # Stores the leaf nodes that the samples lie in.
        self.y_train_leaves_ = self.tree_.apply(X)
        return self


class DecisionTreeQuantileRegressor(DecisionTreeRegressor, BaseTreeQuantileRegressor):
    """A decision tree regressor that provides quantile estimates.

    Parameters
    ----------
    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.
        .. versionadded:: 0.18
        Mean Absolute Error (MAE) criterion.

    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
            number of samples for each split.
        .. versionchanged:: 0.18
        Added float values for percentages.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
        number of samples for each node.
        .. versionchanged:: 0.18
        Added float values for percentages.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    feature_importances_ : array of shape = [n_features]
        The feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the
        (normalized) total reduction of the criterion brought
        by that feature. It is also known as the Gini importance [4]_.

    max_features_ : int,
        The inferred value of max_features.

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree object
        The underlying Tree object.

    y_train_ : array-like
        Train target values.

    y_train_leaves_ : array-like.
        Cache the leaf nodes that each training sample falls into.
        y_train_leaves_[i] is the leaf that y_train[i] ends up at.
    """
    def __init__(self,
                criterion="mse",
                splitter="best",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.,
                max_features=None,
                random_state=None,
                max_leaf_nodes=None):
        super(DecisionTreeQuantileRegressor, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state)


class BaseForestQuantileRegressor(ForestRegressor):
    def fit(self, X, y):
        """
        Build a forest from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels) as integers or strings.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        X_idx_sorted : array-like, shape = [n_samples, n_features], optional
            The indexes of the sorted training input samples. If many tree
            are grown on the same dataset, this allows the ordering to be
            cached between trees. If None, the data will be sorted here.
            Don't use this parameter unless you know what to do.

        Returns
        -------
        self : object
            Returns self.
        """
        # apply method requires X to be of dtype np.float32
        super(BaseForestQuantileRegressor, self).fit(X, y)

        self.y_train_ = y
        self.y_train_leaves_ = -np.ones((self.n_estimators, len(y)), dtype=np.int32)
        self.y_weights_ = np.zeros_like((self.y_train_leaves_), dtype=np.float32)

        for i, est in enumerate(self.estimators_):
            if self.bootstrap:
                bootstrap_indices = generate_sample_indices(
                    est.random_state, len(y))
            else:
                bootstrap_indices = np.arange(len(y))

            est_weights = np.bincount(bootstrap_indices, minlength=len(y))
            y_train_leaves = est.y_train_leaves_
            for curr_leaf in np.unique(y_train_leaves):
                y_ind = y_train_leaves == curr_leaf
                self.y_weights_[i, y_ind] = (
                    est_weights[y_ind] / np.sum(est_weights[y_ind]))

            self.y_train_leaves_[i, bootstrap_indices] = y_train_leaves[bootstrap_indices]
        return self

    def predict(self, X, quantile=None):
        """
        Predict regression value for X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        quantile : int, optional
            Value ranging from 0 to 100. By default, the mean is returned.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        y : array of shape = [n_samples]
            If quantile is set to None, then return E(Y | X). Else return
            y such that F(Y=y | x) = quantile.
        """
        # apply method requires X to be of dtype np.float32
        if quantile is None:
            return super(BaseForestQuantileRegressor, self).predict(X)

        sorter = np.argsort(self.y_train_)
        X_leaves = self.apply(X)
        weights = np.zeros((X.shape[0], len(self.y_train_)))
        quantiles = np.zeros((X.shape[0]))
        for i, x_leaf in enumerate(X_leaves):
            mask = self.y_train_leaves_ != np.expand_dims(x_leaf, 1)
            x_weights = ma.masked_array(self.y_weights_, mask)
            weights = x_weights.sum(axis=0)
            quantiles[i] = weighted_percentile(
                self.y_train_, quantile, weights, sorter)
        return quantiles


class RandomForestQuantileRegressor(BaseForestQuantileRegressor):
    """
    A random forest regressor that provides quantile estimates.

    A random forest is a meta estimator that fits a number of classifying
    decision trees on various sub-samples of the dataset and use averaging
    to improve the predictive accuracy and control over-fitting.
    The sub-sample size is always the same as the original
    input sample size but the samples are drawn with replacement if
    `bootstrap=True` (default).

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.
        .. versionadded:: 0.18
        Mean Absolute Error (MAE) criterion.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
        number of samples for each split.
        .. versionchanged:: 0.18
        Added float values for percentages.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum number of samples for each node.
        .. versionchanged:: 0.18
        Added float values for percentages.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.

    oob_score : bool, optional (default=False)
        whether to use out-of-bag samples to estimate
        the R^2 on unseen data.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    Attributes
    ----------
    estimators_ : list of DecisionTreeQuantileRegressor
        The collection of fitted sub-estimators.

    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_prediction_ : array of shape = [n_samples]
        Prediction computed with out-of-bag estimate on the training set.

    y_train_ : array-like, shape=(n_samples,)
        Cache the target values at fit time.

    y_weights_ : array-like, shape=(n_estimators, n_samples)
        y_weights_[i, j] is the weight given to sample ``j` while
        estimator ``i`` is fit. If bootstrap is set to True, this
        reduces to a 2-D array of ones.

    y_train_leaves_ : array-like, shape=(n_estimators, n_samples)
        y_train_leaves_[i, j] provides the leaf node that y_train_[i]
        ends up when estimator j is fit. If y_train_[i] is given
        a weight of zero when estimator j is fit, then the value is -1.

    References
    ----------
    .. [1] Nicolai Meinshausen, Quantile Regression Forests
        http://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf
    """
    def __init__(self,
                n_estimators=10,
                criterion='mse',
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features='auto',
                max_leaf_nodes=None,
                bootstrap=True,
                oob_score=False,
                n_jobs=1,
                random_state=None,
                verbose=0,
                warm_start=False):
        super(RandomForestQuantileRegressor, self).__init__(
            base_estimator=DecisionTreeQuantileRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                            "min_samples_leaf", "min_weight_fraction_leaf",
                            "max_features", "max_leaf_nodes",
                            "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)
        
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes


class TabularDataset(Dataset):
    def __init__(self, data, cat_cols=None, output_col=None):
        """
        Characterizes a Dataset for PyTorch
        Parameters
        ----------
        data: pandas data frame
        The data frame object for the input data. It must
        contain all the continuous, categorical and the
        output columns to be used.
        cat_cols: List of strings
        The names of the categorical columns in the data.
        These columns will be passed through the embedding
        layers in the model. These columns must be
        label encoded beforehand. 
        output_col: string
        The name of the output variable column in the data
        provided.
        """

        self.n = data.shape[0]

        if output_col:
            self.y = data[output_col].astype(np.float32).values.reshape(-1, 1)
        else:
            self.y =  np.zeros((self.n, 1))

        self.cat_cols = cat_cols if cat_cols else []
        self.cont_cols = [
            col for col in data.columns if col not in self.cat_cols + [output_col]
        ]

        if self.cont_cols:
            self.cont_X = data[self.cont_cols].astype(np.float32).values
        else:
            self.cont_X = np.zeros((self.n, 1))

        if self.cat_cols:
            self.cat_X = data[cat_cols].astype(np.int64).values
        else:
            self.cat_X = np.zeros((self.n, 1))

    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return self.n

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        return [self.y[idx], self.cont_X[idx]]


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles
        self.qv = torch.tensor(quantiles, dtype=torch.float).unsqueeze(0).to(device)
        self.qv_1 = self.qv - 1

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)

        errors = target - preds
        losses = torch.max((self.qv_1) * errors, self.qv * errors).mean(axis=0)
        loss = losses.mean()
        
        ###
        if preds.size(1) > 1:
            diff = preds[:, 1:] - preds[:, :-1]
            penalty = 100 * torch.square(torch.max(torch.zeros(diff.size()), 1e-3 - diff)).mean(axis=0).mean()
            loss = loss + penalty
        
        return loss


class FeedForwardNN(nn.Module):
    def __init__(
        self,
        #emb_dims,
        no_of_cont,
        lin_layer_sizes,
        output_size,
        #emb_dropout,
        lin_layer_dropouts,
    ):

        """
        Parameters
        ----------
        emb_dims: List of two element tuples
        This list will contain a two element tuple for each
        categorical feature. The first element of a tuple will
        denote the number of unique values of the categorical
        feature. The second element will denote the embedding
        dimension to be used for that feature.
        no_of_cont: Integer
        The number of continuous features in the data.
        lin_layer_sizes: List of integers.
        The size of each linear layer. The length will be equal
        to the total number
        of linear layers in the network.
        output_size: Integer
        The size of the final output.
        emb_dropout: Float
        The dropout to be used after the embedding layers.
        lin_layer_dropouts: List of floats
        The dropouts to be used after each linear layer.
        """

        super().__init__()

        self.no_of_cont = no_of_cont

        # Linear Layers
        first_lin_layer = nn.Linear(
            self.no_of_cont, lin_layer_sizes[0]
        )

        self.lin_layers = nn.ModuleList(
            [first_lin_layer]
            + [
                nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1])
                for i in range(len(lin_layer_sizes) - 1)
            ]
        )

        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)

        # Output Layer
        self.output_layer = nn.Linear(lin_layer_sizes[-1], output_size)
        nn.init.kaiming_normal_(self.output_layer.weight.data)


        # Dropout Layers
        self.droput_layers = nn.ModuleList(
            [nn.Dropout(size) for size in lin_layer_dropouts]
        )

    def forward(self, cont_data):

        x = cont_data

        for lin_layer, dropout_layer in zip(
            self.lin_layers, self.droput_layers
        ):

            x = F.relu(lin_layer(x))
            x = dropout_layer(x)

        x = self.output_layer(x)
        return x



def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

def extract_layer(weight, bias, l):
    df_sub = pd.DataFrame(weight[l]).add_prefix('node_')
    df_sub['intercept'] = bias[l]
    df_sub['layer'] = l
    df_sub['node'] = range(len(df_sub))
    return df_sub

def constraint_extrapolation_MLP(weight, bias, names):
    n_layers = len(names)
    constraints = pd.concat([extract_layer(weight, bias, l) for l in range(n_layers)],axis=0)
    cols_to_move = ['intercept', 'layer', 'node']
    constraints = constraints[cols_to_move + [col for col in constraints.columns if col not in cols_to_move]]
    return constraints