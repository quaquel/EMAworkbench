'''

Feature scoring functionality


'''
from operator import itemgetter
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.feature_selection import (f_regression, f_classif, chi2)

from .scenario_discovery_util import RuleInductionType
from ..util import get_module_logger

# Created on Jul 9, 2014
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
#
# TODO:: look at
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_recovery.html#example-linear-model-plot-sparse-recovery-py


__all__ = ['F_REGRESSION', 'F_CLASSIFICATION', 'CHI2',
           'get_univariate_feature_scores', 'get_rf_feature_scores',
           'get_ex_feature_scores', 'get_feature_scores_all']

_logger = get_module_logger(__name__)

F_REGRESSION = f_regression
F_CLASSIFICATION = f_classif
CHI2 = chi2


def _prepare_experiments(experiments):
    '''
    transform the experiments structured array into a numpy array.

    Parameters
    ----------
    experiments :DataFrame

    Returns
    -------
    ndarray, list

    '''
    try:
        experiments = experiments.drop('scenario', axis=1)
    except KeyError:
        pass

    x = experiments.copy()

    x_nominal = x.select_dtypes(exclude=np.number)
    x_nominal_columns = x_nominal.columns.values

    for column in x_nominal_columns:
        if np.unique(x[column]).shape == (1,):
            x = x.drop(column, axis=1)
            _logger.info(("{} dropped from analysis "
                          "because only a single category").format(column))
        else:
            x[column] = x[column].astype('category').cat.codes

    return x.values, x.columns.tolist()


def _prepare_outcomes(outcomes, classify):
    '''
    transform the outcomes dict into a vector with either the class allocation
    or the value.

    Parameters
    ----------
    outcomes : dict
               the outcomes dict
    classify : callable or str
               a classify function or variable analogous to PRIM

    Returns
    -------
    1d ndarray
        the return from classify
    bool
        data is categorical (True) or continuous (False)

    Raises
    --------
    TypeError
        if classify is neither a StringType nor a callable
    KeyError
        if classify is a string which is not a key in the outcomes dict.

    '''
    if isinstance(classify, str):
        try:
            y = outcomes[classify]
        except KeyError as e:
            raise e
        categorical = False
    elif callable(classify):
        y = classify(outcomes)
        categorical = True
    else:
        raise TypeError("unknown type for classify")

    return y, categorical


def get_univariate_feature_scores(x, y, score_func=F_CLASSIFICATION):
    '''

    calculate feature scores using univariate statistical tests. In case of
    categorical data, chi square or the Anova F value is used. In case of
    continuous data the Anova F value is used.

    Parameters
    ----------
    x : structured array
    y : 1D nd.array
    score_func : {F_CLASSIFICATION, F_REGRESSION, CHI2}
                the score function to use, one of f_regression (regression), or
                f_classification or chi2 (classification).
    Returns
    -------
    pandas DataFrame
        sorted in descending order of tuples with uncertainty and feature
        scores (i.e. p values in this case).


    '''
    x, uncs = _prepare_experiments(x)

    pvalues = score_func(x, y)[1]
    pvalues = np.asarray(pvalues)

    pvalues = zip(uncs, pvalues)
    pvalues = list(pvalues)
    pvalues.sort(key=itemgetter(1))

    pvalues = pd.DataFrame(pvalues)
    pvalues = pvalues.set_index(0)

    return pvalues


def get_rf_feature_scores(x, y, mode=RuleInductionType.CLASSIFICATION,
                          nr_trees=250,
                          max_features='auto', max_depth=None,
                          min_samples_split=2, min_samples_leaf=1,
                          bootstrap=True, oob_score=True, random_state=None):
    '''
    Get feature scores using a random forest

    Parameters
    ----------
    x : structured array
    y : 1D nd.array
    mode : {RuleInductionType.CLASSIFICATION, RuleInductionType.REGRESSION}
    nr_trees : int, optional
               nr. of trees in forest (default=250)
    max_features : int, optional
                   see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    max_depth : int, optional
                see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    min_samples : int, optional
                  see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    min_samples_leaf : int, optional
                       see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    bootstrap : bool, optional
                see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    oob_score : bool, optional
                see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    random_state : int, optional
                   see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

    Returns
    -------
    pandas DataFrame
        sorted in descending order of tuples with uncertainty and feature
        scores
    object
        either RandomForestClassifier or RandomForestRegressor

    '''
    x, uncs = _prepare_experiments(x)

    if mode == RuleInductionType.CLASSIFICATION:
        rfc = RandomForestClassifier
        criterion = 'gini'
    elif mode == RuleInductionType.REGRESSION:
        rfc = RandomForestRegressor
        criterion = 'mse'
    else:
        raise ValueError('{} not valid for mode'.format(mode))

    forest = rfc(n_estimators=nr_trees,
                 criterion=criterion,
                 max_features=max_features,
                 max_depth=max_depth,
                 min_samples_split=min_samples_split,
                 min_samples_leaf=min_samples_leaf,
                 bootstrap=bootstrap,
                 oob_score=oob_score,
                 random_state=random_state)
    forest.fit(x, y)

    importances = forest.feature_importances_

    importances = zip(uncs, importances)
    importances = list(importances)
    importances.sort(key=itemgetter(1), reverse=True)

    importances = pd.DataFrame(importances)
    importances = importances.set_index(0)

    return importances, forest


def get_ex_feature_scores(x, y, mode=RuleInductionType.CLASSIFICATION,
                          nr_trees=100, max_features=None, max_depth=None,
                          min_samples_split=2, min_samples_leaf=None,
                          min_weight_fraction_leaf=0, max_leaf_nodes=None,
                          bootstrap=True, oob_score=True, random_state=None):
    '''
    Get feature scores using extra trees

    Parameters
    ----------
    x : structured array
    y : 1D nd.array
    mode : {RuleInductionType.CLASSIFICATION, RuleInductionType.REGRESSION}
    nr_trees : int, optional
               nr. of trees in forest (default=250)
    max_features : int, float, string or None, optional
                   by default, it will use number of featers/3, following
                   Jaxa-Rozen & Kwakkel (2018) doi: 10.1016/j.envsoft.2018.06.011
                   see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
    max_depth : int, optional
                see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
    min_samples_split : int, optional
                  see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
    min_samples_leaf : int, optional
                       defaults to 1 for N=1000 or lower, from there on
                       proportional to sqrt of N
                       (see discussion in Jaxa-Rozen & Kwakkel (2018) doi: 10.1016/j.envsoft.2018.06.011)
                       see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
    min_weight_fraction_leaf : float, optional
                               see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
    max_leaf_nodes: int or None, optional
                    see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
    bootstrap : bool, optional
                see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
    oob_score : bool, optional
                see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
    random_state : int, optional
                   see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html

    Returns
    -------
    pandas DataFrame
        sorted in descending order of tuples with uncertainty and feature
        scores
    object
        either ExtraTreesClassifier or ExtraTreesRegressor

    '''
    x, uncs = _prepare_experiments(x)

    # TODO
    # max_features = number of variables/3
    #
    # min_samples_leaf
    # 1000 - >
    # then proportional based on sqrt of N
    # dus sqrt(N) / Sqrt(1000) met 1 als minimumd
    if max_features is None:
        max_features = int(round(x.shape[1] / 3))
    if min_samples_leaf is None:
        min_samples_leaf = max(1,
                               int(round(math.sqrt(x.shape[0]) / math.sqrt(1000))))

    if mode == RuleInductionType.CLASSIFICATION:
        etc = ExtraTreesClassifier
        criterion = 'gini'
    elif mode == RuleInductionType.REGRESSION:
        etc = ExtraTreesRegressor
        criterion = 'mse'
    else:
        raise ValueError('{} not valid for mode'.format(mode))

    extra_trees = etc(n_estimators=nr_trees,
                      criterion=criterion,
                      max_features=max_features,
                      max_depth=max_depth,
                      min_samples_split=min_samples_split,
                      min_samples_leaf=min_samples_leaf,
                      min_weight_fraction_leaf=min_weight_fraction_leaf,
                      max_leaf_nodes=max_leaf_nodes,
                      bootstrap=bootstrap,
                      oob_score=oob_score,
                      random_state=random_state)
    extra_trees.fit(x, y)

    importances = extra_trees.feature_importances_

    importances = zip(uncs, importances)
    importances = list(importances)
    importances.sort(key=itemgetter(1), reverse=True)

    importances = pd.DataFrame(importances)
    importances = importances.set_index(0)

    return importances, extra_trees


algorithms = {'extra trees': get_ex_feature_scores,
              'random forest': get_rf_feature_scores,
              'univariate': get_univariate_feature_scores}


def get_feature_scores_all(x, y, alg='extra trees',
                           mode=RuleInductionType.REGRESSION,
                           **kwargs):
    '''perform feature scoring for all outcomes using the specified feature
    scoring algorithm

    Parameters
    ----------
    x : numpy structured array
    y : dict of 1d numpy arrays
        the outcomes, with a string as key, and a 1D array for each outcome
    alg : {'extra trees', 'random forest', 'univariate'}, optional
    mode : {RuleInductionType.REGRESSION, RuleInductionType.CLASSIFICATION}, optional
    kwargs : dict, optional
             any remaining keyword arguments will be passed to the specific
             feature scoring algorithm

    Returns
    -------
    DataFrame instance


    '''
    complete = None
    for key, value in y.items():
        fs, _ = algorithms[alg](x, value, mode=mode, **kwargs)

        fs = fs.rename(columns={1: key})

        if complete is None:
            complete = fs.T
        else:
            complete = complete.append(fs.T, sort=True)

    return complete.T
