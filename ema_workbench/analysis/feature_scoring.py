'''

Feature scoring functionality 


'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

from operator import itemgetter

import numpy as np
import numpy.lib.recfunctions as recfunctions
import pandas as pd
import six
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.feature_selection.univariate_selection import (f_regression,
                                                            f_classif, chi2)
from sklearn.linear_model.least_angle import LassoLarsCV
from sklearn.linear_model.randomized_l1 import (RandomizedLogisticRegression,
                                                RandomizedLasso)

from .scenario_discovery_util import CLASSIFICATION, REGRESSION

# Created on Jul 9, 2014
# 
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
#
# TODO:: look at http://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_recovery.html#example-linear-model-plot-sparse-recovery-py


__al__ = ['F_REGRESSION', 'F_CLASSIFICATION', 'CHI2', 
          'get_univariate_feature_scores', 'get_rf_feature_scores',
          'get_lasso_feature_scores', 'get_feature_scores_all']

F_REGRESSION = f_regression


F_CLASSIFICATION = f_classif

CHI2 = chi2


def _prepare_experiments(experiments):
    '''
    transform the experiments structured array into a numpy array.

    Parameters
    ----------
    experiments : structured array
    
    Returns
    -------
    ndarray
    
    '''
    experiments = recfunctions.drop_fields(experiments, "scenario_id", 
                                           asrecarray=True)
    uncs = recfunctions.get_names(experiments.dtype)

    temp_experiments = np.zeros((experiments.shape[0], len(uncs)))
    
    for i, u in enumerate(uncs):
        try: 
            temp_experiments[:,i] = experiments[u].astype(np.float)
        except ValueError:
            
            data = experiments[u]
            entries = sorted(list(set(data)))
            
            for j, entry in enumerate(entries):
                temp_experiments[data==entry,i] = j
    
    return temp_experiments, uncs


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
    if isinstance(classify, six.string_types):
        try:
            y = outcomes[classify]
        except KeyError as e:
            raise e
        categorical = False
    elif callable(classify):
        y = classify(outcomes)
        categorical=True
    else:
        raise TypeError("unknown type for classify")
    
    return y, categorical


def get_univariate_feature_scores(x,y, score_func=F_CLASSIFICATION):
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
    
    return pvalues


def get_rf_feature_scores(x, y, mode=CLASSIFICATION, nr_trees=250, 
                          max_features='auto', max_depth=None, 
                          min_samples_split=2, min_samples_leaf=1, 
                          bootstrap=True, oob_score=True, random_state=None): 
    '''
    Get feature scores using a random forest

    Parameters
    ----------
    x : structured array
    y : 1D nd.array
    mode : {CLASSIFICATION, REGRESSION}
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
    
    if mode==CLASSIFICATION:
        rfc = RandomForestClassifier
        criterion='gini'
    elif mode==REGRESSION:
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
    forest.fit(x,y)

    importances = forest.feature_importances_

    importances = zip(uncs, importances)
    importances = list(importances)
    importances.sort(key=itemgetter(1), reverse=True)

    importances = pd.DataFrame(importances)

    return importances, forest


def get_lasso_feature_scores(x, y, mode=CLASSIFICATION, scaling=0.5, 
                             sample_fraction=0.75, n_resampling=200,
                             random_state=None):
    '''
    Calculate features scores using a randomized lasso (regression) or 
    randomized logistic regression (classification). This is also known as 
    stability selection.
    
    see http://scikit-learn.org/stable/modules/feature_selection.html for 
    details. 
    
    Parameters
    ----------   
    x : structured array
    y : 1D nd.array
    mode : {CLASSIFICATION, REGRESSION}
    scaling : float, optional
              scaling parameter, should be between 0 and 1
    sample_fraction : float, optional
                      the fraction of samples to used in each randomized 
                      dataset
    n_resmpling : int, optional
                  the number of times the model is trained on a random subset 
                  of the data
    random_state : int, optional
                   if it is an int, it specifies the seed to use, defaults to 
                   None.
                         
    Returns
    -------
    pandas DataFrame
        sorted in descending order of tuples with uncertainty and feature 
        scores         
         
    '''
    x, uncs = _prepare_experiments(x)
    
    if mode==CLASSIFICATION:

        lfs = RandomizedLogisticRegression(scaling=scaling, 
                                           sample_fraction=sample_fraction,
                                           n_resampling=n_resampling, 
                                           random_state=random_state)
        lfs.fit(x,y)
    elif  mode==REGRESSION:
        # we use LassoLarsCV to determine alpha see
        # http://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_recovery.html
        lars_cv = LassoLarsCV(cv=6).fit(x, y,)
        alphas = np.linspace(lars_cv.alphas_[0], .1 * lars_cv.alphas_[0], 6)
        
        # fit the randomized lasso        
        lfs = RandomizedLasso(alpha=alphas,scaling=scaling, 
                              sample_fraction=sample_fraction,
                              n_resampling=n_resampling,
                              random_state=random_state)
        lfs.fit(x, y)
    else:
        raise ValueError('{} invalid value for mode'.format(mode))

    importances = lfs.scores_
    importances = zip(uncs, importances)
    importances = list(importances)
    importances.sort(key=itemgetter(1), reverse=True)
    importances = pd.DataFrame(importances)


    return importances

def get_ex_feature_scores(x, y, mode=CLASSIFICATION, nr_trees=250, 
                          max_features='auto', max_depth=None, 
                          min_samples_split=2, min_samples_leaf=1, 
                          min_weight_fraction_leaf=0, max_leaf_nodes=None,
                          bootstrap=True, oob_score=True, random_state=None): 
    '''
    Get feature scores using extra trees

    Parameters
    ----------
    x : structured array
    y : 1D nd.array
    mode : {CLASSIFICATION, REGRESSION}
    nr_trees : int, optional
               nr. of trees in forest (default=250)
    max_features : int, optional
                   see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
    max_depth : int, optional 
                see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
    min_samples_split : int, optional
                  see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
    min_samples_leaf : int, optional
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
    
    if mode==CLASSIFICATION:
        etc = ExtraTreesClassifier
        criterion='gini'
    elif mode==REGRESSION:
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
    extra_trees.fit(x,y)

    importances = extra_trees.feature_importances_

    importances = zip(uncs, importances)
    importances = list(importances)
    importances.sort(key=itemgetter(1), reverse=True)

    importances = pd.DataFrame(importances)

    return importances, extra_trees

algorithms = {'extra trees' : get_ex_feature_scores,
              'lasso' : get_lasso_feature_scores,
              'random forest' : get_rf_feature_scores,
              'univariate' : get_univariate_feature_scores}

def get_feature_scores_all(x, y, alg='extra trees', mode=REGRESSION,
                   **kwargs):
    '''perform feature scoring for all outcomes using the specified feature 
    scoring algorithm
    
    Parameters
    ----------
    x : numpy structured array
    y : dict of 1d numpy arrays
        the outcomes, with a string as key, and a 1D array for each outcome
    alg : {'extra trees', 'lasso', 'random forest', 'univariate'}, optional
    mode : {REGRESSION, CLASSIFICATION}, optional
    kwargs : dict, optional
             any remaining keyword arguments will be passed to the specific
             feature scoring algorithm
    
    Returns
    -------
    DataFrame instance
    
    
    '''
    complete = None
    for key, value in y.items():
        fs, _ = algorithms[alg](x,value,mode=mode, **kwargs)

        fs = fs.rename(columns={1:key})
        fs = fs.set_index(0)

        if complete is None:
            complete = fs.T
        else:
            complete = complete.append(fs.T)

    return complete.T

