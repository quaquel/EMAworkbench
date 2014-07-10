'''
Created on Jul 9, 2014

@author: jhkwakkel@tudelft.net


TODO:: look at http://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_recovery.html#example-linear-model-plot-sparse-recovery-py

'''

import numpy as np
import numpy.lib.recfunctions as recfunctions

from sklearn.ensemble import RandomForestClassifier
from operator import itemgetter


def _preperate_experiments(experiments):
    '''
    transform the experiments structured array into a numpy array.
    
    :poram experiments:
    :returns temp_experiments:
    
    '''
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
    
    return temp_experiments
        
    
def get_rf_feature_scores(results, classify, nr_trees=250, criterion='gini',
                       max_features='auto', max_depth=None, 
                       min_samples_split=2, min_samples_leaf=1, bootstrap=True,
                       oob_score=True, random_state=None): 
    '''
    Get feature scores using a random forest
    
    :param results: results tuple
    :param classify: a classify function analogous to PRIM
    :param nr_trees: nr. of trees in forest (default=250)
    :param criterion: see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    :param max_features: see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    :param max_depth: see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    :param min_samples: see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    :param min_samples_leaf: see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    :param bootstrap: see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    :param oob_score: see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    :param random_state: see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    :returns: a list sorted in descending order of tuples with uncertainty and 
              feature scores, and the forest itself
    
    '''
    experiments, outcomes = results
    uncs = recfunctions.get_names(experiments.dtype)
    
    x = _preperate_experiments(experiments)
    y = classify(outcomes)
    
    forest = RandomForestClassifier(n_estimators=nr_trees, 
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
    importances.sort(key=itemgetter(1), reverse=True)
    
    return importances, forest
    
    
    

    