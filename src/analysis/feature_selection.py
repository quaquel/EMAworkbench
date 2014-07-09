'''
Created on Jul 9, 2014

@author: jhkwakkel@tudelft.net
'''

import numpy as np
import numpy.lib.recfunctions as recfunctions

from sklearn.ensemble import RandomForestClassifier


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
        
    
def get_feature_scores(results, classify, nr_trees=250, criterion='gini',
                       max_features='auto', max_depth=None, 
                       min_samples_split=2, min_samples_leaf=1, bootstrap=True,
                       oob_score=True): 
    '''
    Get feature scores using a random forest
    
    :param results: results tuple
    :param classify: a classify function analogous to PRIM
    :param nr_trees: nr. of trees in forest (default=250)
    :param criterion:
    :param max_features:
    :param max_depth:
    :param min_samples:
    :param min_samples_leaf:
    :param bootstrap:
    :param oob_score:
    :returns: a list sorted in descending order of tuples with uncertainty and 
              feature scores
    
    '''
    experiments, outcomes = results
    x = _preperate_experiments(experiments)
    y = classify(outcomes)
    
    forest = RandomForestClassifier(n_estimators=nr_trees, 
                                    criterion=criterion, 
                                    max_features=max_features, 
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    bootstrap=bootstrap,
                                    oob_score=oob_score)
    forest.fit(x,y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
    
    # todo what do we want to return
    
    
    
    
    

    