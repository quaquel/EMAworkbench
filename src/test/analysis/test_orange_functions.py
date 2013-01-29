'''
Created on 21 nov. 2012

@author: localadmin
'''
import numpy as np

from expWorkbench.EMAlogging import log_to_stderr, INFO
from analysis.orange_functions import tree, random_forest, feature_selection,\
                                     random_forest_measure_attributes
from expWorkbench import ModelEnsemble
from test.test_vensim_flu import FluModel


def classify(data):
    '''
    helper function for classifying data. This is merely an example 
    implementation. Any user supplied function can be used as long as it
    accepts data and returns a 1-D array of classes.
    
    :param data: list of dicts, which each dict containing the results for all
                 outcomes of interest. This is the results as returned by
                 run_experiments
    :rtype: 1-D array of classes 
    
    '''
    
    data = data.get('deceased population region 1')
    data = data[:, -1]
    data = data/np.max(data)*10
    a = np.zeros(data.shape)
    b = np.max(data) - np.min(data)
    b = b/5
    for i in range(5):
        a[(data < b*(i+1))&
          (data > b*i)] = i

    return a

def test_tree():
    
    log_to_stderr(level= INFO)
        
    model = FluModel(r'..\..\models\flu', "fluCase")
    ensemble = ModelEnsemble()
    ensemble.parallel = True
    ensemble.set_model_structure(model)
    
    policies = [{'name': 'no policy',
                 'file': r'\FLUvensimV1basecase.vpm'},
                {'name': 'static policy',
                 'file': r'\FLUvensimV1static.vpm'},
                {'name': 'adaptive policy',
                 'file': r'\FLUvensimV1dynamic.vpm'}
                ]
    ensemble.add_policies(policies)
    
    results = ensemble.perform_experiments(10)
   
    a_tree = tree(results, classify)
    

def test_random_forest_importance():
    log_to_stderr(level= INFO)
        
    model = FluModel(r'..\..\models\flu', "fluCase")
    ensemble = ModelEnsemble()
    ensemble.parallel = True
    ensemble.set_model_structure(model)
    
    policies = [{'name': 'no policy',
                 'file': r'\FLUvensimV1basecase.vpm'},
                {'name': 'static policy',
                 'file': r'\FLUvensimV1static.vpm'},
                {'name': 'adaptive policy',
                 'file': r'\FLUvensimV1dynamic.vpm'}
                ]
    ensemble.add_policies(policies)
    
    results = ensemble.perform_experiments(10000)
   
    results = random_forest_measure_attributes(results, classify)
    for entry in results:
        print entry[0] +"\t" + str(entry[1])

def test_feature_selection():
    log_to_stderr(level= INFO)
        
    model = FluModel(r'..\..\models\flu', "fluCase")
    ensemble = ModelEnsemble()
    ensemble.parallel = True
    ensemble.set_model_structure(model)
    
    policies = [{'name': 'no policy',
                 'file': r'\FLUvensimV1basecase.vpm'},
                {'name': 'static policy',
                 'file': r'\FLUvensimV1static.vpm'},
                {'name': 'adaptive policy',
                 'file': r'\FLUvensimV1dynamic.vpm'}
                ]
    ensemble.add_policies(policies)
    
    results = ensemble.perform_experiments(5000)
   
    results = feature_selection(results, classify)
    for entry in results:
        print entry[0] +"\t" + str(entry[1])