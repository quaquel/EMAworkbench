'''

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>


a more generic 'user friendly' implementation of the pca prim combi discussed 
in the envsoft manuscript D-12-00217: 
Dalal, S., Han, B., Lempert, R., Jayjocks, A., Hackbarth, A. 
Improving Scneario Discovery using Orhogonal Rotations.

this implementation can cope with subsets that are rotated jointly. This 
implementation is data type aware, so categorical variables are not rotated. 

'''
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from expWorkbench import load_results, ema_logging, EMAError
from analysis import prim

def determine_rotation(experiments):
    '''
    Determine the rotation for the specified experiments
    
    :param experiments:
    
    '''
    
    covariance = np.cov(experiments.T)
    eigen_vals, eigen_vectors = np.linalg.eig(covariance)

    indices = np.argsort(eigen_vals)
    indices = indices[::-1]
    eigen_vectors = eigen_vectors[:,indices]
    eigen_vals = eigen_vals[indices]
    
    #make the eigen vectors unit length
    for i in range(eigen_vectors.shape[1]):
        eigen_vectors[:,i] / np.linalg.norm(eigen_vectors[:,i]) * np.sqrt(eigen_vals[i])
        
    return eigen_vectors

def assert_dtypes(keys, dtypes):
    '''
    helper fucntion that checks whether none of the provided keys has
    a dtype object as value.
    '''
    
    for key in keys:
        if dtypes[key][0] == np.dtype(object):
            raise EMAError("%s has dtype object and can thus not be rotated" %key)
    return True

def rotate_subset(value, orig_experiments, logical): 
    '''
    rotate a subset
    
    :param value:
    :param orig_experiment:
    :param logical:
    
    '''
    
     
    list_dtypes = [(name, "<f8") for name in value]
    
    #cast everything to float
    subset_experiments = orig_experiments[value].astype(list_dtypes).view('<f8').reshape(orig_experiments.shape[0], len(value))

    #normalize the data
    mean = np.mean(subset_experiments,axis=0)
    std = np.std(subset_experiments, axis=0)
    std[std==0] = 1 #in order to avoid a devision by zero
    subset_experiments = (subset_experiments - mean)/std
    
    #get the experiments of interest
    experiments_of_interest = subset_experiments[logical]
    
    #determine the rotation
    rotation_matrix =  determine_rotation(experiments_of_interest)
    
    #apply the rotation
    subset_experiments = np.dot(subset_experiments,rotation_matrix)
    return rotation_matrix, subset_experiments

def perform_pca_prim(results, 
                     classify, 
                     exclude=['model', 'policy'],
                     subsets={},
                     peel_alpha = 0.05, 
                     paste_alpha = 0.05,
                     mass_min = 0.05, 
                     threshold = None, 
                     pasting=True, 
                     threshold_type=1,
                     obj_func=prim.def_obj_func):
    '''
    Perform (un)constrained PCA-PRIM. The cases of interest are identified. 
    Next, the experiments are rotated to the eigen space of the covariance 
    matrix of the experiments of interest. 
    
    :param results: the return from perform_experiments
    :param classify: the classify function to be used in PRIM
    :param exclude: The uncertainties that should be excluded, optional argument
    :param subsets: optional kwarg, expects a dictonary with group name as key
                    and a list of uncertainty names as values. If this is used,
                    a constrained PCA-PRIM is executed
                    **note:** the list of uncertainties should not contain 
                    categorical uncertainties. 
    :param classify: either a string denoting the outcome of interest to use
                     or a function. In case of a string and time series data, 
                     the end state is used.
    :param peel_alpha: parameter controlling the peeling stage (default = 0.05). 
    :param paste_alpha: parameter controlling the pasting stage (default = 0.05).
    :param mass_min: minimum mass of a box (default = 0.05). 
    :param threshold: the threshold of the output space that boxes should meet. 
    :param pasting: perform pasting stage (default=True) 
    :param threshold_type: If 1, the boxes should go above the threshold, if -1
                           the boxes should go below the threshold, if 0, the 
                           algorithm looks for both +1 and -1.
    :param obj_func: The objective function to use. Default is 
                     :func:`def_obj_func`
    :return: the rotation_matrix, the row_names, the column_names, 
             the rotated_experiments, and the boxes found by prim              
                    
    
    '''
    orig_experiments, outcomes = results
    
    #transform experiments to numpy array
    dtypes = orig_experiments.dtype.fields
    object_dtypes = [key for key, value in dtypes.items() if value[0]==np.dtype(object)]
    
    #get experiments of interest
    logical = classify(outcomes)==1
    
    # if no subsets are provided all uncertainties with non dtype object are
    # in the same subset, the name of this is r, for rotation
    if not subsets:
#        non_object_dtypes = [key for key, value in dtypes.items() if value[0].name!=np.dtype(object)]
        subsets = {"r":[key for key, value in dtypes.items() if value[0].name!=np.dtype(object)]}
    
    # remove uncertainties that are in exclude and check whether uncertainties
    # occur in more then one subset
    seen = set()
    for key, value in subsets.items():
        value = set(value) - set(exclude)
        subsets[key] = list(value)
        if (seen & value):
            raise EMAError("uncertainty occurs in more then one subset")
        else:
            seen = seen | set(value)
    
    #prepare the dtypes for the new rotated experiments recarray
    new_dtypes = []
    for key, value in subsets.items():
        assert_dtypes(value, dtypes)
        
        # the names of the rotated columns are based on the group name 
        # and an index
        [new_dtypes.append(("%s_%s" % (key, i), float)) for i in range(len(value))]
    
    #add the uncertainties with object dtypes to the end
    included_object_dtypes = set(object_dtypes)-set(exclude)
    [new_dtypes.append((name, object)) for name in included_object_dtypes ]
    
    #make a new empty recarray
    rotated_experiments = np.recarray((orig_experiments.shape[0],),dtype=new_dtypes)
    
    #put the uncertainties with object dtypes already into the new recarray 
    for name in included_object_dtypes :
        rotated_experiments[name] = orig_experiments[name]
    
    #iterate over the subsets, rotate them, and put them into the new recarray
    shape = 0
    for key, value in subsets.items():
        shape += len(value) 
    rotation_matrix = np.zeros((shape,shape))
    column_names = []
    row_names = []
    
    j = 0
    for key, value in subsets.items():
        subset_rotation_matrix, subset_experiments = rotate_subset(value, orig_experiments, logical)
        rotation_matrix[j:j+len(value), j:j+len(value)] = subset_rotation_matrix
        [row_names.append(entry) for entry in value]
        j += len(value)
        
        for i in range(len(value)):
            name = "%s_%s" % (key, i)
            rotated_experiments[name] = subset_experiments[:,i]
            [column_names.append(name)]
    
    results = rotated_experiments, outcomes
    
    #perform prim in the usual way
    
    
    boxes = prim.perform_prim(results, 
                             classify, 
                             peel_alpha=peel_alpha, 
                             paste_alpha=paste_alpha,
                             mass_min=mass_min, 
                             threshold=threshold, 
                             pasting=pasting, 
                             threshold_type=threshold_type,
                             obj_func=obj_func)
    
    return rotation_matrix, row_names, column_names, rotated_experiments, boxes



#
# TODO the code below should be turned into an example of pca prim 
#
#if __name__ =='__main__':
#    ema_logging.log_to_stderr(ema_logging.INFO)
#    
#    def copper_crises_rule(outcomes):
#        '''
#        classify results into crises or not
#        
#        rule is change in real copper price larger than factor 2
#        '''
#        outcome = outcomes['Real price of copper']
#        change = np.abs(outcome[:, 1::]/outcome[:, 0:-1])
#        classes = np.zeros(outcome.shape[0])
#        classes[np.max(change, axis=1)>2] = 1
#        return classes
#    
#    def restrict_to_after_2010(results):
#        logical =  results[1]['TIME'][0,:]>2010
#        for key, value in results[1].items():
#            value = value.T[logical]
#            value = value.T
#            results[1][key] = value
#        return results
#
#    top_down_subsets = {'Capacity': ['Initial mining capacity in preparation',    
#                                     'Initial refining capacity',    
#                                     'Average smelting capacity permit term',    
#                                     'Average mine lifetime',    
#                                     'Initial mining capacity'],    
#                        'Capacity (development)': ['Initial copper price',    
#                                                   'Long term profit period',    
#                                                   'Maximum increase deep sea mine capacity',    
#                                                   'Deep sea capital investment fraction',    
#                                                   'Initial long term profit forecast'],    
#                        'Demand': ['Normalisation value GDP',    
#                                   'Start GDP per capita',    
#                                   'Long term copper price elasticity',    
#                                   'Amplification factor of relative price effect',    
#                                   'Amplification factor of intrinsic demand',    
#                                   'Initial value long term effect intrinsic demand',    
#                                   'Base economic growth',    
#                                   'Long term increase demand due to intrinsic demand'],    
#                        'Marginal costs':['Influence of technology on copper mining energy costs',    
#                                          'Power for oregrades'],    
#                        'Price':['Price amplifying factor',    
#                                 'Production time'],    
#                        'Recycling/Supply':['Copper score during treatment',    
#                                            'Percentage of primary scrap'],    
#                        'Substitute':['GDP growth difference amplifier',    
#                                      'Initial value of long term effect substitution'],    
#                        'Supply':['Switch forecast capacity',    
#                                  'One year',    
#                                  'Initial inventories of refined copper',    
#                                  'Minimum usage smelting and refining capacity']}    
#
#    bottom_up_subsets = {'Capacity':['Average mine lifetime',
#                                     'Initial used deep sea capacity',
#                                     'Initial used mining capacity',
#                                     'Average mine permit term',
#                                     'Initial deep sea mining capacity',
#                                     'Average lifetime of copper mills',
#                                     'Average lifetime of deep sea capacity',
##                                     'Delay order deep sea',
#                                     'Initial deep sea capacity in preparation',
#                                     'Initial preparation of refining capacity'],
#                        'Capacity (development)':['Maximum increase deep sea mine capacity',
#                                                  'Maximum increase mine and smelter capacity',
#                                                  'Initial deep sea long term profit',
#                                                  'Production costs deep sea copper',
#                                                  'Capital investment fraction',
#                                                  'Switch forecast and historical data for capacity',
#                                                  'Initial long term profit forecast'],
#                        'Demand':['Long term increase demand due to intrinsic demand',
#                                  'Amplification factor of relative price effect',
#                                  'Initial value long term effect price',
#                                  'Start GDP per capita',
#                                  'Base economic growth',
#                                  'Long term effect on demand period',
#                                  'Cu in vehicles BEV',
#                                  'Cu in vehicles cityBEV',
#                                  'Amplification factor of intrinsic demand',
#                                  'Number of cars in 2000',
#                                  'Goal for copper in infrastructure 2050',
#                                  'Growth of copper in architecture',
#                                  'Switch new cars',
#                                  'Long term effect intrinsic demand period',
#                                  'Long term copper price elasticity',
#                                  'Switch World population'],
#                        'Marginal costs':['Transport costs of copper',
#                                          'GDP growth difference amplifier energy'],
#                        'Price':['Price amplifying factor',
#                                 'Production time'],
#                        'Recycling':['Initial use grade for WaterTreatment',
#                                     'Initial value global copper in scrap StationaryElectromotors',
#                                     'Average lifetime of copper in use Architecture',
#                                     'Percentage copper recovered from scrap',
#                                     'Average lifetime of copper in use OtherUse',
#                                     'Collection rate copper products Automotive',
#                                     'Collection rate copper products WaterTreatment',
#                                     'Copper score during treatment EnergyInfrastructure',
#                                     'Initial value global copper in scrap OtherUse',
#                                     'Initial total copper in use',
#                                     'Copper score during treatment WaterTreatment'],
#                        'Substitute':['Long term effect of substitution period',
#                                      'Substitution threshold StationaryElectromotors',
#                                      'Substitution threshold OtherUse',
#                                      'Initial price of aluminium',
#                                      'Amplification factor for substitution'],
#                        'Supply':['Part of resource base seabased',
#                                  'One year',
#                                  'Minimum usage smelting and refining capacity',
#                                  'Threshold for junior companies to start deep sea reserve base development']}
#
#
#
#    #retrieve results
#    results = load_results(r'..\..\models\JAN\pickles\copper revision\\result_2500_runs_TopDown_31_uncertainties.cPickle')
#    results = restrict_to_after_2010(results)
#    rotation_matrix, row_names, column_names, boxes = perform_pca_prim(results, 
#                              classify=copper_crises_rule,
#                              exclude=['policy', 'model'])
#    rotation_matrix, row_names, column_names, boxes = perform_pca_prim(results, 
#                              classify=copper_crises_rule,
#                              exclude=['policy', 'model', 'Delay order deep sea'],
#                              subsets=bottom_up_subsets,
#                              threshold=0.7)    
#   
#    results = load_results(r'..\..\models\JAN\pickles\copper revision\\result_2500_runs_BottomUp_57_uncertainties.cPickle')
#    results = restrict_to_after_2010(results)
#    rotation_matrix, row_names, column_names, boxes = perform_pca_prim(results, 
#                              classify=copper_crises_rule,
#                              exclude=['policy', 'model','Delay order deep sea'])
#    rotation_matrix, row_names, column_names, boxes = perform_pca_prim(results, 
#                              classify=copper_crises_rule,
#                              exclude=['policy', 'model', 'Delay order deep sea'],
#                              subsets=bottom_up_subsets,
#                              threshold=0.7)    
#    
#    results = load_results(r'..\..\models\JAN\pickles\copper revision\results_5000_bothModels.cPickle')
#    results = restrict_to_after_2010(results)
#    
#    # perform PCA prim
#
#    for entry in row_names:
#        print entry
#        
#    for entry in column_names:
#        print entry
#    np.savetxt('rotation matrix.txt', rotation_matrix, delimiter=',')
#    
#    import pylab as p
#    p.matshow(rotation_matrix)
#    p.show()
#    
#
#     perform prim in the usual way unrotated for comparison
#    boxes2 = prim.perform_prim(results, 
#                      copper_crises_rule, 
#                      threshold=0.8)
#    prim.show_boxes_together(boxes2, results)
#    prim.write_prim_to_stdout(boxes2)
#    
#    plt.show()
