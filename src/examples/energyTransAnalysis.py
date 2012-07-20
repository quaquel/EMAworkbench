'''
Created on 29 sep. 2011

@author: jhkwakkel
'''
import numpy as np
import matplotlib.pyplot as plt

from expWorkbench import load_results
from analysis import prim
from expWorkbench import EMAlogging
from analysis.graphs import envelopes
EMAlogging.log_to_stderr(EMAlogging.INFO) 


results  = load_results(r'C:\workspace\EMA-workbench\models\TFSC_all_policies.cPickle')
envelopes(results, 
          column='policy', 
          categories=['adaptive policy',
                      'ap with op'])

#exp, res = results
#
##get out only the results related to the last policy
#exp, res = results
#
#logical = exp['policy'] == 'adaptive policy'
#exp = exp[logical]
#
#temp_res = {}
#for key, value in res.items():
#    temp_res[key] = value[logical]
#res = temp_res
#
#results = (exp, res)

#start prim specification 
def classify(data):
    #get the output for deceased population
    result = data['total fraction new technologies']
    
    #make an empty array of length equal to number of cases 
    classes =  np.zeros(result.shape[0])
    
    #if deceased population is higher then 1.000.000 people, classify as 1 
    classes[result[:, -1] >= 0.65] = 1
    
    return classes

#prims, uncertainties, experiments = prim.perform_prim(results, 
#                                                     classify, 
#                                                     threshold=0.75, 
#                                                     threshold_type=1)

#prims, uncertainties, experiments = prim.perform_prim(results, 
#                                                     'total fraction new technologies', 
#                                                     threshold=0.6, 
#                                                     threshold_type=-1)

uv = ["average planning and construction period T1",
      "average planning and construction period T2",
      "average planning and construction period T3",
      "average planning and construction period T4",
      "lifetime T1",
      "lifetime T2",
      "lifetime T3",
      "lifetime T4"]

#figure = prim.visualize_prim(prims, 
#                             uncertainties, 
#                             experiments, 
#                             filter=True,
#                             uv=uv
#                             )
plt.show()
