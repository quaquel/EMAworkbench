'''
Created on Mar 22, 2014

@author: jhkwakkel
'''
import os
import zipfile
import StringIO

import numpy.lib.recfunctions as rf

from matplotlib.mlab import csv2rec
import numpy as np
from expWorkbench import TIME, ema_logging
from expWorkbench.util import load_results

def load_flu_data():
    path = os.path.dirname(__file__)
    fn = './data/1000 flu cases no policy.tar.gz'
    fn = os.path.join(path, fn)

    experiments, outcomes = load_results(fn)
    return experiments, outcomes

def load_scarcity_data():
    path = os.path.dirname(__file__)
    fn = './data/1000 runs scarcity.tar.gz'
    fn = os.path.join(path, fn)

    experiments, outcomes = load_results(fn)
    return experiments, outcomes
    
    return experiments, outcomes

def load_eng_trans_data():

    dt_descr = [('ini PR T4', '<f8'),
                ('lifetime T1', '<f8'),
                ('lifetime T2', '<f8'),
                ('lifetime T3', '<f8'),
                ('lifetime T4', '<f8'),
                ('ec gr t1', '<f8'),
                ('ec gr t2', '<f8'),
                ('ec gr t3', '<f8'),
                ('ec gr t4', '<f8'),
                ('ec gr t5', '<f8'),
                ('ec gr t6', '<f8'),
                ('ec gr t7', '<f8'),
                ('ec gr t8', '<f8'),
                ('ec gr t9', '<f8'),
                ('ec gr t10', '<f8'),
                ('random PR min', '<f8'),
                ('random PR max', '<f8'),
                ('seed PR T1', '<i4'),
                ('seed PR T2', '<i4'),
                ('seed PR T3', '<i4'),
                ('seed PR T4', '<i4'),
                ('absolute preference for MIC', '<f8'),
                ('absolute preference for expected cost per MWe', '<f8'),
                ('absolute preference against unknown', '<f8'),
                ('absolute preference for expected progress', '<f8'),
                ('absolute preference against specific CO2 emissions', '<f8'),
                ('performance expected cost per MWe T1', '<f8'),
                ('performance expected cost per MWe T2', '<f8'),
                ('performance expected cost per MWe T3', '<f8'),
                ('performance expected cost per MWe T4', '<f8'),
                ('performance CO2 avoidance T1', '<f8'),
                ('performance CO2 avoidance T2', '<f8'),
                ('performance CO2 avoidance T3', '<f8'),
                ('performance CO2 avoidance T4', '<f8'),
                ('SWITCH T3', '|O4'),
                ('SWITCH T4', '|O4'),
                ('preference switches', '|O4'),
                ('ini cap T1', '<f8'),
                ('ini cap T2', '<f8'),
                ('ini cap T3', '<f8'),
                ('ini cap T4', '<f8'),
                ('ini cost T1', '<f8'),
                ('ini cost T2', '<f8'),
                ('ini cost T3', '<f8'),
                ('ini cost T4', '<f8'),
                ('ini cum decom cap T1', '<f8'),
                ('ini cum decom cap T2', '<f8'),
                ('ini cum decom cap T3', '<f8'),
                ('ini cum decom cap T4', '<f8'),
                ('average planning and construction period T1', '<f8'),
                ('average planning and construction period T2', '<f8'),
                ('average planning and construction period T3', '<f8'),
                ('average planning and construction period T4', '<f8'),
                ('ini PR T1', '<f8'),
                ('ini PR T2', '<f8'),
                ('ini PR T3', '<f8'),
                ('model', '|O4'),
                ('policy', '|O4')]
    
    path = os.path.dirname(__file__)
    fn = './data/eng_trans.zip'
    fn = os.path.join(path, fn)
    outcomes = {}

    with zipfile.ZipFile(fn) as z:
        experiments = StringIO.StringIO(z.read('x.csv'))
        experiments = csv2rec(experiments)
        dt = np.dtype(dt_descr)
        names = [entry[0] for entry in dt_descr]
        experiments.dtype.names = names
        experiments = experiments.astype(dt)
       
        data = StringIO.StringIO(z.read('total capacity installed.csv'))
        outcomes['total capacity installed'] = np.loadtxt(data, delimiter=',')
         
        data = StringIO.StringIO(z.read('TIME.csv'))
        outcomes[TIME] = np.loadtxt(data, delimiter=',')
      
        data = StringIO.StringIO(z.read('total fraction new technologies.csv'))
        outcomes['total fraction new technologies'] = np.loadtxt(data, 
                                                                 delimiter=',')
    
    return experiments, outcomes
  
if __name__ == '__main__':
    load_flu_data()
    load_scarcity_data()
    load_eng_trans_data()