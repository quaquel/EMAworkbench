'''
Created on Mar 22, 2014

@author: jhkwakkel
'''
import os

import numpy.lib.recfunctions as rf

from matplotlib.mlab import csv2rec
import numpy as np
from expWorkbench import TIME, ema_logging

def load_flu_data():
    path = os.path.dirname(__file__)
    
    fn = './data/flu/experiments.csv'
    fn = os.path.join(path, fn)
    experiments = csv2rec(fn)
    outcomes = {}
        
    dt_descr = experiments.dtype.descr
    dt_descr[-1] = (dt_descr[-1][0], 'object') #transform policy dtype to object
    dt_descr[-2] = (dt_descr[-2][0], 'object') #transform model dtype to object
    dt = np.dtype(dt_descr)
    experiments = experiments.astype(dt)
  
    fn = './data/flu/deceased population region 1.csv'
    fn = os.path.join(path, fn)    
    outcomes['deceased population region 1'] = np.loadtxt(fn, delimiter=',')
    
    fn = './data/flu/time.csv'
    fn = os.path.join(path, fn)
    outcomes[TIME] = np.loadtxt(fn, delimiter=',')

    fn = './data/flu/infected fraction R1.csv'
    fn = os.path.join(path, fn)
    outcomes['infected fraction R1'] = np.loadtxt(fn, delimiter=',')
    
    return experiments, outcomes

def load_scarcity_data():
    path = os.path.dirname(__file__)
    
    fn = './data/scarcity/experiments.csv'
    fn = os.path.join(path, fn)
    experiments = csv2rec(fn)
    outcomes = {}
        
    dt_descr = [('absolute recycling loss fraction', '<f8'),
                ('average construction time extraction capacity', '<f8'),
                ('average lifetime extraction capacity', '<f8'),
                ('average lifetime recycling capacity', '<f8'),
                ('exogenously planned extraction capacity', '<f8'),
                ('fraction of maximum extraction capacity used', '<f8'),
                ('initial annual supply', '<f8'),
                ('initial average recycling cost', '<f8'),
                ('initial extraction capacity under construction', '<f8'),
                ('initial in goods', '<f8'),
                ('initial recycling capacity under construction', '<f8'),
                ('initial recycling infrastructure', '<f8'),
                ('lookup approximated learning scale', '<f8'),
                ('lookup approximated learning speed', '<f8'),
                ('lookup approximated learning start', '<f8'),
                ('lookup price substitute begin', '<f8'),
                ('lookup price substitute end', '<f8'),
                ('lookup price substitute speed', '<f8'),
                ('lookup returns to scale scale', '<f8'),
                ('lookup returns to scale speed', '<f8'),
                ('lookup shortage normalize', '<i4'),
                ('lookup shortage speed', '<f8'),
                ('normal profit margin', '<f8'),
                ('order extraction capacity delay', '|O4'),
                ('order in goods delay', '|O4'),
                ('order recycling capacity delay', '|O4'),
                ('price elasticity of demand', '<f8'),
                ('model', '|O4'),
                ('policy', '|O4')]
    
    
    dt = np.dtype(dt_descr)
    experiments = experiments.astype(dt)
  
    fn = './data/scarcity/relative_market_price.csv'
    fn = os.path.join(path, fn)    
    outcomes['relative market price'] = np.loadtxt(fn, delimiter=',')
    
    fn = './data/scarcity/time.csv'
    fn = os.path.join(path, fn)
    outcomes[TIME] = np.loadtxt(fn, delimiter=',')
    
    return experiments, outcomes

def load_eng_trans_data():
    path = os.path.dirname(__file__)
    
    fn = './data/eng_trans/experiments.csv'
    fn = os.path.join(path, fn)
    experiments = csv2rec(fn)
    outcomes = {}
        
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
    
    
    dt = np.dtype(dt_descr)
    names = [entry[0] for entry in dt_descr]
    experiments.dtype.names = names
    
    experiments = experiments.astype(dt)
  
    fn = './data/eng_trans/total capacity installed.csv'
    fn = os.path.join(path, fn)    
    outcomes['total capacity installed'] = np.loadtxt(fn, delimiter=',')
    
    fn = './data/eng_trans/time.csv'
    fn = os.path.join(path, fn)
    outcomes[TIME] = np.loadtxt(fn, delimiter=',')

    fn = './data/eng_trans/total fraction new technologies.csv'
    fn = os.path.join(path, fn)
    outcomes['total fraction new technologies'] = np.loadtxt(fn, delimiter=',')
    
    return experiments, outcomes
  
if __name__ == '__main__':
    load_flu_data()
    load_scarcity_data()
    load_eng_trans_data()